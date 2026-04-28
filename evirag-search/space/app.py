"""
EVIRAG Search Backend — HF Space FastAPI app.

On startup:
  1. Downloads evirag.index + ids.npy + metadata.parquet from HF Datasets
  2. Loads FAISS index into RAM
  3. Opens metadata.parquet via DuckDB (disk-based, no full RAM load)

Query flow:
  POST /search  →  embed query (SPECTER) → binary quantize → FAISS IVF search
                →  look up paper metadata → return JSON

Endpoints:
  GET  /health   — always 200 (Space keepalive target)
  GET  /status   — loading state, index stats
  POST /search   — main search endpoint
"""

import os
import threading
import time
import numpy as np
import faiss
import duckdb
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATASET_ID  = "amethystani/evirag-index"
DATA_DIR    = Path("/data")
INDEX_PATH  = DATA_DIR / "evirag.index"
IDS_PATH    = DATA_DIR / "ids.npy"
META_PATH   = DATA_DIR / "metadata.parquet"
CORPUS_PATH = DATA_DIR / "corpus.parquet"     # ≤4000-char peS2o full_text
MODEL_NAME  = "allenai-specter"
DIM         = 768
NPROBE      = 128     # clusters to search (higher = better recall, slower)
MAX_K       = 200     # max results a caller can request

# ── Global state ──────────────────────────────────────────────────────────────
_state = {
    "status":    "initializing",
    "index":     None,    # faiss.IndexBinaryIVF
    "ids":       None,    # np.ndarray of str
    "model":     None,    # SentenceTransformer
    "db":        None,    # duckdb.DuckDBPyConnection
    "n_vectors": 0,
    "error":     None,
    "load_start": time.time(),
}


# ── Startup loader (runs in background thread) ────────────────────────────────
def _load():
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Load embedding model first (fast, ~500MB)
        _state["status"] = "loading_model"
        print("Loading SPECTER model...")
        _state["model"] = SentenceTransformer(MODEL_NAME)
        print("Model ready.")

        # 2. Download files from HF Datasets (skip if already cached on disk)
        # corpus.parquet (full_text) is OPTIONAL — search still works without it.
        required_files = [
            ("evirag.index",     INDEX_PATH,   True),
            ("ids.npy",          IDS_PATH,     True),
            ("metadata.parquet", META_PATH,    True),
            ("corpus.parquet",   CORPUS_PATH,  False),   # optional: full_text
        ]
        for filename, dest, required in required_files:
            if dest.exists():
                print(f"Using cached {filename}")
                continue
            _state["status"] = f"downloading_{filename}"
            print(f"Downloading {filename} from HF...")
            try:
                local = hf_hub_download(
                    repo_id=DATASET_ID,
                    filename=filename,
                    repo_type="dataset",
                    token=HF_TOKEN or None,
                    local_dir=str(DATA_DIR),
                    local_dir_use_symlinks=False,
                )
                print(f"  → {local}")
            except Exception as exc:
                if required:
                    raise
                print(f"  (optional file '{filename}' not on HF yet — skipping: {exc})")

        # 3. Load FAISS index into RAM
        _state["status"] = "loading_index"
        print("Loading FAISS index into RAM...")
        t0 = time.time()
        index = faiss.read_index_binary(str(INDEX_PATH))
        index.nprobe = NPROBE
        _state["index"]     = index
        _state["n_vectors"] = index.ntotal
        print(f"  Index loaded: {index.ntotal:,} vectors in {time.time()-t0:.1f}s")

        # 4. Load paper ID array
        _state["status"] = "loading_ids"
        print("Loading paper IDs...")
        _state["ids"] = np.load(str(IDS_PATH), allow_pickle=True)
        print(f"  {len(_state['ids']):,} IDs loaded")

        # 5. Open metadata + corpus Parquet via DuckDB (both stay on disk).
        # corpus.parquet (≤4000-char full_text) JOINed at query time when present.
        _state["status"] = "loading_metadata"
        print("Opening metadata via DuckDB...")
        conn = duckdb.connect(database=":memory:")
        conn.execute(f"CREATE VIEW papers AS SELECT * FROM read_parquet('{META_PATH}')")
        if CORPUS_PATH.exists():
            print("Opening corpus parquet (full_text) via DuckDB...")
            conn.execute(f"CREATE VIEW corpus AS SELECT id, full_text FROM read_parquet('{CORPUS_PATH}')")
            _state["has_full_text"] = True
            print("  Full-text JOIN view ready.")
        else:
            _state["has_full_text"] = False
            print("  (no corpus.parquet — full_text disabled)")
        _state["db"] = conn
        print("  Metadata view ready.")

        _state["status"] = "ready"
        elapsed = time.time() - _state["load_start"]
        print(f"\nReady in {elapsed/60:.1f} min. Serving queries.")

    except Exception as e:
        _state["status"] = "error"
        _state["error"]  = str(e)
        print(f"LOAD ERROR: {e}")
        raise


# ── FastAPI setup ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=_load, daemon=True)
    thread.start()
    yield

app = FastAPI(title="EVIRAG Search", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    k:     int = Field(default=20, ge=1, le=MAX_K)


class Paper(BaseModel):
    rank:          int
    openalex_id:   str
    title:         str
    year:          int | None
    doi:           str | None
    cited_by_count: int | None
    source:        str | None
    score:         float   # Hamming distance (lower = more similar)
    text_excerpt:  str | None = None
    full_text:     str | None = None    # ≤4000-char peS2o body text


class SearchResponse(BaseModel):
    query:        str
    results:      list[Paper]
    latency_ms:   float
    index_size:   int


# ── Helpers ───────────────────────────────────────────────────────────────────
def _embed_query(text: str) -> np.ndarray:
    """Returns binary vector (1, 96) uint8."""
    emb = _state["model"].encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    binary = np.packbits((emb > 0).astype(np.uint8), axis=1)
    return np.ascontiguousarray(binary)


def _fetch_metadata(openalex_ids: list[str]) -> dict[str, dict]:
    """JOIN papers ⨝ corpus to fetch full_text (≤4000 chars) when available."""
    if not openalex_ids or _state["db"] is None:
        return {}
    ids_sql = ", ".join(f"'{i}'" for i in openalex_ids)
    if _state.get("has_full_text"):
        rows = _state["db"].execute(
            f"SELECT p.id, p.title, p.year, p.doi, p.cited_by_count, p.source, "
            f"       p.text_excerpt, c.full_text "
            f"FROM papers p LEFT JOIN corpus c ON p.id = c.id "
            f"WHERE p.id IN ({ids_sql})"
        ).fetchall()
        return {
            r[0]: {
                "title":          r[1],
                "year":           r[2],
                "doi":            r[3],
                "cited_by_count": r[4],
                "source":         r[5],
                "text_excerpt":   r[6] or "",
                "full_text":      r[7] or "",
            }
            for r in rows
        }
    rows = _state["db"].execute(
        f"SELECT id, title, year, doi, cited_by_count, source, text_excerpt "
        f"FROM papers WHERE id IN ({ids_sql})"
    ).fetchall()
    return {
        r[0]: {
            "title": r[1],
            "year": r[2],
            "doi": r[3],
            "cited_by_count": r[4],
            "source": r[5],
            "text_excerpt": r[6] or "",
            "full_text":    "",
        }
        for r in rows
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Always 200 — used by keep-warm ping."""
    return {"ok": True}


@app.get("/status")
def status():
    return {
        "status":    _state["status"],
        "n_vectors": _state["n_vectors"],
        "error":     _state["error"],
        "uptime_s":  round(time.time() - _state["load_start"], 1),
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if _state["status"] != "ready":
        raise HTTPException(
            status_code=503,
            detail=f"Index not ready yet. Status: {_state['status']}",
        )

    t0 = time.perf_counter()

    # Embed + binary quantize
    query_bin = _embed_query(req.query)

    # FAISS search
    k_fetch = min(req.k * 2, MAX_K)   # fetch 2x, filter dups/missing metadata
    D, I    = _state["index"].search(query_bin, k_fetch)

    distances = D[0].tolist()
    indices   = I[0].tolist()

    # Map FAISS int indices → OpenAlex IDs
    ids_arr   = _state["ids"]
    raw_pairs = [
        (ids_arr[i], dist)
        for i, dist in zip(indices, distances)
        if 0 <= i < len(ids_arr)
    ]

    openalex_ids = [p[0] for p in raw_pairs]
    meta_map     = _fetch_metadata(openalex_ids)

    results = []
    for rank, (oa_id, dist) in enumerate(raw_pairs):
        if len(results) >= req.k:
            break
        meta = meta_map.get(oa_id, {})
        results.append(Paper(
            rank=rank + 1,
            openalex_id=oa_id,
            title=meta.get("title") or "",
            year=meta.get("year"),
            doi=meta.get("doi"),
            cited_by_count=meta.get("cited_by_count"),
            source=meta.get("source"),
            score=float(dist),
            text_excerpt=meta.get("text_excerpt") or None,
            full_text=meta.get("full_text") or None,
        ))

    latency_ms = (time.perf_counter() - t0) * 1000

    return SearchResponse(
        query=req.query,
        results=results,
        latency_ms=round(latency_ms, 2),
        index_size=_state["n_vectors"],
    )
