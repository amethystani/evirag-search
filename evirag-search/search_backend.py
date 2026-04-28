"""
EVIRAG Search Backend — Local FastAPI service
Serves the 100k peS2o paper index from Hugging Face with sub-100ms latency.

Startup:
  Downloads evirag.index + ids.npy + metadata.parquet from HF Datasets
  (cached locally, skipped on subsequent runs)

Query flow (sub-100ms):
  POST /search → embed query (SPECTER, MPS/CUDA/CPU) → binary quantize →
  FAISS IVF Hamming search → DuckDB parquet lookup → return JSON

Run:
  python3 search_backend.py
  or
  uvicorn search_backend:app --port 7860 --reload
"""

import os
import sys
import time
import threading
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import faiss
import duckdb
import torch
import torch.nn.functional as F
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
DATASET_ID = "amethystani/evirag-index"
CACHE_DIR  = Path(__file__).parent / "local_100k"   # reuse already-downloaded files
CACHE_DIR.mkdir(exist_ok=True)

INDEX_PATH   = CACHE_DIR / "evirag.index"
IDS_PATH     = CACHE_DIR / "ids.npy"
META_PATH    = CACHE_DIR / "metadata.parquet"
CORPUS_PATH  = CACHE_DIR / "corpus.parquet"     # holds full_text (≤4000 chars)

NPROBE = 64      # clusters to search — 64 gives sub-10ms on 100k, use 128 for 8M
MAX_K  = 50      # max results per query
PORT   = 7860

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Search device: {DEVICE}")

# ── Global state ──────────────────────────────────────────────────────────────
_state = {
    "status":     "initializing",
    "index":      None,   # faiss.IndexBinaryIVF (in RAM)
    "ids":        None,   # np.ndarray[str]
    "tokenizer":  None,
    "model":      None,
    "db":         None,   # duckdb connection
    "n_vectors":  0,
    "error":      None,
    "load_start": time.time(),
}


# ── Background loader ─────────────────────────────────────────────────────────
def _load():
    try:
        # 1. Download files from HF (skip if cached)
        for fname in ["evirag.index", "ids.npy", "metadata.parquet"]:
            dest = CACHE_DIR / fname
            if dest.exists():
                size_mb = dest.stat().st_size / 1e6
                print(f"  Using cached {fname} ({size_mb:.0f} MB)")
                continue
            _state["status"] = f"downloading_{fname}"
            print(f"  Downloading {fname} from HF Datasets...")
            hf_hub_download(
                repo_id=DATASET_ID,
                filename=fname,
                repo_type="dataset",
                token=HF_TOKEN or None,
                local_dir=str(CACHE_DIR),
                local_dir_use_symlinks=False,
            )
            print(f"  ✓ {fname}")

        # 2. Load SPECTER tokenizer + model
        _state["status"] = "loading_model"
        print("Loading SPECTER model...")
        tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
        # Always load in float32 first, then cast on CUDA only
        model = AutoModel.from_pretrained("allenai/specter").eval()
        if DEVICE.type == "cuda":
            model = model.to(dtype=torch.bfloat16).to(DEVICE)
            print(f"  ✓ SPECTER (bfloat16 on {DEVICE})")
        else:
            model = model.to(DEVICE)
            print(f"  ✓ SPECTER (float32 on {DEVICE})")

        _state["tokenizer"] = tokenizer
        _state["model"]     = model

        # 3. Load FAISS index into RAM
        _state["status"] = "loading_index"
        print("Loading FAISS index...")
        t0    = time.time()
        index = faiss.read_index_binary(str(INDEX_PATH))
        index.nprobe = NPROBE
        _state["index"]     = index
        _state["n_vectors"] = index.ntotal
        print(f"  ✓ {index.ntotal:,} vectors loaded in {time.time()-t0:.1f}s")

        # 4. Load IDs
        _state["ids"] = np.load(str(IDS_PATH), allow_pickle=True)
        print(f"  ✓ {len(_state['ids']):,} paper IDs")

        # 5. Open metadata + corpus (full_text) via DuckDB
        # JOIN strategy: papers (lightweight) ⨝ corpus (full_text, ≤4000 chars per paper)
        # corpus.parquet is optional — if absent, we fall back to text_excerpt only.
        _state["status"] = "loading_metadata"
        print("Opening metadata parquet via DuckDB...")
        conn = duckdb.connect(database=":memory:")
        conn.execute(f"CREATE VIEW papers AS SELECT * FROM read_parquet('{META_PATH}')")
        count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        print(f"  ✓ {count:,} papers in metadata view")

        if CORPUS_PATH.exists():
            print(f"Opening corpus parquet (full_text) via DuckDB...")
            conn.execute(f"CREATE VIEW corpus AS SELECT id, full_text FROM read_parquet('{CORPUS_PATH}')")
            ccount = conn.execute("SELECT COUNT(*) FROM corpus").fetchone()[0]
            print(f"  ✓ {ccount:,} papers with full_text (≤4000 chars each)")
            _state["has_full_text"] = True
        else:
            print(f"  (no corpus.parquet — full_text disabled, fallback to text_excerpt)")
            _state["has_full_text"] = False

        _state["db"] = conn

        # 6. Warm-up pass — embed one query so first real request is fast
        _state["status"] = "warming_up"
        _embed_query("machine learning")
        print("  ✓ Model warmed up")

        _state["status"]  = "ready"
        elapsed = time.time() - _state["load_start"]
        print(f"\n✅ Ready in {elapsed:.1f}s — serving on http://localhost:{PORT}\n")

    except Exception as e:
        _state["status"] = "error"
        _state["error"]  = str(e)
        print(f"LOAD ERROR: {e}")
        raise


# ── Embedding ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def _embed_query(text: str) -> np.ndarray:
    """Embed one query → binary uint8 vector (1, 96)."""
    tokenizer = _state["tokenizer"]
    model     = _state["model"]
    enc  = tokenizer([text], max_length=512, padding=True,
                     truncation=True, return_tensors="pt")
    ids  = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)

    if DEVICE.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, attention_mask=mask)
    else:
        out = model(input_ids=ids, attention_mask=mask)

    # Mean pool weighted by attention mask
    emb = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) \
          / mask.sum(1, keepdim=True).clamp(min=1)
    emb = F.normalize(emb, p=2, dim=1)

    bits   = (emb > 0).to(torch.uint8).cpu().numpy()
    binary = np.packbits(bits, axis=1)
    return np.ascontiguousarray(binary)


def _fetch_metadata(paper_ids: list[str]) -> dict[str, dict]:
    """DuckDB lookup for a list of paper IDs — sub-millisecond.

    JOINs corpus.parquet on id to fetch the full ≤4000-char body text
    when available. Falls back to text_excerpt for backwards compatibility.
    """
    if not paper_ids or _state["db"] is None:
        return {}
    ids_sql = ", ".join(f"'{i}'" for i in paper_ids)
    has_ft  = _state.get("has_full_text", False)

    try:
        if has_ft:
            # JOIN papers ⨝ corpus on id — pulls in the full_text column
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
                "title":          r[1],
                "year":           r[2],
                "doi":            r[3],
                "cited_by_count": r[4],
                "source":         r[5],
                "text_excerpt":   r[6] or "",
                "full_text":      "",
            }
            for r in rows
        }
    except Exception:
        rows = _state["db"].execute(
            f"SELECT id, title, year, doi, cited_by_count, source "
            f"FROM papers WHERE id IN ({ids_sql})"
        ).fetchall()
        return {
            r[0]: {
                "title":          r[1],
                "year":           r[2],
                "doi":            r[3],
                "cited_by_count": r[4],
                "source":         r[5],
                "text_excerpt":   "",
                "full_text":      "",
            }
            for r in rows
        }


# ── FastAPI ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=_load, daemon=True)
    thread.start()
    yield


app = FastAPI(
    title="EVIRAG Search",
    description="Sub-100ms semantic search over 100k scientific papers",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    k:     int = Field(default=20, ge=1, le=MAX_K)


class Paper(BaseModel):
    rank:           int
    id:             str
    title:          str
    year:           Optional[int]
    doi:            Optional[str]
    cited_by_count: Optional[int]
    source:         Optional[str]
    score:          float   # Hamming distance — lower = more similar
    text_excerpt:   Optional[str] = None
    full_text:      Optional[str] = None      # ≤4000-char peS2o body text


class SearchResponse(BaseModel):
    query:       str
    results:     list[Paper]
    latency_ms:  float
    index_size:  int
    status:      str = "ok"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True, "status": _state["status"]}


@app.get("/status")
def status():
    return {
        "status":    _state["status"],
        "n_vectors": _state["n_vectors"],
        "error":     _state["error"],
        "uptime_s":  round(time.time() - _state["load_start"], 1),
        "device":    str(DEVICE),
    }


@app.post("/search", response_model=SearchResponse)
@app.post("/api/search", response_model=SearchResponse)   # also mount at /api/search for frontend
def search(req: SearchRequest):
    if _state["status"] != "ready":
        raise HTTPException(
            status_code=503,
            detail=f"Index not ready yet — status: {_state['status']}",
        )

    t0 = time.perf_counter()

    # --- 1. Embed query (~20-80ms depending on device)
    query_bin = _embed_query(req.query)
    t_embed = (time.perf_counter() - t0) * 1000

    # --- 2. FAISS binary search (~1-5ms for 100k, ~10-30ms for 8M)
    t1   = time.perf_counter()
    k_fetch = min(req.k * 3, MAX_K)
    D, I = _state["index"].search(query_bin, k_fetch)
    t_faiss = (time.perf_counter() - t1) * 1000

    distances = D[0].tolist()
    indices   = I[0].tolist()
    ids_arr   = _state["ids"]

    raw_pairs = [
        (ids_arr[i], dist)
        for i, dist in zip(indices, distances)
        if 0 <= i < len(ids_arr)
    ]

    # --- 3. Metadata lookup via DuckDB (~0.5ms)
    t2       = time.perf_counter()
    all_ids  = [p[0] for p in raw_pairs]
    meta_map = _fetch_metadata(all_ids)
    t_meta   = (time.perf_counter() - t2) * 1000

    # --- 4. Build results
    results = []
    for rank, (paper_id, dist) in enumerate(raw_pairs):
        if len(results) >= req.k:
            break
        meta = meta_map.get(paper_id, {})
        results.append(Paper(
            rank=rank + 1,
            id=paper_id,
            title=meta.get("title") or "(no title)",
            year=int(meta["year"]) if meta.get("year") else None,
            doi=meta.get("doi") or None,
            cited_by_count=int(meta["cited_by_count"]) if meta.get("cited_by_count") else None,
            source=meta.get("source") or None,
            score=float(dist),
            text_excerpt=meta.get("text_excerpt") or None,
            full_text=meta.get("full_text") or None,
        ))

    latency_ms = (time.perf_counter() - t0) * 1000
    print(f"[search] '{req.query[:50]}' → {len(results)} results | "
          f"total={latency_ms:.1f}ms (embed={t_embed:.1f}ms, "
          f"faiss={t_faiss:.2f}ms, meta={t_meta:.2f}ms)")

    return SearchResponse(
        query=req.query,
        results=results,
        latency_ms=round(latency_ms, 2),
        index_size=_state["n_vectors"],
    )


if __name__ == "__main__":
    uvicorn.run("search_backend:app", host="0.0.0.0", port=PORT, reload=False)
