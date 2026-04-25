"""
End-to-end smoke test with 5 papers.
Runs the full pipeline locally: download → embed → index → upload → query.

Usage:
    python test_e2e.py
"""

import os, warnings, shutil, tempfile, time
warnings.filterwarnings("ignore")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import faiss
import duckdb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi, create_repo, hf_hub_download

HF_TOKEN  = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("Warning: HF_TOKEN not set. Test will use public access only.")
    HF_TOKEN = None
DATASET   = "amethystani/evirag-index"
WORK_DIR  = Path(tempfile.mkdtemp(prefix="evirag_test_"))
print(f"Working dir: {WORK_DIR}\n")

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Download 5 papers from OpenAlex")
print("=" * 60)

resp = requests.get(
    "https://api.openalex.org/works"
    "?filter=has_abstract:true,cited_by_count:>50,language:en"
    "&per-page=5"
    "&select=id,doi,title,abstract_inverted_index,publication_year,cited_by_count,primary_location"
    "&mailto=animeshmishra0567@gmail.com",
    timeout=15,
)
resp.raise_for_status()
works = resp.json()["results"]
print(f"Downloaded {len(works)} papers")

def reconstruct_abstract(inv):
    if not inv: return ""
    pairs = [(pos, w) for w, positions in inv.items() for pos in positions]
    return " ".join(w for _, w in sorted(pairs))

papers = []
for w in works:
    abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
    src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    papers.append({
        "id":             w["id"],
        "doi":            w.get("doi") or "",
        "title":          (w.get("title") or "").strip(),
        "abstract":       abstract.strip(),
        "year":           w.get("publication_year") or 0,
        "cited_by_count": w.get("cited_by_count") or 0,
        "source":         src,
    })
    print(f"  [{papers[-1]['year']}] {papers[-1]['title'][:70]}...")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Embed with SPECTER + binary quantize")
print("=" * 60)

model = SentenceTransformer("allenai-specter")
texts = [f"{p['title']} [SEP] {p['abstract']}" for p in papers]

embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
binary     = np.packbits((embeddings > 0).astype(np.uint8), axis=1)

print(f"Embeddings shape : {embeddings.shape}  (float32)")
print(f"Binary shape     : {binary.shape}  (uint8, {binary.nbytes} bytes total)")

ids_arr = np.array([p["id"] for p in papers], dtype=object)

# Save locally
np.save(WORK_DIR / "ids.npy",    ids_arr,    allow_pickle=True)
np.save(WORK_DIR / "binary.npy", binary,     allow_pickle=False)

meta_table = pa.table({
    "id":            pa.array([p["id"]            for p in papers]),
    "title":         pa.array([p["title"]         for p in papers]),
    "year":          pa.array([p["year"]          for p in papers], type=pa.int16()),
    "doi":           pa.array([p["doi"]           for p in papers]),
    "cited_by_count":pa.array([p["cited_by_count"]for p in papers], type=pa.int32()),
    "source":        pa.array([p["source"]        for p in papers]),
})
pq.write_table(meta_table, WORK_DIR / "metadata.parquet", compression="zstd")
print("Saved ids.npy, binary.npy, metadata.parquet")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Build FAISS IndexBinaryIVF (tiny: nlist=1 for 5 papers)")
print("=" * 60)

DIM   = embeddings.shape[1]   # 768
# For 5 papers use nlist=1 (IVF needs nlist ≤ N; for real run use 8192)
nlist = 1
quantizer = faiss.IndexBinaryFlat(DIM)
index     = faiss.IndexBinaryIVF(quantizer, DIM, nlist)
index.nprobe = 1

train_vecs = np.ascontiguousarray(binary)
index.train(train_vecs)
index.add(train_vecs)
print(f"Index trained and populated: {index.ntotal} vectors")

# Sanity search — query with paper 0, should get itself as #1
D, I = index.search(np.ascontiguousarray(binary[0:1]), k=3)
print(f"Sanity search (query=paper[0]): top-3 indices = {I[0]}, distances = {D[0]}")
assert 0 in I[0], "paper[0] not in its own top-3 — something is wrong"
print("Sanity check PASSED")

faiss.write_index_binary(index, str(WORK_DIR / "evirag.index"))
print(f"Index saved ({(WORK_DIR / 'evirag.index').stat().st_size} bytes)")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Upload test files to HF Dataset repo")
print("=" * 60)

api = HfApi(token=HF_TOKEN)
create_repo(DATASET, repo_type="dataset", private=False, exist_ok=True, token=HF_TOKEN)

for fname in ["evirag.index", "ids.npy", "metadata.parquet"]:
    fpath = WORK_DIR / fname
    api.upload_file(
        path_or_fileobj=str(fpath),
        path_in_repo=f"test/{fname}",   # upload under test/ prefix to not overwrite production
        repo_id=DATASET,
        repo_type="dataset",
        commit_message=f"[smoke-test] {fname}",
    )
    print(f"  Uploaded test/{fname}")

print(f"Files at: https://huggingface.co/datasets/{DATASET}/tree/main/test")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Download back from HF + query (simulates Space startup)")
print("=" * 60)

dl_dir = WORK_DIR / "downloaded"
dl_dir.mkdir()

for fname in ["evirag.index", "ids.npy", "metadata.parquet"]:
    local = hf_hub_download(
        repo_id=DATASET,
        filename=f"test/{fname}",
        repo_type="dataset",
        token=HF_TOKEN,
        local_dir=str(dl_dir),
        local_dir_use_symlinks=False,
    )
    print(f"  Downloaded → {Path(local).name}")

# Load index
loaded_index = faiss.read_index_binary(str(dl_dir / "test" / "evirag.index"))
loaded_index.nprobe = 1
loaded_ids = np.load(str(dl_dir / "test" / "ids.npy"), allow_pickle=True)
conn = duckdb.connect(":memory:")
conn.execute(f"CREATE VIEW papers AS SELECT * FROM read_parquet('{dl_dir}/test/metadata.parquet')")

# Embed a query
query_text = papers[0]["title"]   # search by first paper's title
query_emb  = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)
query_bin  = np.packbits((query_emb > 0).astype(np.uint8), axis=1)
query_bin  = np.ascontiguousarray(query_bin)

t0   = time.perf_counter()
D, I = loaded_index.search(query_bin, k=3)
latency_ms = (time.perf_counter() - t0) * 1000

top_ids = [loaded_ids[i] for i in I[0] if 0 <= i < len(loaded_ids)]
ids_sql = ", ".join(f"'{i}'" for i in top_ids)
rows = conn.execute(f"SELECT id, title, year FROM papers WHERE id IN ({ids_sql})").fetchall()

print(f"\nQuery : \"{query_text[:60]}...\"")
print(f"Latency: {latency_ms:.2f} ms")
print("Results:")
for rank, (oid, title, year) in enumerate(rows, 1):
    print(f"  #{rank}  [{year}] {title[:70]}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL STEPS PASSED — pipeline is working end to end")
print("=" * 60)
print(f"\nNext: run the 4 pipeline scripts with your full 40M paper corpus.")
print(f"Temp files cleaned up from {WORK_DIR}")
shutil.rmtree(WORK_DIR)
