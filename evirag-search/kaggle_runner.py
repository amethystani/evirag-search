"""
EVIRAG Pipeline Runner for Kaggle Notebooks (T4x2 GPU)

Usage in Kaggle notebook:
  1. Create a new code cell
  2. Paste this entire script
  3. Run it — handles all 4 steps end to end

Features:
  - Auto-detects Kaggle environment
  - Uses /kaggle/working for persistent storage
  - T4x2 GPU auto-detection
  - Resumable (saves checkpoints)
  - Uploads to HF Datasets when done
"""

import os
import sys
import subprocess
import shutil

# ══════════════════════════════════════════════════════════════════════════════
# KAGGLE ENVIRONMENT SETUP
# ══════════════════════════════════════════════════════════════════════════════

IS_KAGGLE = os.path.exists("/kaggle/working")
if IS_KAGGLE:
    WORK_DIR = "/kaggle/working"
    print(f"✓ Running on Kaggle | Work dir: {WORK_DIR}")

    # Check GPU
    result = subprocess.run(["nvidia-smi", "--query-gpu=name", "-u"],
                          capture_output=True, text=True)
    gpu_info = result.stdout.strip()
    print(f"✓ GPU: {gpu_info}")
else:
    WORK_DIR = os.getcwd()
    print(f"⚠ Not on Kaggle | Work dir: {WORK_DIR}")

# Set HF token (provide via environment before running, or set here)
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("⚠ HF_TOKEN not set. Will skip upload step.")
    print("  To enable: !export HF_TOKEN=your_token_here")

# ══════════════════════════════════════════════════════════════════════════════
# INSTALL DEPENDENCIES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("Installing dependencies...")
print("="*70)

deps = [
    "requests>=2.31.0",
    "pyarrow>=14.0.0",
    "sentence-transformers>=2.7.0",
    "faiss-cpu>=1.7.4",
    "duckdb>=0.10.0",
    "huggingface_hub>=0.21.0",
    "tqdm>=4.66.0",
]

for dep in deps:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", dep],
                   check=False)
print("✓ Dependencies installed")

# ══════════════════════════════════════════════════════════════════════════════
# INLINE PIPELINE (copy of the 4-step process)
# ══════════════════════════════════════════════════════════════════════════════

import warnings
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
from tqdm import tqdm
import time


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 1: Download from OpenAlex")
print("="*70)

RAW_DIR = Path(WORK_DIR) / "raw_corpus"
RAW_DIR.mkdir(exist_ok=True)

batch_num = len(list(RAW_DIR.glob("batch_*.parquet")))
print(f"Existing batches: {batch_num} (will continue from here)")

cursor = "*"
cursor_file = RAW_DIR / "cursor.txt"
if cursor_file.exists():
    cursor = cursor_file.read_text().strip()
    print(f"Resuming from cursor")

BATCH_SIZE = 200_000
downloaded_count = batch_num * BATCH_SIZE
total_results = 0

def reconstruct_abstract(inv):
    if not inv: return ""
    pairs = [(pos, w) for w, positions in inv.items() for pos in positions]
    return " ".join(w for _, w in sorted(pairs))

try:
    print("Downloading papers (max 100 to demo)...")
    papers_downloaded = 0
    max_papers_demo = 100  # For Kaggle speed, just demo with 100 papers

    session = requests.Session()
    session.headers.update({"User-Agent": "EVIRAG-Kaggle/1.0 (mailto:animeshmishra0567@gmail.com)"})

    while papers_downloaded < max_papers_demo:
        url = (
            f"https://api.openalex.org/works"
            f"?filter=has_abstract:true,cited_by_count:>5,language:en,publication_year:2000-2025"
            f"&per-page=200&cursor={cursor}&select=id,doi,title,abstract_inverted_index,publication_year,cited_by_count,primary_location"
            f"&mailto=animeshmishra0567@gmail.com"
        )
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            print("No more results")
            break

        buffer = []
        for work in results:
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
            if not abstract:
                continue
            src = ((work.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
            buffer.append({
                "id": work["id"],
                "doi": work.get("doi") or "",
                "title": (work.get("title") or "").strip(),
                "abstract": abstract.strip(),
                "year": work.get("publication_year") or 0,
                "cited_by_count": work.get("cited_by_count") or 0,
                "source": src,
            })
            papers_downloaded += 1
            if papers_downloaded >= max_papers_demo:
                break

        if buffer:
            table = pa.table({
                "id": pa.array([r["id"] for r in buffer]),
                "doi": pa.array([r["doi"] for r in buffer]),
                "title": pa.array([r["title"] for r in buffer]),
                "abstract": pa.array([r["abstract"] for r in buffer]),
                "year": pa.array([r["year"] for r in buffer], type=pa.int16()),
                "cited_by_count": pa.array([r["cited_by_count"] for r in buffer], type=pa.int32()),
                "source": pa.array([r["source"] for r in buffer]),
            })
            pq.write_table(table, RAW_DIR / f"batch_{batch_num:04d}.parquet", compression="zstd")
            batch_num += 1
            print(f"  Saved batch {batch_num-1}: {len(buffer)} papers (total: {papers_downloaded})")

        meta = data.get("meta", {})
        cursor = meta.get("next_cursor")
        if not cursor:
            break

        time.sleep(0.11)  # polite rate limit

    cursor_file.unlink(missing_ok=True)
    print(f"✓ Downloaded {papers_downloaded} papers")

except Exception as e:
    print(f"✗ Step 1 failed: {e}")
    raise

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 2: Embed & Binary Quantize")
print("="*70)

EMB_DIR = Path(WORK_DIR) / "embeddings"
EMB_DIR.mkdir(exist_ok=True)

try:
    print("Loading SPECTER model...")
    model = SentenceTransformer("allenai-specter")
    print("✓ Model loaded")

    print("Processing Parquet batches...")
    all_ids = []
    all_binary = []
    all_meta = []

    for pq_file in sorted(RAW_DIR.glob("batch_*.parquet")):
        table = pq.read_table(pq_file)
        df = table.to_pydict()

        texts = [f"{t} [SEP] {a}" for t, a in zip(df["title"], df["abstract"])]
        embeddings = model.encode(texts, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
        binary = np.packbits((embeddings > 0).astype(np.uint8), axis=1)

        all_ids.extend(df["id"])
        all_binary.append(binary)
        all_meta.extend([
            {"id": df["id"][i], "title": df["title"][i], "year": df["year"][i],
             "doi": df["doi"][i], "cited_by_count": df["cited_by_count"][i],
             "source": df["source"][i]}
            for i in range(len(df["id"]))
        ])

        print(f"  {pq_file.name}: {len(texts)} papers")

    ids_arr = np.array(all_ids, dtype=object)
    binary_arr = np.vstack(all_binary) if all_binary else np.empty((0, 96), dtype=np.uint8)

    np.save(EMB_DIR / "ids.npy", ids_arr, allow_pickle=True)
    np.save(EMB_DIR / "binary.npy", binary_arr, allow_pickle=False)

    meta_table = pa.table({
        "id": pa.array([m["id"] for m in all_meta]),
        "title": pa.array([m["title"] for m in all_meta]),
        "year": pa.array([m["year"] for m in all_meta], type=pa.int16()),
        "doi": pa.array([m["doi"] for m in all_meta]),
        "cited_by_count": pa.array([m["cited_by_count"] for m in all_meta], type=pa.int32()),
        "source": pa.array([m["source"] for m in all_meta]),
    })
    pq.write_table(meta_table, EMB_DIR / "metadata.parquet", compression="zstd")

    print(f"✓ Embedded {len(all_ids)} papers | Binary size: {binary_arr.nbytes / 1e9:.2f} GB")

except Exception as e:
    print(f"✗ Step 2 failed: {e}")
    raise

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 3: Build FAISS Index")
print("="*70)

try:
    binary = np.load(EMB_DIR / "binary.npy", mmap_mode="r")
    N = binary.shape[0]
    DIM = binary.shape[1] * 8  # convert bytes back to bits

    quantizer = faiss.IndexBinaryFlat(DIM)
    nlist = min(max(int(np.sqrt(N)), 64), 4096)  # adaptive nlist
    index = faiss.IndexBinaryIVF(quantizer, DIM, nlist)
    index.nprobe = min(nlist // 16, 128)

    print(f"Training on {min(N, 2_000_000):,} vectors (nlist={nlist})...")
    train_size = min(2_000_000, N)
    idx = np.random.choice(N, size=train_size, replace=False)
    train_vecs = np.ascontiguousarray(binary[idx])

    index.train(train_vecs)
    del train_vecs

    print("Adding vectors...")
    for start in tqdm(range(0, N, 500_000)):
        end = min(start + 500_000, N)
        batch = np.ascontiguousarray(binary[start:end])
        index.add(batch)

    faiss.write_index_binary(index, str(EMB_DIR / "evirag.index"))
    print(f"✓ Index built: {index.ntotal} vectors")

except Exception as e:
    print(f"✗ Step 3 failed: {e}")
    raise

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 4: Upload to HF (optional)")
print("="*70)

if HF_TOKEN:
    try:
        api = HfApi(token=HF_TOKEN)
        create_repo("amethystani/evirag-index", repo_type="dataset",
                   private=False, exist_ok=True, token=HF_TOKEN)

        for fname in ["evirag.index", "ids.npy", "metadata.parquet"]:
            fpath = EMB_DIR / fname
            if fpath.exists():
                print(f"Uploading {fname}...")
                api.upload_file(str(fpath), fname, repo_id="amethystani/evirag-index",
                              repo_type="dataset", commit_message=f"[Kaggle] {fname}")

        print(f"✓ Uploaded to https://huggingface.co/datasets/amethystani/evirag-index")
    except Exception as e:
        print(f"⚠ Upload failed: {e}")
else:
    print("⚠ Skipped (HF_TOKEN not set)")

# ─────────────────────────────────────────────────────────────────────────════
print("\n" + "="*70)
print("PIPELINE COMPLETE")
print("="*70)
print(f"Output directory: {WORK_DIR}")
print("Files saved:")
print(f"  - {EMB_DIR}/evirag.index")
print(f"  - {EMB_DIR}/ids.npy")
print(f"  - {EMB_DIR}/metadata.parquet")
print("\nTo query: Use the HF Space API at")
print("  https://huggingface.co/spaces/amethystani/evirag-search")
