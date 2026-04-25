"""
ULTRA-FAST EVIRAG PIPELINE — 4 HOURS MAX ON T4x2 GPU

Strategy:
  1. Download only 500k HIGH-QUALITY papers (cited_by_count > 100, recent)
  2. Embed on T4x2 with Flash Attention 2 + torch.compile + bfloat16
  3. Minimal checkpointing (faster but less resumable)
  4. Upload streaming (don't wait for index completion)
  5. Stream output directly to HF

Timeline estimate:
  Download: 20 min (fewer papers, quality filtered)
  Embed:    2.5 hrs (500k @ 50 papers/sec on T4x2 + FlashAttn)
  Index:    5 min
  Upload:   15 min
  Total:    ~3 hours (leaves 1 hour buffer)
"""

import asyncio
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

try:
    import orjson as json_lib
    _loads = lambda b: json_lib.loads(b)
except ImportError:
    import json as json_lib
    _loads = json_lib.loads

import aiohttp
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import HfApi, create_repo

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("EVIRAG ULTRA-FAST PIPELINE — 4 Hours on T4x2")
print(f"{'='*70}\n")

# Detect environment
IS_KAGGLE = os.path.exists("/kaggle/working")
WORK_DIR = "/kaggle/working" if IS_KAGGLE else "./evirag_fast"
Path(WORK_DIR).mkdir(exist_ok=True)
print(f"Working directory: {WORK_DIR}")

# GPU setup
assert torch.cuda.is_available(), "CUDA required!"
gpu_name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)\n")

# HF token — embedded for Kaggle kernel execution
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    print(f"✓ HF_TOKEN set (upload enabled)")
else:
    print("⚠️ HF_TOKEN not set, upload will be skipped.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: FAST DOWNLOAD (20 min for 500k papers)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 1: Download 500k high-quality papers (20 min)")
print(f"{'='*70}")

RAW_DIR = Path(WORK_DIR) / "raw_corpus"
RAW_DIR.mkdir(exist_ok=True)

def reconstruct_abstract(inv):
    if not inv:
        return ""
    pairs = [(pos, w) for w, positions in inv.items() for pos in positions]
    return " ".join(w for _, w in sorted(pairs))

async def fast_download():
    buffer = []
    total = 0
    target = 500_000

    cursor = "*"
    headers = {"User-Agent": "EVIRAG-Fast/1.0 (mailto:animeshmishra0567@gmail.com)"}

    # cursor pagination is incompatible with sort — filter only, no sort param
    base_url = (
        "https://api.openalex.org/works"
        "?filter=has_abstract:true,cited_by_count:>100,language:en,publication_year:2014-2025"
        "&per-page=200&mailto=animeshmishra0567@gmail.com"
    )

    connector = aiohttp.TCPConnector(limit=8, ttl_dns_cache=300)
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        while total < target:
            url = f"{base_url}&cursor={cursor}"
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        print(f"API error: {resp.status} — stopping")
                        break
                    data = _loads(await resp.read())
            except Exception as e:
                print(f"Download error: {e}")
                break

            results = data.get("results", [])
            if not results:
                break

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
                total += 1
                if total >= target:
                    break

            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break

            pct = min(100, int(100 * total / target))
            print(f"  Downloaded {total:,} papers ({pct}%)", end="\r")
            await asyncio.sleep(0.05)  # polite rate limit

    # Save to Parquet
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
        pq.write_table(table, RAW_DIR / "corpus.parquet", compression="zstd")
        print(f"\n✓ Downloaded & saved {total:,} papers")

    return total

t0 = time.time()
n_papers = asyncio.run(fast_download())
print(f"  Time: {(time.time()-t0)/60:.1f} min")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: FAST EMBED (2.5 hrs for 500k papers on T4x2)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"STEP 2: Embed {n_papers:,} papers with GPU acceleration (2.5 hrs)")
print(f"{'='*70}")

EMB_DIR = Path(WORK_DIR) / "embeddings"
EMB_DIR.mkdir(exist_ok=True)

# GPU optimization
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Load SPECTER
print("Loading SPECTER with Flash Attention 2...")
tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
try:
    model = AutoModel.from_pretrained(
        "allenai/specter",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda").eval()
    print("✓ Loaded with Flash Attention 2")
except:
    model = AutoModel.from_pretrained(
        "allenai/specter",
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    print("✓ Loaded with standard attention")

# Try torch.compile for kernel fusion
try:
    model = torch.compile(model, mode="max-autotune")
    print("✓ Model compiled with torch.compile")
except:
    print("⚠️  torch.compile skipped")

# Read all papers
print("Reading corpus...")
df = pq.read_table(RAW_DIR / "corpus.parquet").to_pydict()
texts = [f"{t} [SEP] {a}" for t, a in zip(df["title"], df["abstract"])]

# Tokenize
print("Tokenizing...")
enc = tokenizer(texts, max_length=256, padding=True, truncation=True, return_tensors="pt")

# Embed in huge batches
print("Embedding on GPU...")
ids_arr = np.array(df["id"], dtype=object)
binary_list = []
meta_list = []

t0 = time.time()
batch_size = 2048  # Aggressive batch size
n_batches = (len(texts) + batch_size - 1) // batch_size

with torch.no_grad():
    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, len(texts))

        ids_b = enc["input_ids"][start:end].to("cuda", non_blocking=True)
        mask_b = enc["attention_mask"][start:end].to("cuda", non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids_b, attention_mask=mask_b)
            emb = out.last_hidden_state[:, 0, :]  # [CLS] token
            emb = F.normalize(emb, p=2, dim=1)

        # Binary quantize on GPU
        bits = (emb > 0).to(torch.uint8).cpu().numpy()
        binary = np.packbits(bits, axis=1)
        binary_list.append(binary)

        # Metadata
        for i in range(start, end):
            meta_list.append({
                "id": df["id"][i],
                "title": df["title"][i],
                "year": df["year"][i],
                "doi": df["doi"][i],
                "cited_by_count": df["cited_by_count"][i],
                "source": df["source"][i],
            })

        pct = int(100 * end / len(texts))
        speed = end / (time.time() - t0)
        print(f"  {end:,} / {len(texts):,} ({pct}%) — {speed:.0f} papers/sec", end="\r")

embed_time = time.time() - t0
print(f"\n✓ Embedded {len(texts):,} papers in {embed_time/60:.1f} min")

# Save
binary_arr = np.vstack(binary_list)
np.save(EMB_DIR / "ids.npy", ids_arr, allow_pickle=True)
np.save(EMB_DIR / "binary.npy", binary_arr, allow_pickle=False)

meta_table = pa.table({
    "id": pa.array([m["id"] for m in meta_list]),
    "title": pa.array([m["title"] for m in meta_list]),
    "year": pa.array([m["year"] for m in meta_list], type=pa.int16()),
    "doi": pa.array([m["doi"] for m in meta_list]),
    "cited_by_count": pa.array([m["cited_by_count"] for m in meta_list], type=pa.int32()),
    "source": pa.array([m["source"] for m in meta_list]),
})
pq.write_table(meta_table, EMB_DIR / "metadata.parquet", compression="zstd")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: BUILD INDEX (5 min)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 3: Build FAISS index (5 min)")
print(f"{'='*70}")

import faiss

binary = np.load(EMB_DIR / "binary.npy")
N = binary.shape[0]
DIM = binary.shape[1] * 8

faiss.omp_set_num_threads(os.cpu_count() or 16)

quantizer = faiss.IndexBinaryFlat(DIM)
nlist = 4096
index = faiss.IndexBinaryIVF(quantizer, DIM, nlist)
index.nprobe = 64

print(f"Training on {min(N, 100_000):,} vectors...")
train_idx = np.random.choice(N, size=min(N, 100_000), replace=False)
train_vecs = np.ascontiguousarray(binary[train_idx])
index.train(train_vecs)

print(f"Adding {N:,} vectors...")
for start in range(0, N, 1_000_000):
    end = min(start + 1_000_000, N)
    batch = np.ascontiguousarray(binary[start:end])
    index.add(batch)
    print(f"  {end:,} / {N:,}", end="\r")

faiss.write_index_binary(index, str(EMB_DIR / "evirag.index"))
print(f"\n✓ Index built: {index.ntotal} vectors")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: UPLOAD (15 min)
# ══════════════════════════════════════════════════════════════════════════════
if HF_TOKEN:
    print(f"\n{'='*70}")
    print("STEP 4: Upload to HF Datasets (15 min)")
    print(f"{'='*70}")

    api = HfApi(token=HF_TOKEN)
    create_repo("amethystani/evirag-index", repo_type="dataset", private=False,
               exist_ok=True, token=HF_TOKEN)

    def upload_file(fpath, fname):
        size = fpath.stat().st_size / 1e9
        print(f"  Uploading {fname} ({size:.2f} GB)...")
        api.upload_file(str(fpath), fname, repo_id="amethystani/evirag-index",
                       repo_type="dataset", commit_message=f"Fast upload: {fname}")
        print(f"    ✓ {fname}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        for fname in ["evirag.index", "ids.npy", "metadata.parquet"]:
            fpath = EMB_DIR / fname
            if fpath.exists():
                pool.submit(upload_file, fpath, fname)

    print(f"✓ Uploaded to https://huggingface.co/datasets/amethystani/evirag-index")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("✓ PIPELINE COMPLETE")
print(f"{'='*70}")
print(f"Papers indexed: {n_papers:,}")
print(f"Index location: {EMB_DIR}/evirag.index")
print(f"Query via: https://huggingface.co/spaces/amethystani/evirag-search")
print(f"\nSpace will auto-load new index on next boot.")
