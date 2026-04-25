# EVIRAG Search — Full Pipeline Run on GPU

**TL;DR for your 48GB Ada 4500 machine:**

```bash
cd evirag-search/pipeline
pip install -r requirements.txt

# These run sequentially. Times below assume Ada 4500.
python3 1_download_corpus.py        # ~5-8 hrs (CPU/network-bound, no GPU help)
python3 2_embed_and_quantize.py     # ~4-6 hrs on GPU (vs 2 days on CPU)
python3 3_build_faiss_index.py      # ~15-20 min
python3 4_upload_to_hf.py           # ~30-60 min (network, no GPU help)

# Then the Space auto-fetches and serves queries at 10-15ms.
```

---

## What This Pipeline Does

Builds a **40M+ paper search index** from OpenAlex (free):

1. **Download** — OpenAlex API cursor pagination → filtered corpus (5-8 hrs, resumable)
2. **Embed** — SPECTER2 scientific paper embeddings → binary quantized (4-6 hrs on Ada)
3. **Index** — FAISS `IndexBinaryIVF` (8192 clusters, 10-20ms queries)
4. **Upload** — Index to HF Datasets (30-60 min upload)
5. **Serve** — FastAPI on HF Spaces (already deployed at https://huggingface.co/spaces/amethystani/evirag-search)

**Index size:** ~4GB binary + ~2GB metadata = 6GB total on HF

---

## Setup

### Before running:

Set HF token (if you need to upload to a different repo, otherwise use existing `amethystani/evirag-index`):

```bash
export HF_TOKEN=hf_YOUR_TOKEN_HERE
```

Or edit the `HF_TOKEN` in `pipeline/4_upload_to_hf.py` before running.

### Install dependencies:

```bash
cd evirag-search/pipeline
pip install -r requirements.txt
```

All libs support GPU. SentenceTransformers will auto-detect CUDA.

---

## Step 1: Download Corpus (CPU-bound, ~5-8 hrs)

```bash
python3 1_download_corpus.py
```

**What it does:**
- Queries OpenAlex API with cursor pagination (no API key needed)
- Filters: `has_abstract=true, cited_by_count>5, language=en, year>=2000`
- Saves batches to `./raw_corpus/batch_NNNN.parquet` (~200k papers each)
- Saves cursor to `./raw_corpus/cursor.txt` for resumption

**Output:** ~40-50M papers in 200+ Parquet files (~100GB uncompressed)

**Resume:** If it crashes, just re-run—it loads the cursor and continues.

---

## Step 2: Embed + Quantize (GPU-intensive, 4-6 hrs on Ada)

```bash
python3 2_embed_and_quantize.py
```

**What it does:**
- Loads batches from `raw_corpus/`
- Embeds abstracts with SPECTER2 (allenai-specter, 768-dim)
- Binary-quantizes: sign(embedding) → 96 bytes/vector (32x compression)
- Saves to `./embeddings/`:
  - `ids.npy` — OpenAlex IDs
  - `binary.npy` — 40M × 96 byte binary vectors
  - `metadata.parquet` — title, year, doi, cited_by_count, source

**GPU usage:** ~30-35GB VRAM peak (safe on 48GB Ada)

**Checkpoint:** Saves progress every 5 batch files. If it crashes, resume by setting `RESUME_FROM = X` in the script.

---

## Step 3: Build Index (CPU, ~15-20 min)

```bash
python3 3_build_faiss_index.py
```

**What it does:**
- Loads binary vectors (memory-mapped, doesn't load all at once)
- Builds `IndexBinaryIVF` with 8192 clusters
- Trains on 2M random vectors
- Adds all 40M vectors in batches
- Saves to `./embeddings/evirag.index` (~4GB)

**Output:** Ready-to-query FAISS index

---

## Step 4: Upload to HF (Network-bound, 30-60 min)

```bash
HF_TOKEN=your_hf_token_here python3 4_upload_to_hf.py
```

**What it does:**
- Creates `amethystani/evirag-index` dataset repo on HF (if not exists)
- Uploads 3 files:
  - `evirag.index` (~4GB)
  - `ids.npy` (~320MB)
  - `metadata.parquet` (~2GB)

**Output:** Files live at https://huggingface.co/datasets/amethystani/evirag-index

The Space (https://huggingface.co/spaces/amethystani/evirag-search) auto-downloads these on first boot.

---

## Query the Live Space

Once step 4 completes and the index is on HF:

```bash
curl -X POST https://amethystani-evirag-search.hf.space/search \
  -H "Content-Type: application/json" \
  -d '{"query": "vaccine hesitancy misinformation", "k": 20}'
```

**Response:** JSON with top-20 papers, latency, index stats.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Step 1 crashes mid-download | Re-run—it resumes from cursor |
| Step 2 OOM (even on 48GB) | Lower `BATCH_SIZE` in the script |
| Step 2 very slow | Check GPU via `nvidia-smi`; ensure SentenceTransformers is using CUDA |
| Step 4 upload timeout | Your HF token might be invalid; check `HF_TOKEN` env var |

---

## Timing Estimate (Ada 4500, 48GB VRAM)

| Step | Time | Bottleneck |
|------|------|-----------|
| 1. Download | 5-8 hrs | Network + API rate-limit (10 req/sec) |
| 2. Embed | 4-6 hrs | GPU (SPECTER2 inference) |
| 3. Index | 15-20 min | CPU (FAISS training) |
| 4. Upload | 30-60 min | Network (if not saturated) |
| **Total** | **~10-15 hrs** | Mostly GPU embedding |

---

## For EVIRAG Paper

After full pipeline completes, you have:
- **40M+ scientific papers**, binary-indexed, searchable in <20ms
- **Free infrastructure** (HF Datasets + Spaces)
- **Engineering contribution:** "Binary-quantized FAISS on commodity hardware serves large scientific corpora faster than in-memory vector DBs at a fraction of the cost"

The disagreement-detection thesis can now run at full scale.

---

## Questions?

Check `space/README.md` for API details.
Check `pipeline/` scripts for inline comments on each step.
