# Running EVIRAG Pipeline on Kaggle (T4x2 GPU)

## Quick Setup (5 min)

1. **Go to:** https://www.kaggle.com/code/create
2. **Create a new notebook** (Python, Dataset: None)
3. **In the first cell, paste this one-liner:**

```python
%cd /tmp && !git clone https://github.com/amethystani/evirag-search.git && %run evirag-search/kaggle_runner.py
```

4. **Run the cell** — it will:
   - Clone the repo
   - Install all dependencies (30-60s)
   - Download 100 papers (to demo in reasonable time)
   - Embed them with SPECTER2 on T4x2
   - Build FAISS index
   - Upload to HF Datasets (if you set HF_TOKEN)

**Total runtime: ~20-30 min for 100 papers on T4x2 GPU**

---

## Setup HF Token (Optional, for upload)

In a **separate cell**, before running the pipeline:

```python
import os
os.environ["HF_TOKEN"] = "your_hf_token_here"  # Get from https://huggingface.co/settings/tokens
```

Then run `kaggle_runner.py` — it will auto-upload results to HF Datasets.

---

## For Full 40M Papers (Advanced)

Replace the one-liner with:

```python
# Cell 1: Clone and setup
%cd /tmp
!git clone https://github.com/amethystani/evirag-search.git
os.environ["HF_TOKEN"] = "your_token_here"
```

```python
# Cell 2: Edit the demo limit
# Open kaggle_runner.py and change:
# max_papers_demo = 100  →  max_papers_demo = 40_000_000
# Then run the full pipeline (takes ~10 hrs on T4x2)
```

**⚠️ Note:** Kaggle notebooks have a **12-hour runtime limit**. For 40M papers:
- Step 1 (download): 5-8 hrs
- Step 2 (embed on T4x2): 4-6 hrs
- Steps 3-4: <1 hr
- **Total: fits in 12-hour limit with ~1 hr buffer**

---

## File Locations in Kaggle

After running `kaggle_runner.py`:
- Intermediate data: `/kaggle/working/raw_corpus/` and `/kaggle/working/embeddings/`
- All files auto-download to "Outputs" if you export them:

```python
import shutil
shutil.copy("/kaggle/working/embeddings/evirag.index", "/kaggle/output/")
shutil.copy("/kaggle/working/embeddings/ids.npy", "/kaggle/output/")
shutil.copy("/kaggle/working/embeddings/metadata.parquet", "/kaggle/output/")
```

---

## Verify It Works (Smoke Test)

```python
# In a cell after pipeline completes:
import faiss
import numpy as np

index = faiss.read_index_binary("/kaggle/working/embeddings/evirag.index")
print(f"Index loaded: {index.ntotal} vectors")

ids = np.load("/kaggle/working/embeddings/ids.npy", allow_pickle=True)
print(f"IDs loaded: {len(ids)} papers")

# Quick query
query_vec = np.random.randint(0, 2, (1, 768 // 8), dtype=np.uint8)
D, I = index.search(query_vec.astype(np.uint8), k=5)
print(f"Top-5 indices: {I[0]}")
```

---

## Why Kaggle?

- **Free T4x2 GPU** (vs no free GPU elsewhere)
- **Persistent storage** (500GB per notebook)
- **No setup** (Python, CUDA, deps pre-installed)
- **12-hour runtime** (enough for full pipeline)
- **Auto-upload** to HF Datasets via HF token

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Import error (faiss, etc) | `!pip install faiss-cpu sentence-transformers duckdb` |
| GPU not detected | Run `!nvidia-smi` to verify GPU |
| Timeout after 12 hrs | Use 40M limit but upload checkpoint every 5 batches (resumable) |
| HF upload fails | Verify `HF_TOKEN` with `!echo $HF_TOKEN` |

---

## Next Steps

After pipeline finishes:
1. Verify files in `/kaggle/working/embeddings/`
2. If you set HF_TOKEN, check upload at: https://huggingface.co/datasets/amethystani/evirag-index
3. Query the live API: https://huggingface.co/spaces/amethystani/evirag-search

---

**Questions?** Check the `kaggle_runner.py` inline comments.
