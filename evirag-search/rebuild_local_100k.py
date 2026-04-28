"""
Rebuild the local 100k corpus so FAISS index, ids.npy, and metadata.parquet
are all derived from the same source — corpus.parquet's 4000-char `full_text`.

The previous local_100k mismatch:
  corpus.parquet     →  100k peS2o papers, 4000 chars each, IDs set A
  metadata.parquet   →  100k peS2o papers, 2000 chars each, IDs set B
  evirag.index       →  embeddings of IDs set B (so corpus.parquet text was wasted)

After this script:
  corpus.parquet     →  unchanged
  metadata.parquet   →  re-derived from corpus.parquet (same IDs, full_text → text_excerpt)
  ids.npy            →  same order as corpus.parquet rows
  evirag.index       →  rebuilt from SPECTER embeddings of corpus.parquet's title + full_text

Run on Apple Silicon (MPS) — takes ~30-45 min for 100k papers.
"""

import os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import faiss
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
LOCAL_DIR  = ROOT / "local_100k"
CORPUS_IN  = LOCAL_DIR / "corpus.parquet"
META_OUT   = LOCAL_DIR / "metadata.parquet"
IDS_OUT    = LOCAL_DIR / "ids.npy"
INDEX_OUT  = LOCAL_DIR / "evirag.index"
BACKUP_DIR = LOCAL_DIR / "backup_pre_realign"

MAX_LEN    = 256        # SPECTER's effective input length
BATCH      = 64         # MPS-friendly batch size
NLIST      = 256        # FAISS IVF clusters for 100k vectors

# ── Device ─────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda"); DTYPE = torch.bfloat16
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps");  DTYPE = torch.float32
else:
    DEVICE = torch.device("cpu");  DTYPE = torch.float32
print(f"Device: {DEVICE}  dtype: {DTYPE}")


def main():
    if not CORPUS_IN.exists():
        sys.exit(f"corpus.parquet not found at {CORPUS_IN}")

    # ── 1. Load corpus.parquet ────────────────────────────────────────────────
    print(f"\nLoading {CORPUS_IN.name}...")
    t0 = time.time()
    corpus = pq.read_table(str(CORPUS_IN))
    print(f"  ✓ {corpus.num_rows:,} papers in {time.time()-t0:.1f}s")
    print(f"  schema: {[f.name for f in corpus.schema]}")

    ids        = corpus.column("id").to_pylist()
    titles     = corpus.column("title").to_pylist()
    full_texts = corpus.column("full_text").to_pylist()
    years      = corpus.column("year").to_pylist()
    dois       = corpus.column("doi").to_pylist()
    cites      = corpus.column("cited_by_count").to_pylist()
    sources    = corpus.column("source").to_pylist()
    domains    = corpus.column("domain").to_pylist()

    # Build SPECTER inputs: title + [SEP] + full_text (truncated to 256 tokens)
    inputs = []
    for t, ft in zip(titles, full_texts):
        t  = (t  or "").strip()
        ft = (ft or "").strip()
        if t and ft:
            inputs.append(f"{t} [SEP] {ft}")
        else:
            inputs.append(t or ft)
    print(f"  ✓ Built {len(inputs):,} SPECTER inputs")

    # ── 2. Backup existing files ──────────────────────────────────────────────
    BACKUP_DIR.mkdir(exist_ok=True)
    for f in [META_OUT, IDS_OUT, INDEX_OUT]:
        if f.exists():
            backup = BACKUP_DIR / f.name
            if not backup.exists():
                f.replace(backup)   # move (atomic) — keeps the backup pristine
                print(f"  Backed up {f.name} → {backup.name}")

    # ── 3. Load SPECTER ────────────────────────────────────────────────────────
    print(f"\nLoading SPECTER on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
    model     = AutoModel.from_pretrained("allenai/specter").eval()
    if DEVICE.type == "cuda":
        model = model.to(dtype=torch.bfloat16).to(DEVICE)
    else:
        model = model.to(DEVICE)

    # ── 4. Embed in batches ────────────────────────────────────────────────────
    print(f"\nEmbedding {len(inputs):,} papers (batch={BATCH})...")
    all_packed = []
    t_start = time.time()

    @torch.no_grad()
    def embed_batch(texts):
        enc = tokenizer(texts, max_length=MAX_LEN, padding=True,
                        truncation=True, return_tensors="pt")
        ids_b  = enc["input_ids"].to(DEVICE)
        mask_b = enc["attention_mask"].to(DEVICE)

        if DEVICE.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids_b, attention_mask=mask_b)
        else:
            out = model(input_ids=ids_b, attention_mask=mask_b)

        # Mean pool weighted by attention mask
        emb = (out.last_hidden_state * mask_b.unsqueeze(-1)).sum(1) \
              / mask_b.sum(1, keepdim=True).clamp(min=1)
        emb = F.normalize(emb, p=2, dim=1)
        bits = (emb > 0).to(torch.uint8).cpu().numpy()       # (B, 768)
        return np.packbits(bits, axis=1)                      # (B, 96) uint8

    for i in range(0, len(inputs), BATCH):
        batch_in = inputs[i : i + BATCH]
        all_packed.append(embed_batch(batch_in))
        if (i // BATCH) % 50 == 0 and i:
            elapsed = time.time() - t_start
            rate    = i / elapsed
            eta_min = (len(inputs) - i) / rate / 60
            print(f"  {i:>7,}/{len(inputs):,}  ({rate:.0f} papers/s)  ETA {eta_min:.1f} min")

    binary = np.vstack(all_packed)   # (N, 96) uint8
    elapsed = time.time() - t_start
    print(f"  ✓ Embedded {binary.shape[0]:,} vectors in {elapsed/60:.1f} min "
          f"({binary.shape[0]/elapsed:.0f} papers/s)")

    # ── 5. Build FAISS IVF binary index ────────────────────────────────────────
    print(f"\nBuilding FAISS IndexBinaryIVF (nlist={NLIST})...")
    t1 = time.time()
    quantizer = faiss.IndexBinaryFlat(768)
    index     = faiss.IndexBinaryIVF(quantizer, 768, NLIST)
    # Train on a sample (or all 100k)
    train_n = min(50000, binary.shape[0])
    perm    = np.random.RandomState(42).permutation(binary.shape[0])[:train_n]
    index.train(binary[perm])
    index.add(binary)
    index.nprobe = 64
    faiss.write_index_binary(index, str(INDEX_OUT))
    print(f"  ✓ {INDEX_OUT.name}  ntotal={index.ntotal:,}  in {time.time()-t1:.1f}s")

    # ── 6. Save aligned ids.npy ────────────────────────────────────────────────
    ids_arr = np.array(ids, dtype=object)
    np.save(str(IDS_OUT), ids_arr, allow_pickle=True)
    print(f"  ✓ {IDS_OUT.name}  ({len(ids_arr):,} IDs)")

    # ── 7. Write aligned metadata.parquet ──────────────────────────────────────
    # text_excerpt = first 4000 chars of full_text (already capped at 4000 in corpus.parquet)
    new_meta = pa.table({
        "id":             pa.array(ids,        type=pa.string()),
        "title":          pa.array(titles,     type=pa.string()),
        "year":           pa.array([int(y or 0) for y in years], type=pa.int16()),
        "doi":            pa.array(dois,       type=pa.string()),
        "cited_by_count": pa.array([int(c or 0) for c in cites], type=pa.int32()),
        "source":         pa.array(sources,    type=pa.string()),
        "domain":         pa.array(domains,    type=pa.string()),
        "text_excerpt":   pa.array([(ft or "")[:4000] for ft in full_texts], type=pa.string()),
    })
    pq.write_table(new_meta, str(META_OUT), compression="zstd", compression_level=6)
    size_mb = META_OUT.stat().st_size / 1e6
    print(f"  ✓ {META_OUT.name}  ({size_mb:.0f} MB, 4000-char excerpts aligned with index)")

    print(f"\n{'='*70}\nRebuild complete. corpus.parquet IDs now match the index 1:1.")
    print(f"Backups in: {BACKUP_DIR}")
    print(f"Restart search_backend.py to pick up the new files.")


if __name__ == "__main__":
    main()
