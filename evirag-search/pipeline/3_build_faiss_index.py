"""
Step 3: Build FAISS IndexBinaryIVF from binary embeddings.

Input:  ./embeddings/binary.npy   (N × 96 uint8)
Output: ./embeddings/evirag.index  (FAISS binary IVF index)

IndexBinaryIVF with nlist=8192:
  - Training: ~2M random vectors, ~10 min on CPU
  - Adding 40M vectors: ~20 min on CPU
  - Query latency: ~10-20ms for k=100 with nprobe=128

The index is ~4.1 GB on disk (binary IVF adds ~8% overhead over raw vectors).
"""

import numpy as np
import faiss
from pathlib import Path
import time

# ── Config ────────────────────────────────────────────────────────────────────
EMB_DIR     = Path("./embeddings")
INDEX_PATH  = EMB_DIR / "evirag.index"
DIM         = 768         # embedding dimensions (bits)
NLIST       = 8192        # IVF clusters (√40M ≈ 6324; use 8192 for better recall)
NPROBE      = 128         # clusters searched per query (set at query time too)
TRAIN_SIZE  = 2_000_000  # vectors used for training (subset)
ADD_BATCH   = 500_000     # add in batches to avoid peak memory

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading binary embeddings...")
    binary = np.load(EMB_DIR / "binary.npy", mmap_mode="r")  # memory-mapped, no full load
    N = binary.shape[0]
    print(f"  Shape: {binary.shape}  ({N:,} vectors, {binary.nbytes / 1e9:.2f} GB)")

    # ── Build index ───────────────────────────────────────────────────────────
    quantizer = faiss.IndexBinaryFlat(DIM)
    index     = faiss.IndexBinaryIVF(quantizer, DIM, NLIST)
    index.nprobe = NPROBE

    # ── Train on a random subset ──────────────────────────────────────────────
    train_size = min(TRAIN_SIZE, N)
    print(f"\nTraining on {train_size:,} vectors (nlist={NLIST})...")
    rng     = np.random.default_rng(42)
    idx     = rng.choice(N, size=train_size, replace=False)
    train_vecs = np.ascontiguousarray(binary[idx])  # copy subset into contiguous memory

    t0 = time.time()
    index.train(train_vecs)
    del train_vecs
    print(f"  Training done in {time.time()-t0:.1f}s")

    # ── Add all vectors in batches ────────────────────────────────────────────
    print(f"\nAdding {N:,} vectors in batches of {ADD_BATCH:,}...")
    t0 = time.time()
    for start in range(0, N, ADD_BATCH):
        end   = min(start + ADD_BATCH, N)
        batch = np.ascontiguousarray(binary[start:end])
        index.add(batch)
        elapsed = time.time() - t0
        speed   = (end) / elapsed
        eta     = (N - end) / speed if speed > 0 else 0
        print(f"  {end:>12,} / {N:,}  ({speed/1e3:.1f}k vec/s  ETA {eta/60:.1f}min)", end="\r")
    print()

    # ── Verify with a quick sanity search ────────────────────────────────────
    print("\nSanity check: searching a random query...")
    query = np.ascontiguousarray(binary[0:1])
    D, I  = index.search(query, k=5)
    print(f"  Top-5 indices: {I[0]}  (index 0 should be in there)")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving index to {INDEX_PATH}...")
    faiss.write_index_binary(index, str(INDEX_PATH))
    size_gb = INDEX_PATH.stat().st_size / 1e9
    print(f"  Saved. Size: {size_gb:.2f} GB")
    print(f"\nDone. {N:,} vectors indexed. Query with nprobe={NPROBE} for best recall.")


if __name__ == "__main__":
    main()
