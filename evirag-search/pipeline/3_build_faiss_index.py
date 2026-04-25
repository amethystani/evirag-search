"""
Step 3: Build FAISS IndexBinaryIVF from binary embeddings.

CPU MAXIMIZATION — FAISS binary indexes do NOT support GPU, so we max out CPU:
  • faiss.omp_set_num_threads(all cores) — saturate every vCPU
  • Load full binary array into RAM (3.84 GB is nothing on a GPU server)
  • 5M training vectors (vs 2M) for better IVF cluster quality at 40M scale
  • Add in 2M-vector batches for speed (fewer Python calls)
  • np.ascontiguousarray for every slice (avoids internal FAISS copies)

Input:  ./embeddings/binary.npy   (N × 96 uint8)
Output: ./embeddings/evirag.index  (FAISS IndexBinaryIVF, ~4.1 GB)

Runtime: ~10-15 min on a 32-core machine for 40M vectors.
"""

import os
import numpy as np
import faiss
from pathlib import Path
import time

# ── Config ────────────────────────────────────────────────────────────────────
EMB_DIR     = Path("./embeddings")
INDEX_PATH  = EMB_DIR / "evirag.index"
DIM         = 768           # embedding dimension in bits
NLIST       = 8192          # IVF clusters  (√40M ≈ 6324; 8192 gives better recall)
NPROBE      = 128           # clusters searched per query
TRAIN_SIZE  = 5_000_000    # more training data → better cluster centroids
ADD_BATCH   = 2_000_000    # larger add batches → fewer Python round-trips

# ── Saturate all CPU cores ────────────────────────────────────────────────────
n_cores = os.cpu_count() or 16
faiss.omp_set_num_threads(n_cores)
print(f"FAISS OMP threads: {n_cores}  (all cores)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading binary embeddings into RAM...")
    t0     = time.time()
    binary = np.load(EMB_DIR / "binary.npy")   # full load — faster than mmap for random access
    N      = binary.shape[0]
    print(f"  Shape: {binary.shape}  |  {N:,} vectors  |  {binary.nbytes/1e9:.2f} GB  "
          f"(loaded in {time.time()-t0:.1f}s)")

    # ── Build index ───────────────────────────────────────────────────────────
    quantizer = faiss.IndexBinaryFlat(DIM)
    index     = faiss.IndexBinaryIVF(quantizer, DIM, NLIST)
    index.nprobe = NPROBE

    # ── Train ─────────────────────────────────────────────────────────────────
    train_size = min(TRAIN_SIZE, N)
    print(f"\nTraining on {train_size:,} vectors  (nlist={NLIST})...")
    rng        = np.random.default_rng(42)
    idx        = rng.choice(N, size=train_size, replace=False)
    train_vecs = np.ascontiguousarray(binary[idx])

    t0 = time.time()
    index.train(train_vecs)
    del train_vecs   # free immediately
    print(f"  Training done in {time.time()-t0:.1f}s")

    # ── Add all vectors ───────────────────────────────────────────────────────
    print(f"\nAdding {N:,} vectors in batches of {ADD_BATCH:,}...")
    t0 = time.time()
    for start in range(0, N, ADD_BATCH):
        end   = min(start + ADD_BATCH, N)
        batch = np.ascontiguousarray(binary[start:end])
        index.add(batch)
        elapsed = time.time() - t0
        speed   = end / elapsed
        eta     = (N - end) / speed if speed > 0 else 0
        print(f"  {end:>12,} / {N:,}  ({speed/1e6:.2f}M vec/s  ETA {eta/60:.1f}min)", end="\r")

    print(f"\n  All {N:,} vectors added in {(time.time()-t0)/60:.1f}min")

    # ── Sanity check ──────────────────────────────────────────────────────────
    print("\nSanity check...")
    query    = np.ascontiguousarray(binary[0:1])
    D, I     = index.search(query, k=5)
    assert 0 in I[0], f"Self-match missing! Got {I[0]}"
    print(f"  Top-5 indices: {I[0]}  ✓ (index 0 present)")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nWriting index to {INDEX_PATH}...")
    t0 = time.time()
    faiss.write_index_binary(index, str(INDEX_PATH))
    size_gb = INDEX_PATH.stat().st_size / 1e9
    print(f"  Saved {size_gb:.2f} GB in {time.time()-t0:.1f}s")
    print(f"\nDone. {N:,} vectors indexed. Query nprobe={NPROBE} for best recall.")


if __name__ == "__main__":
    main()
