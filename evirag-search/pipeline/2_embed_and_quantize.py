"""
Step 2: Embed abstracts with SPECTER2 and binary-quantize.

Input:  ./raw_corpus/batch_*.parquet
Output: ./embeddings/
          ids.npy            — OpenAlex paper IDs (str, N)
          binary.npy         — binary vectors (uint8, N × 96)  [768 dims / 8]
          metadata.parquet   — id, title, year, doi, cited_by_count, source

SPECTER2 produces 768-dim embeddings tuned for scientific papers.
Binary quantization: sign(embedding) → 1 bit per dimension → 96 bytes/vector.
At 40M papers: 40M × 96 bytes ≈ 3.8 GB.

Run on a machine with a GPU if possible (10x faster).
CPU: ~2-3 days for 40M papers.  GPU (T4): ~4-6 hours.
"""

import os
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIR   = Path("./raw_corpus")
OUTPUT_DIR  = Path("./embeddings")
MODEL_NAME  = "allenai-specter"          # 768-dim, scientific papers
BATCH_SIZE  = 512 if torch.cuda.is_available() else 64
MAX_LENGTH  = 256                         # title + abstract, truncated
RESUME_FROM = 0                           # set to last completed batch to resume

# ── Setup ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}  |  Batch size: {BATCH_SIZE}")

model = SentenceTransformer(MODEL_NAME, device=device)
model.max_seq_length = MAX_LENGTH


def to_binary(embeddings: np.ndarray) -> np.ndarray:
    """Convert float32 (N, 768) → uint8 (N, 96) via sign-bit packing."""
    bits = (embeddings > 0).astype(np.uint8)
    return np.packbits(bits, axis=1)


def format_input(title: str, abstract: str) -> str:
    """SPECTER format: title [SEP] abstract."""
    title    = (title    or "").strip()
    abstract = (abstract or "").strip()
    if title and abstract:
        return f"{title} [SEP] {abstract}"
    return title or abstract


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parquet_files = sorted(INPUT_DIR.glob("batch_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {INPUT_DIR}")
    print(f"Found {len(parquet_files)} batch files.")

    all_ids      : list[str]        = []
    all_binary   : list[np.ndarray] = []
    all_meta_rows: list[dict]       = []

    for file_idx, parquet_path in enumerate(parquet_files):
        if file_idx < RESUME_FROM:
            print(f"Skipping {parquet_path.name} (resume mode)")
            continue

        print(f"\n[{file_idx+1}/{len(parquet_files)}] {parquet_path.name}")
        table = pq.read_table(parquet_path)
        df    = table.to_pydict()

        ids       = df["id"]
        titles    = df["title"]
        abstracts = df["abstract"]
        years     = df["year"]
        dois      = df["doi"]
        cites     = df["cited_by_count"]
        sources   = df["source"]

        texts = [format_input(t, a) for t, a in zip(titles, abstracts)]

        # Embed in batches
        print(f"  Embedding {len(texts):,} papers...")
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,   # unit sphere → cosine = dot product
            convert_to_numpy=True,
        )

        binary = to_binary(embeddings)

        all_ids.extend(ids)
        all_binary.append(binary)
        all_meta_rows.extend([
            {
                "id":            ids[i],
                "title":         titles[i],
                "year":          years[i],
                "doi":           dois[i],
                "cited_by_count": cites[i],
                "source":        sources[i],
            }
            for i in range(len(ids))
        ])

        # Checkpoint every 5 files to avoid losing work
        if (file_idx + 1) % 5 == 0 or file_idx == len(parquet_files) - 1:
            print("  Checkpointing...")
            _save(all_ids, all_binary, all_meta_rows)

    print("\nFinalizing outputs...")
    _save(all_ids, all_binary, all_meta_rows)
    print(f"Done. {len(all_ids):,} papers embedded.")
    print(f"Binary array size: {np.vstack(all_binary).nbytes / 1e9:.2f} GB")


def _save(ids, binary_chunks, meta_rows):
    ids_arr    = np.array(ids, dtype=object)
    binary_arr = np.vstack(binary_chunks) if binary_chunks else np.empty((0, 96), dtype=np.uint8)

    np.save(OUTPUT_DIR / "ids.npy",    ids_arr,    allow_pickle=True)
    np.save(OUTPUT_DIR / "binary.npy", binary_arr, allow_pickle=False)

    meta_table = pa.table({
        "id":            pa.array([r["id"]            for r in meta_rows], type=pa.string()),
        "title":         pa.array([r["title"]         for r in meta_rows], type=pa.string()),
        "year":          pa.array([r["year"]          for r in meta_rows], type=pa.int16()),
        "doi":           pa.array([r["doi"]           for r in meta_rows], type=pa.string()),
        "cited_by_count": pa.array([r["cited_by_count"] for r in meta_rows], type=pa.int32()),
        "source":        pa.array([r["source"]        for r in meta_rows], type=pa.string()),
    })
    pq.write_table(meta_table, OUTPUT_DIR / "metadata.parquet", compression="zstd")
    print(f"  Checkpointed {len(ids):,} papers to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
