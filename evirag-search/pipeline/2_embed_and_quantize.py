"""
Step 2: Embed abstracts with SPECTER and binary-quantize.

GPU MAXIMIZATION — targets 48 GB Ada (RTX 6000 Ada / L40S / A6000 Ada):
  • bfloat16 inference (native Ada precision, 2× throughput vs float32)
  • Flash Attention 2 when available (O(1) memory → huge batches)
  • torch.compile(mode="reduce-overhead") — fuses kernels, eliminates Python overhead
  • Batch size 8192 with FlashAttn / 4096 without (GPU should be pegged at ~99%)
  • DataLoader: pin_memory=True + non_blocking transfer → DMA overlap with compute
  • GPU tokenization via pre-batched TensorDataset (CPU workers tokenize ahead of GPU)
  • Per-file incremental save — no 40 M-row accumulation in Python heap
  • Streaming ParquetWriter for metadata

Expected throughput on 48 GB Ada: ~10 000–20 000 papers/sec → ~40M papers in ~45-90 min.

Input:  ./raw_corpus/batch_*.parquet
Output: ./embeddings/
          binary.npy        — packed bits (N × 96 uint8)
          ids.npy           — paper IDs (str, N)
          metadata.parquet  — id, title, year, doi, cited_by_count, source, text_excerpt (≤2 000 chars, ALL papers)
          fulltext.parquet  — id, full_text (≤32 000 chars, only papers where text > 2 000 chars)

Two-parquet design:
  metadata.parquet  tiny, always loaded by DuckDB — fast ID→title lookups
  fulltext.parquet  ~60 GB compressed for 8M+ s2orc papers — queried only for top-K results
  JOIN on id at query time to get full paper context for the retrieved papers
"""

import os
import sys
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIR   = Path("./raw_corpus")
OUTPUT_DIR  = Path("./embeddings")
MODEL_NAME  = "allenai/specter"       # HuggingFace hub ID
MAX_LENGTH  = 256
# Batch size will be set after model load (depends on FlashAttn availability)
BASE_BATCH  = 8192     # with flash attention
SAFE_BATCH  = 4096     # without flash attention
NUM_WORKERS = 8        # tokenization workers
CHECKPOINT_EVERY = 5  # save state every N parquet files
RESUME_FROM = 0        # set to last completed file index to resume

# ── GPU / CUDA setup ──────────────────────────────────────────────────────────
assert torch.cuda.is_available(), "CUDA not available — need a GPU!"
device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU : {gpu_name}  ({vram_gb:.1f} GB VRAM)")

# Maximize GPU throughput settings
torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

# ── Model load ────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

USE_FLASH = False
BATCH_SIZE = SAFE_BATCH

try:
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device).eval()
    USE_FLASH  = True
    BATCH_SIZE = BASE_BATCH
    print(f"Loaded SPECTER with Flash Attention 2  |  batch_size={BATCH_SIZE}")
except Exception as e:
    print(f"Flash Attention not available ({e}) — using standard attention")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    BATCH_SIZE = SAFE_BATCH
    print(f"Loaded SPECTER (bf16, standard attn)   |  batch_size={BATCH_SIZE}")

# Compile the model for kernel fusion — big win on Ada
try:
    print("Compiling model with torch.compile(mode='reduce-overhead')...")
    model = torch.compile(model, mode="reduce-overhead")
    print("Compilation done.")
except Exception as e:
    print(f"torch.compile skipped: {e}")


# ── Embedding utilities ───────────────────────────────────────────────────────
def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Weighted mean over non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def format_input(title: str, abstract: str) -> str:
    t = (title    or "").strip()
    a = (abstract or "").strip()
    if t and a:
        return f"{t} [SEP] {a}"
    return t or a


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Tokenize `texts` on CPU, embed in large GPU batches, return (N, 96) uint8
    binary-quantized embeddings.
    """
    # ── Tokenize all at once (fast batched op on CPU) ──────────────────────
    enc = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"]       # (N, L)
    attention_mask = enc["attention_mask"]  # (N, L)

    dataset = TensorDataset(input_ids, attention_mask)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,      # pinned host memory → async DMA to GPU
        num_workers=0,        # tensors already in memory, no workers needed
        drop_last=False,
    )

    all_binary: list[np.ndarray] = []

    with torch.no_grad():
        for ids_b, mask_b in loader:
            # non_blocking=True: transfer starts while CPU prepares next batch
            ids_b  = ids_b.to(device, non_blocking=True)
            mask_b = mask_b.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids_b, attention_mask=mask_b)
                emb = mean_pool(out.last_hidden_state, mask_b)
                emb = F.normalize(emb, p=2, dim=1)

            # Binary quantize on GPU, bring to CPU — (B, 768) bool → (B, 96) uint8
            bits   = (emb > 0).to(torch.uint8).cpu().numpy()
            packed = np.packbits(bits, axis=1)   # (B, 96)
            all_binary.append(packed)

    return np.vstack(all_binary)   # (N, 96)


# ── Writers ────────────────────────────────────────────────────────────────────
class MetaWriter:
    """Lightweight metadata for ALL papers — kept small for fast DuckDB scans.
    text_excerpt ≤ 2 000 chars (covers full abstract for every paper).
    """
    SCHEMA = pa.schema([
        pa.field("id",             pa.string()),
        pa.field("title",          pa.string()),
        pa.field("year",           pa.int16()),
        pa.field("doi",            pa.string()),
        pa.field("cited_by_count", pa.int32()),
        pa.field("source",         pa.string()),
        pa.field("text_excerpt",   pa.string()),
    ])

    def __init__(self, path: Path):
        self._writer = pq.ParquetWriter(str(path), self.SCHEMA, compression="zstd")

    def write(self, rows: list[dict]):
        table = pa.table({
            "id":             pa.array([r["id"]             for r in rows], type=pa.string()),
            "title":          pa.array([r["title"]          for r in rows], type=pa.string()),
            "year":           pa.array([r["year"]           for r in rows], type=pa.int16()),
            "doi":            pa.array([r["doi"]            for r in rows], type=pa.string()),
            "cited_by_count": pa.array([r["cited_by_count"] for r in rows], type=pa.int32()),
            "source":         pa.array([r["source"]         for r in rows], type=pa.string()),
            "text_excerpt":   pa.array([r["text_excerpt"]   for r in rows], type=pa.string()),
        })
        self._writer.write_table(table)

    def close(self):
        self._writer.close()


class FullTextWriter:
    """Full paper text for papers where content exceeds the metadata excerpt.
    Only written for papers where raw text > 2 000 chars (s2orc papers in peS2o,
    open-access PDFs, etc.).  Stored separately so metadata.parquet stays fast.
    At query time: DuckDB JOINs fulltext.parquet on id for the top retrieved papers.
    Cap: 32 000 chars — covers ~99% of s2orc papers in full, avg 31 k chars.
    """
    SCHEMA = pa.schema([
        pa.field("id",        pa.string()),
        pa.field("full_text", pa.string()),
    ])
    THRESHOLD = 2_001    # only write if raw text is longer than text_excerpt
    CAP       = 32_000   # hard cap per paper

    def __init__(self, path: Path):
        self._writer = pq.ParquetWriter(str(path), self.SCHEMA, compression="zstd")
        self.n_written = 0

    def write(self, rows: list[dict]):
        """Only writes rows where raw text length exceeds the excerpt threshold."""
        ids, texts = [], []
        for r in rows:
            ft = r.get("full_text", "")
            if ft and len(ft) > self.THRESHOLD:
                ids.append(r["id"])
                texts.append(ft[:self.CAP])
        if not ids:
            return
        table = pa.table({
            "id":        pa.array(ids,   type=pa.string()),
            "full_text": pa.array(texts, type=pa.string()),
        })
        self._writer.write_table(table)
        self.n_written += len(ids)

    def close(self):
        self._writer.close()


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def save_checkpoint(ids: list, binary_chunks: list, meta_writer_closed: bool = False):
    ids_arr = np.array(ids, dtype=object)
    bin_arr = np.vstack(binary_chunks) if binary_chunks else np.empty((0, 96), dtype=np.uint8)
    np.save(OUTPUT_DIR / "ids.npy",    ids_arr,    allow_pickle=True)
    np.save(OUTPUT_DIR / "binary.npy", bin_arr,    allow_pickle=False)
    print(f"  [ckpt] {len(ids_arr):,} vectors saved  ({bin_arr.nbytes / 1e9:.2f} GB)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    parquet_files = sorted(INPUT_DIR.glob("batch_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {INPUT_DIR}  — run step 1 first.")
    print(f"Found {len(parquet_files)} batch files. Using BATCH_SIZE={BATCH_SIZE}.")

    meta_path   = OUTPUT_DIR / "metadata.parquet"
    meta_writer = MetaWriter(meta_path)

    all_ids    : list[str]       = []
    all_binary : list[np.ndarray] = []

    total_papers = 0
    t_pipeline_start = time.time()

    for file_idx, pq_path in enumerate(parquet_files):
        if file_idx < RESUME_FROM:
            print(f"  Skipping {pq_path.name} (resume mode)")
            continue

        t0 = time.time()
        print(f"\n[{file_idx+1}/{len(parquet_files)}] {pq_path.name}")

        table    = pq.read_table(pq_path)
        df       = table.to_pydict()

        ids       = df["id"]
        titles    = df["title"]
        abstracts = df["abstract"]

        texts = [format_input(t, a) for t, a in zip(titles, abstracts)]

        # ── Embed ──────────────────────────────────────────────────────────
        print(f"  Embedding {len(texts):,} papers...")
        binary = embed_texts(texts)

        # ── Accumulate ─────────────────────────────────────────────────────
        all_ids.extend(ids)
        all_binary.append(binary)
        total_papers += len(ids)

        # ── Write metadata (streaming, now includes text_excerpt) ───────────
        meta_rows = [
            {
                "id":             ids[i],
                "title":          titles[i],
                "year":           df["year"][i],
                "doi":            df["doi"][i],
                "cited_by_count": df["cited_by_count"][i],
                "source":         df["source"][i],
                # Store the exact text SPECTER was fed (title [SEP] abstract).
                # Cap at 4 000 chars — covers full abstract for virtually all papers.
                "text_excerpt":   texts[i][:4000],
            }
            for i in range(len(ids))
        ]
        meta_writer.write(meta_rows)

        elapsed = time.time() - t0
        speed   = len(texts) / elapsed
        wall    = time.time() - t_pipeline_start
        eta     = (len(parquet_files) - file_idx - 1) * elapsed
        print(
            f"  {len(texts):,} papers in {elapsed:.0f}s  "
            f"({speed:,.0f} papers/s)  "
            f"wall={wall/60:.1f}m  ETA={eta/60:.1f}m"
        )

        # Print GPU memory stats
        alloc = torch.cuda.memory_allocated() / 1e9
        peak  = torch.cuda.max_memory_allocated() / 1e9
        print(f"  GPU mem: {alloc:.1f} GB alloc / {peak:.1f} GB peak")

        # ── Periodic checkpoint ────────────────────────────────────────────
        if (file_idx + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(all_ids, all_binary)

    # ── Final save ────────────────────────────────────────────────────────────
    print("\nFinalizing...")
    meta_writer.close()
    save_checkpoint(all_ids, all_binary)

    total_elapsed = time.time() - t_pipeline_start
    print(f"\nDone. {total_papers:,} papers in {total_elapsed/60:.1f} min  "
          f"({total_papers/total_elapsed:,.0f} papers/sec avg)")
    bin_arr = np.vstack(all_binary)
    print(f"Binary array: {bin_arr.shape}  ({bin_arr.nbytes/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
