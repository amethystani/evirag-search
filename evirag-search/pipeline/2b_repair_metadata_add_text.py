"""
2b_repair_metadata_add_text.py
───────────────────────────────
Run this ONCE on the server after step 2 has already finished WITHOUT text_excerpt.

Problem: MetaWriter in step 2 dropped the abstract column.
Fix:     Re-read all raw_corpus batch files, join on paper ID,
         write a new metadata.parquet that includes text_excerpt.

The raw_corpus/batch_*.parquet files are still on disk and contain
the original abstract text — we never need to re-embed anything.

Runtime: ~5-10 min for 40M papers (pure pandas/pyarrow, no GPU needed).

Usage:
    cd /home/snu/evirag-search/evirag-search
    python pipeline/2b_repair_metadata_add_text.py
"""

import time
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR      = Path("./raw_corpus")       # has batch_*.parquet with 'abstract'
EMB_DIR      = Path("./embeddings")       # has metadata.parquet WITHOUT text_excerpt
OUT_PATH     = EMB_DIR / "metadata.parquet"   # will be overwritten in-place
BACKUP_PATH  = EMB_DIR / "metadata_no_text.parquet"
MAX_TEXT_LEN = 4000                       # chars — covers full abstract for all papers
BATCH_ROWS   = 500_000                    # rows per write batch (memory safety)

# ── Schema with text_excerpt ──────────────────────────────────────────────────
NEW_SCHEMA = pa.schema([
    pa.field("id",             pa.string()),
    pa.field("title",          pa.string()),
    pa.field("year",           pa.int16()),
    pa.field("doi",            pa.string()),
    pa.field("cited_by_count", pa.int32()),
    pa.field("source",         pa.string()),
    pa.field("text_excerpt",   pa.string()),
])


def main():
    t0 = time.time()

    # ── 1. Build id → abstract map from raw_corpus ───────────────────────────
    print("Step 1: Building id→abstract lookup from raw_corpus...")
    raw_files = sorted(RAW_DIR.glob("batch_*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"No batch_*.parquet in {RAW_DIR}")
    print(f"  Found {len(raw_files)} raw batch files")

    abstract_map: dict[str, str] = {}
    for i, f in enumerate(raw_files, 1):
        tbl = pq.read_table(f, columns=["id", "title", "abstract"])
        d   = tbl.to_pydict()
        for pid, title, abstract in zip(d["id"], d["title"], d["abstract"]):
            # Reproduce exactly what format_input() gave SPECTER
            t = (title    or "").strip()
            a = (abstract or "").strip()
            if t and a:
                text = f"{t} [SEP] {a}"
            else:
                text = t or a
            abstract_map[pid] = text[:MAX_TEXT_LEN]
        print(f"  [{i}/{len(raw_files)}] {f.name}  cumulative={len(abstract_map):,}", end="\r")

    print(f"\n  Abstract map: {len(abstract_map):,} papers  ({(time.time()-t0)/60:.1f} min)")

    # ── 2. Backup old metadata ─────────────────────────────────────────────────
    if OUT_PATH.exists() and not BACKUP_PATH.exists():
        import shutil
        print(f"\nStep 2: Backing up old metadata → {BACKUP_PATH.name}")
        shutil.copy2(str(OUT_PATH), str(BACKUP_PATH))
        print(f"  Backup size: {BACKUP_PATH.stat().st_size/1e9:.2f} GB")

    # ── 3. Re-write metadata with text_excerpt ────────────────────────────────
    print(f"\nStep 3: Writing new metadata.parquet with text_excerpt...")
    old_tbl = pq.read_table(str(BACKUP_PATH) if BACKUP_PATH.exists() else str(OUT_PATH))
    old     = old_tbl.to_pydict()
    N       = len(old["id"])
    print(f"  Total rows to process: {N:,}")

    writer = pq.ParquetWriter(str(OUT_PATH), NEW_SCHEMA, compression="zstd")
    written = 0
    missing = 0

    for start in range(0, N, BATCH_ROWS):
        end = min(start + BATCH_ROWS, N)

        ids             = old["id"][start:end]
        titles          = old["title"][start:end]
        years           = old["year"][start:end]
        dois            = old["doi"][start:end]
        cited           = old["cited_by_count"][start:end]
        sources         = old["source"][start:end]
        text_excerpts   = []

        for pid, title in zip(ids, titles):
            text = abstract_map.get(pid)
            if text:
                text_excerpts.append(text)
            else:
                # Paper not in raw_corpus (shouldn't happen) → use title only
                text_excerpts.append((title or "").strip()[:MAX_TEXT_LEN])
                missing += 1

        batch = pa.table({
            "id":             pa.array(ids,    type=pa.string()),
            "title":          pa.array(titles, type=pa.string()),
            "year":           pa.array(years,  type=pa.int16()),
            "doi":            pa.array(dois,   type=pa.string()),
            "cited_by_count": pa.array(cited,  type=pa.int32()),
            "source":         pa.array(sources,type=pa.string()),
            "text_excerpt":   pa.array(text_excerpts, type=pa.string()),
        })
        writer.write_table(batch)
        written += (end - start)
        elapsed = time.time() - t0
        print(f"  {written:>12,} / {N:,}  ({elapsed/60:.1f} min)", end="\r")

    writer.close()

    total = time.time() - t0
    size  = OUT_PATH.stat().st_size / 1e9
    print(f"\n\nDone!")
    print(f"  Papers written:  {written:,}")
    print(f"  Missing text:    {missing:,}  (fell back to title-only)")
    print(f"  Output size:     {size:.2f} GB")
    print(f"  Total time:      {total/60:.1f} min")
    print(f"\nNext step: run pipeline/4_upload_to_hf.py to push the new metadata to HF.")


if __name__ == "__main__":
    main()
