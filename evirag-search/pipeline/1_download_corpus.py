"""
Step 1: Download scientific papers from OpenAlex API.

Filters:
  - has_abstract: true
  - cited_by_count: >5   (removes junk/predatory)
  - language: en
  - publication_year: 2000-2025

Output: ./raw_corpus/batch_NNNN.parquet  (200k papers each)
State:  ./raw_corpus/cursor.txt           (resume support)

Runtime estimate: ~5-8 hours for 40M papers at 10 req/sec.
Run once, then move to step 2.
"""

import os
import json
import time
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("./raw_corpus")
BATCH_SIZE = 200_000          # papers per Parquet file
PAGE_SIZE  = 200              # max per API call
MAILTO     = "animeshmishra0567@gmail.com"   # polite pool: ~10 req/sec

FILTER = (
    "has_abstract:true,"
    "cited_by_count:>5,"
    "language:en,"
    "publication_year:2000-2025"
)

BASE_URL = (
    f"https://api.openalex.org/works"
    f"?filter={FILTER}"
    f"&per-page={PAGE_SIZE}"
    f"&select=id,doi,title,abstract_inverted_index,publication_year,cited_by_count,primary_location"
    f"&mailto={MAILTO}"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """OpenAlex stores abstracts as inverted index {word: [positions]}. Rebuild."""
    if not inverted_index:
        return None
    word_pos = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_pos.append((pos, word))
    word_pos.sort()
    return " ".join(w for _, w in word_pos)


def load_cursor() -> str:
    cursor_file = OUTPUT_DIR / "cursor.txt"
    if cursor_file.exists():
        cursor = cursor_file.read_text().strip()
        print(f"Resuming from cursor: {cursor[:40]}...")
        return cursor
    return "*"


def save_cursor(cursor: str):
    (OUTPUT_DIR / "cursor.txt").write_text(cursor)


def save_batch(records: list[dict], batch_num: int):
    path = OUTPUT_DIR / f"batch_{batch_num:04d}.parquet"
    table = pa.table({
        "id":           pa.array([r["id"]           for r in records], type=pa.string()),
        "doi":          pa.array([r["doi"]           for r in records], type=pa.string()),
        "title":        pa.array([r["title"]         for r in records], type=pa.string()),
        "abstract":     pa.array([r["abstract"]      for r in records], type=pa.string()),
        "year":         pa.array([r["year"]          for r in records], type=pa.int16()),
        "cited_by_count": pa.array([r["cited_by_count"] for r in records], type=pa.int32()),
        "source":       pa.array([r["source"]        for r in records], type=pa.string()),
    })
    pq.write_table(table, path, compression="zstd")
    print(f"  Saved {len(records):,} papers → {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Count existing batch files to continue numbering
    existing = sorted(OUTPUT_DIR.glob("batch_*.parquet"))
    batch_num = len(existing)
    total_downloaded = batch_num * BATCH_SIZE
    print(f"Starting. Existing batches: {batch_num}, ~{total_downloaded:,} papers already downloaded.")

    cursor = load_cursor()
    buffer: list[dict] = []
    session = requests.Session()
    session.headers.update({"User-Agent": f"EVIRAG-Research/1.0 (mailto:{MAILTO})"})

    while True:
        url = f"{BASE_URL}&cursor={cursor}"
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"Request error: {e} — retrying in 5s")
            time.sleep(5)
            continue

        results = data.get("results", [])
        if not results:
            print("No more results. Download complete.")
            break

        for work in results:
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
            if not abstract:
                continue  # skip if abstract missing despite filter (API lag)

            source = ""
            pl = work.get("primary_location") or {}
            src = pl.get("source") or {}
            source = src.get("display_name") or ""

            buffer.append({
                "id":            work.get("id", ""),
                "doi":           work.get("doi") or "",
                "title":         (work.get("title") or "").strip(),
                "abstract":      abstract.strip(),
                "year":          work.get("publication_year") or 0,
                "cited_by_count": work.get("cited_by_count") or 0,
                "source":        source,
            })

        total_downloaded += len(results)

        if len(buffer) >= BATCH_SIZE:
            save_batch(buffer[:BATCH_SIZE], batch_num)
            buffer = buffer[BATCH_SIZE:]
            batch_num += 1

        meta = data.get("meta", {})
        next_cursor = meta.get("next_cursor")
        if not next_cursor:
            break
        cursor = next_cursor
        save_cursor(cursor)

        count = meta.get("count", "?")
        print(f"  Progress: {total_downloaded:,} / {count:,}  (buffer: {len(buffer):,})", end="\r")
        time.sleep(0.11)  # ~9 req/sec to stay in polite pool

    # Flush remaining
    if buffer:
        save_batch(buffer, batch_num)

    print(f"\nDone. Total batches: {batch_num + 1}, total papers: ~{total_downloaded:,}")
    # Clean up cursor so re-run starts fresh
    (OUTPUT_DIR / "cursor.txt").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
