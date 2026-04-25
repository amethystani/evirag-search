"""
Step 1: Download scientific papers from OpenAlex API.

Optimized: async aiohttp pipeline — HTTP fetch, JSON parse, and parquet write
run concurrently so no wall-clock time is wasted waiting on disk I/O.

Filters:
  - has_abstract: true
  - cited_by_count: >5   (removes junk/predatory)
  - language: en
  - publication_year: 2000-2025

Output: ./raw_corpus/batch_NNNN.parquet  (200k papers each)
State:  ./raw_corpus/cursor.txt           (resume support)

Runtime estimate: ~5-8 hours (OpenAlex polite-pool is 10 req/sec, hard limit).
"""

import asyncio
import os
import time
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

try:
    import orjson as json_lib
    _loads = lambda b: json_lib.loads(b)
except ImportError:
    import json as json_lib
    _loads = json_lib.loads

import aiohttp

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("./raw_corpus")
BATCH_SIZE  = 200_000
PAGE_SIZE   = 200
MAILTO      = "animeshmishra0567@gmail.com"
MAX_RETRIES = 8
# Stay safely under 10 req/sec polite-pool limit with a sliding window
REQ_INTERVAL = 0.12   # ~8.3 req/sec — leaves headroom

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

HEADERS = {"User-Agent": f"EVIRAG-Research/1.0 (mailto:{MAILTO})"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def reconstruct_abstract(inverted_index: dict | None) -> str | None:
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
        "id":             pa.array([r["id"]            for r in records], type=pa.string()),
        "doi":            pa.array([r["doi"]           for r in records], type=pa.string()),
        "title":          pa.array([r["title"]         for r in records], type=pa.string()),
        "abstract":       pa.array([r["abstract"]      for r in records], type=pa.string()),
        "year":           pa.array([r["year"]          for r in records], type=pa.int16()),
        "cited_by_count": pa.array([r["cited_by_count"] for r in records], type=pa.int32()),
        "source":         pa.array([r["source"]        for r in records], type=pa.string()),
    })
    pq.write_table(table, path, compression="zstd")
    print(f"\n  Saved {len(records):,} papers → {path.name}")


def parse_results(results: list) -> list[dict]:
    records = []
    for work in results:
        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
        if not abstract:
            continue
        pl  = work.get("primary_location") or {}
        src = (pl.get("source") or {}).get("display_name") or ""
        records.append({
            "id":            work.get("id", ""),
            "doi":           work.get("doi") or "",
            "title":         (work.get("title") or "").strip(),
            "abstract":      abstract.strip(),
            "year":          work.get("publication_year") or 0,
            "cited_by_count": work.get("cited_by_count") or 0,
            "source":        src,
        })
    return records


# ── Async fetch with retry ────────────────────────────────────────────────────
async def fetch_page(session: aiohttp.ClientSession, url: str) -> dict:
    backoff = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=45)) as resp:
                if resp.status == 429:
                    wait = float(resp.headers.get("Retry-After", backoff))
                    print(f"\n  Rate limited — sleeping {wait:.0f}s")
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                raw = await resp.read()
                return _loads(raw)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"\n  Error (attempt {attempt+1}): {e} — retry in {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
    raise RuntimeError("Max retries exceeded")


# ── Main async loop ───────────────────────────────────────────────────────────
async def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    existing   = sorted(OUTPUT_DIR.glob("batch_*.parquet"))
    batch_num  = len(existing)
    total_done = batch_num * BATCH_SIZE
    print(f"Starting. Existing batches: {batch_num} (~{total_done:,} papers already done).")

    cursor  = load_cursor()
    buffer: list[dict] = []

    connector = aiohttp.TCPConnector(limit=4, ttl_dns_cache=300)
    async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:
        t_last = time.monotonic()

        while True:
            url = f"{BASE_URL}&cursor={cursor}"

            # ── Rate-limit pacing ──────────────────────────────────────────
            elapsed = time.monotonic() - t_last
            wait    = max(0.0, REQ_INTERVAL - elapsed)
            if wait > 0:
                await asyncio.sleep(wait)
            t_last = time.monotonic()

            data = await fetch_page(session, url)

            results = data.get("results", [])
            if not results:
                print("\nNo more results. Download complete.")
                break

            # Parse off the hot path (CPU-bound but fast)
            records = parse_results(results)
            buffer.extend(records)
            total_done += len(results)

            if len(buffer) >= BATCH_SIZE:
                # Write is blocking I/O — run in executor to not block the loop
                loop = asyncio.get_event_loop()
                chunk, buffer = buffer[:BATCH_SIZE], buffer[BATCH_SIZE:]
                await loop.run_in_executor(None, save_batch, chunk, batch_num)
                batch_num += 1

            meta        = data.get("meta", {})
            next_cursor = meta.get("next_cursor")
            if not next_cursor:
                break

            cursor = next_cursor
            save_cursor(cursor)

            count = meta.get("count", "?")
            speed = total_done / (time.monotonic() - t_last + 1e-9)
            print(f"  Progress: {total_done:,} / {count:,}  ({int(speed):,} papers/s)", end="\r")

    # Flush remainder
    if buffer:
        save_batch(buffer, batch_num)
        batch_num += 1

    print(f"\nDone. Total batches: {batch_num}, total papers: ~{total_done:,}")
    (OUTPUT_DIR / "cursor.txt").unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
