"""
EVIRAG Local Pipeline — 100k full-text papers, peS2o v3, MPS-accelerated.

Source: AllenAI peS2o v3 (Semantic Scholar, ~8M papers, full text)
  https://huggingface.co/datasets/allenai/peS2o/tree/main/data/v3

Pipeline:
  1. Stream peS2o v3 shards — collect 100k papers balanced across 10 domains
  2. Embed full text with SPECTER on MPS (Apple Silicon GPU)
  3. Binary quantize → FAISS IndexBinaryIVF
  4. Upload to HF Datasets (amethystani/evirag-index)
  5. Sanity query

Timeline on Apple Silicon M-series:
  Stream shards    : ~15 min (stops as soon as quotas fill)
  Embed on MPS     : ~45 min (~37 papers/sec at BATCH_SIZE=64)
  Index + upload   : ~10 min
  Total            : ~1.2 hours

Run: python3 run_local_100k.py
"""

import io, os, json, time, warnings, random
warnings.filterwarnings("ignore")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import faiss
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import HfApi, create_repo, hf_hub_download

try:
    import zstandard as zstd
except ImportError:
    raise SystemExit("Install zstandard first:  pip3 install zstandard")

# ── Config ────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN   = os.environ.get("HF_TOKEN")
REPO_ID    = "amethystani/evirag-index"
OUT_DIR    = Path("./local_100k")
TARGET     = 100_000      # total papers to collect
BATCH_SIZE = 64           # MPS embedding batch size
MAX_TOKENS = 512          # SPECTER max input length
TEXT_CHARS = 4_000        # chars to keep per paper (well past 512 tokens)
N_SHARDS   = 136          # peS2o v3 total shards

# Domains from Semantic Scholar taxonomy — 10k papers each
TARGET_DOMAINS = [
    "Computer Science",
    "Mathematics",
    "Physics",
    "Biology",
    "Medicine",
    "Chemistry",
    "Engineering",
    "Economics",
    "Psychology",
    "Environmental Science",
]
QUOTA = TARGET // len(TARGET_DOMAINS)   # 10 000 per domain

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Device: Apple MPS (Metal GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Device: CUDA — {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("Device: CPU (slow — expect 3-4 hrs)")

OUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Stream peS2o v3
# ══════════════════════════════════════════════════════════════════════════════

def step1_download() -> list[dict]:
    print(f"\n{'='*60}")
    print("STEP 1: Stream peS2o v3 — 100k papers, 10 domains")
    print(f"{'='*60}")
    t0 = time.time()

    # Build shard list and shuffle so we get domain diversity immediately
    # (early shards may over-represent certain fields)
    shard_names = [f"data/v3/train-{i:04d}-of-{N_SHARDS:04d}.zst" for i in range(N_SHARDS)]
    random.shuffle(shard_names)

    domain_counts: dict[str, int] = {d: 0 for d in TARGET_DOMAINS}
    papers: list[dict] = []

    for shard_idx, shard_path in enumerate(shard_names):
        total_so_far = sum(domain_counts.values())
        if total_so_far >= TARGET:
            break

        # Skip domains already full — if all full we're done
        open_domains = {d for d, n in domain_counts.items() if n < QUOTA}
        if not open_domains:
            break

        print(f"  Shard {shard_idx+1}/{N_SHARDS}: {shard_path.split('/')[-1]}"
              f"  [{total_so_far:,}/{TARGET:,}]")

        # Download shard to HF cache (~/.cache/huggingface/hub)
        try:
            local_path = hf_hub_download(
                repo_id="allenai/peS2o",
                filename=shard_path,
                repo_type="dataset",
                token=HF_TOKEN,
            )
        except Exception as e:
            print(f"    ⚠ Download failed: {e} — skipping")
            continue

        shard_added = 0
        dctx = zstd.ZstdDecompressor()

        try:
            with open(local_path, "rb") as fh:
                with dctx.stream_reader(fh) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                    for line in text_stream:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Pick first target domain with room left
                        fields = (rec.get("metadata") or {}).get("s2fieldsofstudy") or []
                        domain = None
                        for f in fields:
                            if f in open_domains:
                                domain = f
                                break
                        if domain is None:
                            continue

                        # Validate text
                        full_text = rec.get("text") or ""
                        if len(full_text) < 200:
                            continue

                        meta   = rec.get("metadata") or {}
                        title  = (meta.get("title") or "").strip()
                        if not title:
                            title = full_text.split("\n")[0][:200].strip()
                        ext_ids = meta.get("external_ids") or {}

                        papers.append({
                            "id":             f"pes2o:{rec['id']}",
                            "doi":            ext_ids.get("DOI") or "",
                            "title":          title,
                            "full_text":      full_text[:TEXT_CHARS],
                            "year":           int(meta.get("year") or 0),
                            "cited_by_count": 0,
                            "source":         "peS2o/Semantic Scholar",
                            "domain":         domain,
                        })
                        domain_counts[domain] += 1
                        shard_added += 1
                        open_domains = {d for d, n in domain_counts.items() if n < QUOTA}

                        if sum(domain_counts.values()) >= TARGET:
                            break
        except Exception as e:
            print(f"    ⚠ Parse error: {e} — skipping shard")
            continue

        print(f"    +{shard_added:,} papers", end="  |  ")
        print("  ".join(f"{d[:4]}:{n:,}" for d, n in sorted(domain_counts.items()) if n > 0))

    total = len(papers)
    elapsed = time.time() - t0
    print(f"\nCollected {total:,} papers in {elapsed/60:.1f} min")
    print("Domain breakdown:")
    for d, n in sorted(domain_counts.items(), key=lambda x: -x[1]):
        if n > 0:
            bar = "█" * (n // 500)
            print(f"  {d:25s} {n:6,}  {bar}")

    # Save full corpus (for inspection / reuse)
    table = pa.table({
        "id":            pa.array([p["id"]            for p in papers]),
        "doi":           pa.array([p["doi"]           for p in papers]),
        "title":         pa.array([p["title"]         for p in papers]),
        "full_text":     pa.array([p["full_text"]     for p in papers]),
        "year":          pa.array([p["year"]          for p in papers], type=pa.int16()),
        "cited_by_count":pa.array([p["cited_by_count"]for p in papers], type=pa.int32()),
        "source":        pa.array([p["source"]        for p in papers]),
        "domain":        pa.array([p["domain"]        for p in papers]),
    })
    pq.write_table(table, OUT_DIR / "corpus.parquet", compression="zstd")
    print(f"Saved: {OUT_DIR}/corpus.parquet")
    return papers


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Embed full text on MPS
# ══════════════════════════════════════════════════════════════════════════════

def step2_embed(papers: list[dict]) -> np.ndarray:
    print(f"\n{'='*60}")
    print(f"STEP 2: Embed {len(papers):,} papers on {DEVICE}")
    print(f"{'='*60}")

    print("Loading SPECTER...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
    model     = AutoModel.from_pretrained("allenai/specter").to(DEVICE).eval()
    print("SPECTER ready.")

    def make_input(p: dict) -> str:
        # SPECTER canonical format: "title [SEP] body"
        body = p["full_text"][:3_000].replace("\n", " ").strip()
        return f"{p['title']} [SEP] {body}"

    texts = [make_input(p) for p in papers]
    N     = len(texts)
    all_binary: list[np.ndarray] = []
    t0 = time.time()

    with torch.no_grad():
        for start in range(0, N, BATCH_SIZE):
            end   = min(start + BATCH_SIZE, N)
            batch = texts[start:end]

            enc  = tokenizer(batch, max_length=MAX_TOKENS, padding=True,
                             truncation=True, return_tensors="pt")
            ids  = enc["input_ids"].to(DEVICE)
            mask = enc["attention_mask"].to(DEVICE)

            out  = model(input_ids=ids, attention_mask=mask)
            # Mean pooling (weighted by attention mask)
            emb  = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) \
                   / mask.sum(1, keepdim=True).clamp(min=1)
            emb  = F.normalize(emb, p=2, dim=1)

            # Binary quantize: sign bit → packbits → 96 bytes/vector for 768-dim
            bits   = (emb > 0).to(torch.uint8).cpu().numpy()
            binary = np.packbits(bits, axis=1)
            all_binary.append(binary)

            elapsed = time.time() - t0
            speed   = end / elapsed
            eta     = (N - end) / speed if speed > 0 else 0
            print(f"  {end:>6,}/{N:,}  {int(100*end/N):3d}%  "
                  f"{speed:.0f} papers/sec  ETA {eta/60:.1f}min", end="\r")

    print(f"\nEmbedding done in {(time.time()-t0)/60:.1f} min")
    return np.vstack(all_binary)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Build FAISS IndexBinaryIVF
# ══════════════════════════════════════════════════════════════════════════════

def step3_index(binary: np.ndarray, papers: list[dict]):
    print(f"\n{'='*60}")
    print("STEP 3: Build FAISS IndexBinaryIVF")
    print(f"{'='*60}")
    N, D_bytes = binary.shape
    DIM = D_bytes * 8   # 768 bits

    faiss.omp_set_num_threads(os.cpu_count() or 8)

    # nlist: rule of thumb = sqrt(N), capped to ensure enough training points
    nlist     = min(1024, max(64, N // 50))
    quantizer = faiss.IndexBinaryFlat(DIM)
    index     = faiss.IndexBinaryIVF(quantizer, DIM, nlist)
    index.nprobe = 64   # search 64 clusters at query time

    t0 = time.time()
    print(f"Training on {N:,} vectors (nlist={nlist})...")
    index.train(np.ascontiguousarray(binary))
    print(f"Adding {N:,} vectors...")
    index.add(np.ascontiguousarray(binary))
    print(f"Index built: {index.ntotal:,} vectors in {time.time()-t0:.1f}s")

    # Persist
    ids_arr = np.array([p["id"] for p in papers], dtype=object)
    np.save(OUT_DIR / "ids.npy",    ids_arr, allow_pickle=True)
    np.save(OUT_DIR / "binary.npy", binary,  allow_pickle=False)
    faiss.write_index_binary(index, str(OUT_DIR / "evirag.index"))

    meta = pa.table({
        "id":            pa.array([p["id"]               for p in papers]),
        "title":         pa.array([p["title"]             for p in papers]),
        "year":          pa.array([p["year"]              for p in papers], type=pa.int16()),
        "doi":           pa.array([p["doi"]               for p in papers]),
        "cited_by_count":pa.array([p["cited_by_count"]    for p in papers], type=pa.int32()),
        "source":        pa.array([p["source"]            for p in papers]),
        "domain":        pa.array([p["domain"]            for p in papers]),
        "text_excerpt":  pa.array([p["full_text"][:2_000] for p in papers]),
    })
    pq.write_table(meta, OUT_DIR / "metadata.parquet", compression="zstd")
    print("Saved: evirag.index, ids.npy, binary.npy, metadata.parquet")
    return index


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Upload to HF Datasets
# ══════════════════════════════════════════════════════════════════════════════

def step4_upload():
    print(f"\n{'='*60}")
    print("STEP 4: Upload to HF Datasets")
    print(f"{'='*60}")
    api = HfApi(token=HF_TOKEN)
    create_repo(REPO_ID, repo_type="dataset", private=False, exist_ok=True, token=HF_TOKEN)
    for fname in ["evirag.index", "ids.npy", "metadata.parquet"]:
        fpath = OUT_DIR / fname
        if not fpath.exists():
            print(f"  ⚠ {fname} not found — skipping")
            continue
        mb = fpath.stat().st_size / 1e6
        print(f"  {fname} ({mb:.0f} MB)...", end=" ", flush=True)
        api.upload_file(
            str(fpath), fname,
            repo_id=REPO_ID, repo_type="dataset",
            commit_message=f"100k peS2o full-text: {fname}",
        )
        print("✓")
    print(f"Live: https://huggingface.co/datasets/{REPO_ID}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Sanity query
# ══════════════════════════════════════════════════════════════════════════════

def step5_test(index, binary: np.ndarray, papers: list[dict]):
    print(f"\n{'='*60}")
    print("STEP 5: Sanity query")
    print(f"{'='*60}")
    q  = np.ascontiguousarray(binary[0:1])
    t0 = time.perf_counter()
    D, I = index.search(q, k=5)
    ms = (time.perf_counter() - t0) * 1000
    print(f"Query  : \"{papers[0]['title'][:65]}\"")
    print(f"Domain : {papers[0]['domain']}")
    print(f"Latency: {ms:.1f} ms")
    print("Top-5  :")
    for rank, idx in enumerate(I[0], 1):
        if 0 <= idx < len(papers):
            p = papers[idx]
            print(f"  #{rank} [{p['domain']:22s}] {p['title'][:55]}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_total = time.time()

    papers = step1_download()
    if not papers:
        raise SystemExit("No papers collected — check HF_TOKEN and network access.")

    binary = step2_embed(papers)
    index  = step3_index(binary, papers)
    step4_upload()
    step5_test(index, binary, papers)

    domains = len(set(p["domain"] for p in papers))
    idx_mb  = (OUT_DIR / "evirag.index").stat().st_size / 1e6

    print(f"\n{'='*60}")
    print(f"DONE in {(time.time()-t_total)/60:.1f} min")
    print(f"{'='*60}")
    print(f"Papers  : {len(papers):,} full-text across {domains} domains")
    print(f"Index   : {idx_mb:.0f} MB")
    print(f"Search  : https://huggingface.co/spaces/amethystani/evirag-search")
