"""
EVIRAG FULL DATASET PIPELINE — All 8M papers from peS2o v3
============================================================
- Streams all 136 shards from allenai/peS2o v3
- Checkpoints every shard so it can resume safely on restart
- Embeds on CUDA (RTX 4500 Ada, batch=256, bfloat16, Flash Attn 2)
- Builds FAISS IndexBinaryIVF
- Uploads to HF at the VERY END only, with correct API
- No torch.compile (crashes on this system's triton)
"""

import io, os, json, time, warnings, random, gc, shutil
warnings.filterwarnings("ignore")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import faiss
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import HfApi, create_repo, hf_hub_download

try:
    import zstandard as zstd
except ImportError:
    raise SystemExit("Install zstandard: pip install zstandard")

# ── Environment ───────────────────────────────────────────────────────────────
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

HF_TOKEN   = os.environ.get("HF_TOKEN", "")  # set HF_TOKEN env var
REPO_ID    = "amethystani/evirag-index"
OUT_DIR    = Path("./full_pipeline")
RAW_DIR    = OUT_DIR / "shards"
EMB_DIR    = OUT_DIR / "embeddings"
CKPT_FILE  = OUT_DIR / "checkpoint.json"

BATCH_SIZE = 256      # safe for 24GB VRAM with bfloat16
MAX_TOKENS = 512
TEXT_CHARS = 4_000
N_SHARDS   = 136

OUT_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)
EMB_DIR.mkdir(exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"Device: CUDA — {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Device: Apple MPS")
else:
    DEVICE = torch.device("cpu")
    print("Device: CPU")

# ── Checkpoint helpers ────────────────────────────────────────────────────────
def load_checkpoint():
    if CKPT_FILE.exists():
        with open(CKPT_FILE) as f:
            return json.load(f)
    return {"shards_done": [], "total_papers": 0, "embed_shards_done": []}

def save_checkpoint(ckpt):
    with open(CKPT_FILE, "w") as f:
        json.dump(ckpt, f, indent=2)

def reconstruct_abstract(inv):
    if not inv:
        return ""
    pairs = [(pos, w) for w, positions in inv.items() for pos in positions]
    return " ".join(w for _, w in sorted(pairs))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Stream ALL shards from peS2o v3
# ══════════════════════════════════════════════════════════════════════════════
def step1_download(ckpt):
    print(f"\n{'='*60}")
    print(f"STEP 1: Stream all {N_SHARDS} peS2o v3 shards")
    print(f"{'='*60}")

    shards_done = set(ckpt["shards_done"])
    all_shards  = [f"data/v3/train-{i:04d}-of-{N_SHARDS:04d}.zst" for i in range(N_SHARDS)]

    for shard_idx, shard_path in enumerate(all_shards):
        shard_name = shard_path.split("/")[-1]

        if shard_name in shards_done:
            print(f"  [{shard_idx+1}/{N_SHARDS}] {shard_name} — already done, skipping")
            continue

        out_file = RAW_DIR / (shard_name.replace(".zst", ".parquet"))
        print(f"  [{shard_idx+1}/{N_SHARDS}] {shard_name} → {out_file.name}")

        # Download from HF hub with retry
        local_path = None
        for attempt in range(5):
            try:
                local_path = hf_hub_download(
                    repo_id="allenai/peS2o",
                    filename=shard_path,
                    repo_type="dataset",
                    token=HF_TOKEN,
                )
                break
            except Exception as e:
                print(f"    ⚠ Download error (attempt {attempt+1}/5): {e}")
                time.sleep(5 * (attempt + 1))

        if not local_path:
            print(f"    ✗ Skipping {shard_name} after 5 failures")
            shards_done.add(shard_name)
            ckpt["shards_done"] = list(shards_done)
            save_checkpoint(ckpt)
            continue

        # Parse shard
        papers = []
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

                        # Safely handle metadata
                        meta = rec.get("metadata") or {}
                        if not isinstance(meta, dict):
                            meta = {}

                        full_text = rec.get("text") or ""
                        if len(full_text) < 200:
                            continue

                        title = (meta.get("title") or "").strip()
                        if not title:
                            title = full_text.split("\n")[0][:200].strip()

                        ext_ids = meta.get("external_ids") or {}
                        if not isinstance(ext_ids, dict):
                            ext_ids = {}

                        fields = meta.get("s2fieldsofstudy") or []
                        if not isinstance(fields, list):
                            fields = []
                        domain = fields[0] if fields else "Unknown"

                        papers.append({
                            "id":              f"pes2o:{rec.get('id', '')}",
                            "doi":             ext_ids.get("DOI") or "",
                            "title":           title,
                            "full_text":       full_text[:TEXT_CHARS],
                            "year":            int(meta.get("year") or 0),
                            "cited_by_count":  0,
                            "source":          "peS2o/Semantic Scholar",
                            "domain":          domain,
                        })

        except Exception as e:
            print(f"    ⚠ Parse error: {e} — saving what we have")

        if papers:
            table = pa.table({
                "id":             pa.array([p["id"]            for p in papers]),
                "doi":            pa.array([p["doi"]           for p in papers]),
                "title":          pa.array([p["title"]         for p in papers]),
                "full_text":      pa.array([p["full_text"]     for p in papers]),
                "year":           pa.array([p["year"]          for p in papers], type=pa.int16()),
                "cited_by_count": pa.array([p["cited_by_count"]for p in papers], type=pa.int32()),
                "source":         pa.array([p["source"]        for p in papers]),
                "domain":         pa.array([p["domain"]        for p in papers]),
            })
            pq.write_table(table, out_file, compression="zstd")
            ckpt["total_papers"] += len(papers)
            print(f"    ✓ {len(papers):,} papers saved (total: {ckpt['total_papers']:,})")
        else:
            print(f"    ⚠ No usable papers from {shard_name}")

        shards_done.add(shard_name)
        ckpt["shards_done"] = list(shards_done)
        save_checkpoint(ckpt)

    print(f"\n✓ Download complete. Total papers: {ckpt['total_papers']:,}")
    return ckpt


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Embed each shard's parquet with GPU — save .npy per shard
# ══════════════════════════════════════════════════════════════════════════════
def step2_embed(ckpt):
    print(f"\n{'='*60}")
    print(f"STEP 2: Embed all shards on {DEVICE}")
    print(f"{'='*60}")

    print("Loading SPECTER...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
    try:
        model = AutoModel.from_pretrained(
            "allenai/specter",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(DEVICE).eval()
        print("✓ Flash Attention 2")
    except Exception:
        model = AutoModel.from_pretrained(
            "allenai/specter",
            torch_dtype=torch.bfloat16,
        ).to(DEVICE).eval()
        print("✓ Standard attention")

    embed_shards_done = set(ckpt.get("embed_shards_done", []))
    parquet_files = sorted(RAW_DIR.glob("*.parquet"))
    total = len(parquet_files)

    for idx, pfile in enumerate(parquet_files):
        stem = pfile.stem
        if stem in embed_shards_done:
            print(f"  [{idx+1}/{total}] {stem} — already embedded, skipping")
            continue

        out_binary = EMB_DIR / f"{stem}.bin.npy"
        out_ids    = EMB_DIR / f"{stem}.ids.npy"
        out_meta   = EMB_DIR / f"{stem}.meta.parquet"

        print(f"  [{idx+1}/{total}] Embedding {stem}...")
        df = pq.read_table(pfile).to_pydict()
        texts = [f"{t} [SEP] {b[:3000].replace(chr(10), ' ')}"
                 for t, b in zip(df["title"], df["full_text"])]
        N = len(texts)
        all_binary = []
        t0 = time.time()

        with torch.no_grad():
            for start in range(0, N, BATCH_SIZE):
                end   = min(start + BATCH_SIZE, N)
                batch = texts[start:end]

                for attempt in range(3):
                    try:
                        enc  = tokenizer(batch, max_length=MAX_TOKENS, padding=True,
                                         truncation=True, return_tensors="pt")
                        ids_t  = enc["input_ids"].to(DEVICE)
                        mask   = enc["attention_mask"].to(DEVICE)

                        if DEVICE.type == "cuda":
                            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                                out = model(input_ids=ids_t, attention_mask=mask)
                        else:
                            out = model(input_ids=ids_t, attention_mask=mask)

                        emb = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) \
                              / mask.sum(1, keepdim=True).clamp(min=1)
                        emb = F.normalize(emb, p=2, dim=1)

                        bits   = (emb > 0).to(torch.uint8).cpu().numpy()
                        binary = np.packbits(bits, axis=1)
                        all_binary.append(binary)
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        gc.collect()
                        if attempt == 2:
                            # Fall back: embed one by one
                            for single in batch:
                                enc2  = tokenizer([single], max_length=MAX_TOKENS,
                                                  padding=True, truncation=True, return_tensors="pt")
                                ids2  = enc2["input_ids"].to(DEVICE)
                                mask2 = enc2["attention_mask"].to(DEVICE)
                                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                                    out2 = model(input_ids=ids2, attention_mask=mask2)
                                emb2 = (out2.last_hidden_state * mask2.unsqueeze(-1)).sum(1) \
                                       / mask2.sum(1, keepdim=True).clamp(min=1)
                                emb2 = F.normalize(emb2, p=2, dim=1)
                                b2   = np.packbits((emb2 > 0).to(torch.uint8).cpu().numpy(), axis=1)
                                all_binary.append(b2)

                elapsed = time.time() - t0
                speed   = end / elapsed if elapsed > 0 else 0
                print(f"    {end:>6,}/{N:,}  {speed:.0f} papers/sec", end="\r")

        print()
        binary_arr = np.vstack(all_binary)
        ids_arr    = np.array(df["id"], dtype=object)
        np.save(out_binary, binary_arr, allow_pickle=False)
        np.save(out_ids,    ids_arr,    allow_pickle=True)

        meta_table = pa.table({
            "id":              pa.array(df["id"]),
            "title":           pa.array(df["title"]),
            "year":            pa.array(df["year"],           type=pa.int16()),
            "doi":             pa.array(df["doi"]),
            "cited_by_count":  pa.array(df["cited_by_count"], type=pa.int32()),
            "source":          pa.array(df["source"]),
            "domain":          pa.array(df["domain"]),
            "text_excerpt":    pa.array([t[:2000] for t in df["full_text"]]),
        })
        pq.write_table(meta_table, out_meta, compression="zstd")

        embed_shards_done.add(stem)
        ckpt["embed_shards_done"] = list(embed_shards_done)
        save_checkpoint(ckpt)
        print(f"    ✓ {N:,} vectors saved")

        # Free GPU memory between shards
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n✓ All shards embedded.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Merge all shard embeddings and build FAISS index
# ══════════════════════════════════════════════════════════════════════════════
def step3_index():
    print(f"\n{'='*60}")
    print("STEP 3: Merge embeddings and build FAISS index")
    print(f"{'='*60}")

    bin_files = sorted(EMB_DIR.glob("*.bin.npy"))
    id_files  = sorted(EMB_DIR.glob("*.ids.npy"))
    meta_files = sorted(EMB_DIR.glob("*.meta.parquet"))

    print(f"  Merging {len(bin_files)} embedding shards...")
    t0 = time.time()
    all_binary = []
    all_ids    = []
    for bf, idf in zip(bin_files, id_files):
        all_binary.append(np.load(bf, allow_pickle=False))
        all_ids.append(np.load(idf, allow_pickle=True))
        print(f"    Loaded {bf.name}", end="\r")

    binary = np.vstack(all_binary)
    ids    = np.concatenate(all_ids)
    print(f"\n  ✓ Total vectors: {len(binary):,} in {(time.time()-t0)/60:.1f} min")

    # Merge metadata parquets — STREAMING VERSION to avoid OOM
    print("  Merging metadata parquets (streaming)...")
    meta_out = OUT_DIR / "metadata.parquet"
    writer = None
    t_m = time.time()
    for i, f in enumerate(meta_files):
        table = pq.read_table(f)
        if writer is None:
            writer = pq.ParquetWriter(meta_out, table.schema, compression="zstd")
        writer.write_table(table)
        print(f"    Merged {i+1}/{len(meta_files)}: {f.name}", end="\r")
        del table 
        gc.collect()
    
    if writer:
        writer.close()
    
    size_mb = meta_out.stat().st_size / 1e6
    print(f"\n  ✓ metadata.parquet → {size_mb:.1f} MB (Done in {(time.time()-t_m)/60:.1f} min)")

    # Save merged ids and binary
    ids_out    = OUT_DIR / "ids.npy"
    binary_out = OUT_DIR / "binary.npy"
    np.save(ids_out,    ids,    allow_pickle=True)
    np.save(binary_out, binary, allow_pickle=False)
    print(f"  ✓ ids.npy and binary.npy saved")

    # Build FAISS IndexBinaryIVF
    N, D_bytes = binary.shape
    DIM = D_bytes * 8
    faiss.omp_set_num_threads(os.cpu_count() or 32)

    nlist     = min(65536, max(1024, N // 50))
    quantizer = faiss.IndexBinaryFlat(DIM)
    index     = faiss.IndexBinaryIVF(quantizer, DIM, nlist)
    index.nprobe = 128

    n_train = min(N, 500_000)
    print(f"  Training FAISS on {n_train:,} vectors (nlist={nlist})...")
    train_idx = np.random.choice(N, size=n_train, replace=False)
    index.train(np.ascontiguousarray(binary[train_idx]))

    print(f"  Adding {N:,} vectors to index...")
    for start in range(0, N, 1_000_000):
        end   = min(start + 1_000_000, N)
        index.add(np.ascontiguousarray(binary[start:end]))
        print(f"    {end:,}/{N:,}", end="\r")

    idx_out = OUT_DIR / "evirag.index"
    faiss.write_index_binary(index, str(idx_out))
    idx_mb = idx_out.stat().st_size / 1e6
    print(f"\n  ✓ FAISS index built: {index.ntotal:,} vectors ({idx_mb:.0f} MB)")

    return index, binary, ids


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Upload everything to HF — LAST STEP ONLY
# ══════════════════════════════════════════════════════════════════════════════
def step4_upload():
    print(f"\n{'='*60}")
    print("STEP 4: Upload to Hugging Face Datasets")
    print(f"{'='*60}")

    api = HfApi(token=HF_TOKEN)
    create_repo(REPO_ID, repo_type="dataset", private=False, exist_ok=True, token=HF_TOKEN)

    files_to_upload = [
        OUT_DIR / "evirag.index",
        OUT_DIR / "ids.npy",
        OUT_DIR / "binary.npy",
        OUT_DIR / "metadata.parquet",
    ]

    for fpath in files_to_upload:
        if not fpath.exists():
            print(f"  ⚠ {fpath.name} not found — skipping")
            continue

        local_size = fpath.stat().st_size
        mb = local_size / 1e6
        print(f"  {fpath.name} ({mb:.1f} MB)...", end=" ", flush=True)

        for attempt in range(5):
            try:
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=fpath.name,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    commit_message=f"Full peS2o pipeline: {fpath.name}",
                )
                print("✓")
                break
            except Exception as e:
                print(f"\n    ⚠ Upload failed (attempt {attempt+1}/5): {e}")
                time.sleep(10 * (attempt + 1))
        else:
            print(f"  ✗ Failed to upload {fpath.name} after 5 attempts")

    print(f"\n✓ All files uploaded.")
    print(f"Live: https://huggingface.co/datasets/{REPO_ID}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t_total = time.time()

    print(f"\n{'='*60}")
    print("EVIRAG FULL PIPELINE — ALL peS2o v3 shards (~8M papers)")
    print(f"{'='*60}")

    ckpt = load_checkpoint()
    print(f"Checkpoint loaded. Shards done: {len(ckpt['shards_done'])}/{N_SHARDS}, "
          f"Embed done: {len(ckpt.get('embed_shards_done', []))}/{N_SHARDS}")

    ckpt = step1_download(ckpt)
    step2_embed(ckpt)
    step3_index()
    step4_upload()

    elapsed = (time.time() - t_total) / 3600
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE in {elapsed:.1f} hours")
    print(f"{'='*60}")
