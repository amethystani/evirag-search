"""
Step 4: Upload FAISS index and metadata to Hugging Face Datasets.

FAST UPLOAD — uses hf_transfer (Rust-based multipart uploader, ~10x faster
than the default Python urllib uploader):
  • HF_HUB_ENABLE_HF_TRANSFER=1 env var activates the Rust path
  • Files are uploaded in parallel via ThreadPoolExecutor
  • Large files (>5 GB) are auto-chunked and parallelized by hf_transfer

Install: pip install hf_transfer   (already in requirements.txt)

Uploads to: amethystani/evirag-index
Files:
  evirag.index       — FAISS IndexBinaryIVF (~4 GB for 40M)
  ids.npy            — paper ID strings
  metadata.parquet   — id, title, year, doi, cited_by_count, source, text_excerpt

NOTE: metadata.parquet now includes text_excerpt (title [SEP] abstract, ≤4000 chars).
      This is what agents read — every paper has full abstract context.
      For 40M papers this file is ~8-12 GB compressed (zstd).
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Activate Rust uploader before importing huggingface_hub ──────────────────
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    sys.exit("ERROR: HF_TOKEN environment variable not set. Run:  export HF_TOKEN=hf_...")

REPO_ID = "amethystani/evirag-index"
EMB_DIR = Path("./embeddings")

FILES = [
    EMB_DIR / "evirag.index",
    EMB_DIR / "ids.npy",
    EMB_DIR / "metadata.parquet",
]


def upload_file(api: HfApi, file_path: Path) -> str:
    size_gb = file_path.stat().st_size / 1e9
    print(f"  → Uploading {file_path.name}  ({size_gb:.2f} GB)...")
    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=file_path.name,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Add {file_path.name}",
    )
    url = f"https://huggingface.co/datasets/{REPO_ID}/blob/main/{file_path.name}"
    print(f"  ✓ {file_path.name} → {url}")
    return url


def main():
    api = HfApi(token=HF_TOKEN)

    print(f"Ensuring dataset repo: {REPO_ID}")
    create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=False,
        exist_ok=True,
        token=HF_TOKEN,
    )

    to_upload = [f for f in FILES if f.exists()]
    missing   = [f for f in FILES if not f.exists()]
    for f in missing:
        print(f"  SKIP (not found): {f}")

    if not to_upload:
        sys.exit("Nothing to upload.")

    print(f"\nUploading {len(to_upload)} file(s) in parallel (hf_transfer enabled)...")

    # Upload in parallel — hf_transfer internally uses parallel chunks per file too
    with ThreadPoolExecutor(max_workers=len(to_upload)) as pool:
        futures = {pool.submit(upload_file, api, f): f for f in to_upload}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"  ERROR uploading {futures[fut].name}: {e}")

    print(f"\nAll done! Dataset: https://huggingface.co/datasets/{REPO_ID}")
    print("The Space will pick up the new index on next cold start.")


if __name__ == "__main__":
    main()
