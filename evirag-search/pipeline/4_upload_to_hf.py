"""
Step 4: Upload FAISS index and metadata to Hugging Face Datasets.

Uploads to: amethystani/evirag-index  (public dataset repo)

Files uploaded:
  evirag.index       — FAISS IndexBinaryIVF (~4GB)
  ids.npy            — paper ID strings (maps FAISS int → OpenAlex ID)
  metadata.parquet   — title, year, doi, cited_by_count, source

Run after step 3. Takes ~30-60 min depending on upload speed.
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_TOKEN   = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Set it before running.")
REPO_ID    = "amethystani/evirag-index"
EMB_DIR    = Path("./embeddings")

FILES = [
    EMB_DIR / "evirag.index",
    EMB_DIR / "ids.npy",
    EMB_DIR / "metadata.parquet",
]


def main():
    api = HfApi(token=HF_TOKEN)

    # Create dataset repo (idempotent)
    print(f"Ensuring dataset repo exists: {REPO_ID}")
    create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=False,
        exist_ok=True,
        token=HF_TOKEN,
    )

    for file_path in FILES:
        if not file_path.exists():
            print(f"  SKIP (not found): {file_path}")
            continue
        size_gb = file_path.stat().st_size / 1e9
        print(f"\nUploading {file_path.name}  ({size_gb:.2f} GB)...")
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=file_path.name,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Add {file_path.name}",
        )
        print(f"  Done: https://huggingface.co/datasets/{REPO_ID}/blob/main/{file_path.name}")

    print(f"\nAll files uploaded to https://huggingface.co/datasets/{REPO_ID}")
    print("Next: deploy the Space from evirag-search/space/")


if __name__ == "__main__":
    main()
