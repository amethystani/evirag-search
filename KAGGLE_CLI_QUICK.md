# Run EVIRAG Pipeline on Kaggle Using CLI

**One command to rule them all:**

```bash
pip install kaggle
kaggle kernels push -p evirag-search/ -u
```

That's it. Kaggle will:
1. Create a new notebook in your account
2. Upload all code + dependencies
3. Run on **T4x2 GPU** (free tier)
4. Stream output to your terminal
5. Upload results to HF when done

---

## Step-by-Step

### 1. Install Kaggle CLI
```bash
pip install kaggle
```

### 2. Set up Kaggle API credentials
- Go to: https://www.kaggle.com/settings/account
- Click "Create New API Token"
- Save as `~/.kaggle/kaggle.json` (or `%USERPROFILE%/.kaggle/kaggle.json` on Windows)

### 3. Set HF token (for uploading results)
```bash
export HF_TOKEN=your_hf_token_here
```

### 4. Run the pipeline
```bash
cd evirag-search
kaggle kernels push -p . -u
```

The `-u` flag runs it immediately after push.

---

## What It Does

1. **Creates a notebook** in your Kaggle account (`amethystani/evirag-search-pipeline`)
2. **Uploads** all Python scripts + requirements
3. **Runs** on **T4x2 GPU** (automatically selected in `kaggle_kernel.json`)
4. **Streams** output to your terminal in real-time
5. **Saves** results to `/kaggle/working/` (persistent)
6. **Uploads** final index to HF Datasets (if `HF_TOKEN` set)

---

## Timeline

**Demo (100 papers):**
- Download: 30s
- Embed (GPU): 2 min
- Index: 1 min
- Total: ~5 min

**Full (40M papers):**
- Download: 5-8 hrs
- Embed (GPU): 4-6 hrs
- Index: 15-20 min
- Upload: 30-60 min
- **Total: ~10-15 hrs** (fits in Kaggle's 12-hour limit)

---

## Monitor Progress

After pushing, Kaggle streams output. Or manually check:

```bash
kaggle kernels status amethystani/evirag-search-pipeline
```

---

## Download Results

After it finishes:

```bash
kaggle kernels output -p amethystani/evirag-search-pipeline
```

This fetches:
- `evirag.index` (~4GB)
- `ids.npy` (~320MB)
- `metadata.parquet` (~2GB)

---

## Edit & Re-run

Change something? Push again:

```bash
kaggle kernels push -p . -u
```

It auto-increments the version and re-runs.

---

## Full Pipeline from Scratch

Clone + run in one command:

```bash
git clone https://github.com/amethystani/evirag-search.git
cd evirag-search
export HF_TOKEN=your_token_here
kaggle kernels push -p . -u
```

Done!

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `kaggle: command not found` | `pip install kaggle` + add `~/.local/bin` to PATH |
| API credentials error | Check `~/.kaggle/kaggle.json` exists + has right permissions |
| Kernel not found | First push creates it; subsequent pushes update it |
| GPU not allocated | Check your Kaggle tier (free has limited GPU hours) |

---

## Notes

- Kaggle notebooks are **public by default** (set `"isPrivate": true` in `kaggle_kernel.json` if sensitive)
- Output persists in `/kaggle/working/` for 30 days
- Kaggle has **12-hour runtime limit** (full pipeline fits with ~1 hr buffer)
- Free GPU is **T4** (same as you would use for the quick demo)
- **Pro users** get longer runtimes + more GPU hours (optional)

---

**Next:** Watch the output, grab the index from HF Datasets when done!
