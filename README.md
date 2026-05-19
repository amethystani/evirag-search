# EVIRAG: Evidence-Centric, Disagreement-Aware Scientific RAG

> **Standard scientific RAG compresses genuine expert disagreement into a single fluent answer — erasing viewpoint diversity. EVIRAG treats disagreement as the primary output signal, not noise to suppress.**

This repository accompanies the paper:  
**"Against Epistemic Collapse: Disagreement-Aware Scientific Retrieval-Augmented Generation"**  
Anonymous submission — EMNLP 2026  
Benchmark and evaluation materials: [https://anonymous.4open.science/r/evirag-search-76F5](https://anonymous.4open.science/r/evirag-search-76F5)

---

## The Problem: Epistemic Collapse

When asked *"Does homework improve academic achievement?"*, a standard RAG system returns:

> "Research shows that homework improves academic performance, with stronger effects for older students."

EVIRAG returns:

> **View 1 — Dominant [6 sources, confidence: medium]:** Homework improves achievement for grades 7–12 (d≈0.30); near-zero effect for primary grades. *(CDA-7: methodological)*  
> **View 2 — Alternative [3 sources, confidence: medium]:** No significant effect after controlling for SES and teacher quality. *(CDA-7: statistical + methodological)*  
> **View 3 — Minority [2 sources, confidence: low]:** High homework loads harm wellbeing with no academic benefit. *(CDA-7: operational)*  
> Controversy class: **Stable** | CCS: 0.52

The first answer is fluent but epistemically incomplete. The second is honest about the state of the literature.

---

## Seven-Stage Pipeline

```
Query
  │
  ▼
Stage 1: Intent Analysis          — estimates controversy level, scales retrieval
  │
  ▼
Stage 2: Adversarial Retrieval    — four agents with distinct epistemic objectives
  ├── Precision Agent             — high-confidence supporting evidence
  ├── Recall Agent                — broad corpus coverage
  ├── Skeptic Agent               — contradictory and dissenting findings
  └── Counterfactual Agent        — alternative framings and minority positions
  │
  ▼
Stage 3: Claim Graph Construction — atomic claim extraction + NLI-based disagreement graph G
  │
  ▼
Stage 4: CDA-7 Attribution        — causal disagreement type for each contradiction edge
  │
Stage 5: Temporal Drift (TDC)     — publication-year trend in disagreement density
  │
  ▼
Stage 6: Multi-View Synthesis     — Louvain community detection on G → one view per community
  │
  ▼
Stage 7: Evidence-Structured Confidence — calibrated confidence from claim agreement,
                                          source diversity, contradiction severity, CDA-7 priors
```

### CDA-7 Disagreement Taxonomy

Each contradiction edge in the claim graph is labeled with its causal type:

| Class | Description |
|-------|-------------|
| **Methodological** | Different experimental designs or protocols |
| **Population** | Different study populations or settings |
| **Temporal** | Newer evidence supersedes older claims |
| **Operational** | Different operationalizations of key terms |
| **Statistical** | Conflicting effect-size or p-value interpretations |
| **Theoretical** | Competing theoretical frameworks |
| **Replication** | Failure to replicate original findings |

---

## Repository Structure

```
evirag-search/                     ← root
│
├── README.md                      ← this file
├── requirements.txt               ← Python dependencies
├── config.py                      ← centralized configuration (models, paths, prompts)
│
├── Core pipeline
│   ├── epistemic_engine.py        ← intent analysis, claim extraction, NLI
│   ├── multi_agent.py             ← four-agent adversarial retrieval
│   ├── claim_graph.py             ← disagreement graph construction
│   ├── causal_attribution.py      ← CDA-7 classification
│   ├── temporal_tracker.py        ← temporal disagreement curve
│   ├── disagreement.py            ← graph metrics (CCS, ED, PI)
│   ├── epistemic_divergence.py    ← viewpoint separation measures
│   ├── data_layer.py              ← PDF processing, chunking, FAISS indexing
│   ├── vlm_module.py              ← CLIP visual-text alignment
│   ├── groq_client.py             ← Groq API client (cloud backend)
│   └── ollama_cloud_client.py     ← Ollama cloud client
│
├── Entry points
│   ├── run.py                     ← command-line query runner
│   ├── fastapi_service.py         ← REST API service
│   ├── streamlit_app.py           ← Streamlit UI
│   ├── serve_frontend.py          ← serves the React frontend locally
│   └── evirag_system.py           ← high-level system wrapper
│
├── Evaluation
│   ├── evaluation_framework.py    ← CR, VC_emb, VC_kw, CCS, CCE, FS metrics
│   ├── evirag_eval_qwen32b.py     ← evaluation script (Qwen3.6-35B backbone)
│   ├── benchmark_fast_graph.py    ← fast offline graph benchmarking
│   └── autoresearch_eval.py       ← automated research evaluation
│
├── Utility scripts
│   ├── quick_run.py               ← quick sanity-check query
│   ├── quickstart.py              ← guided first-run setup
│   ├── mode_comparison.py         ← compare vanilla RAG vs EVIRAG output
│   ├── run_full_pipeline_fixed.py ← end-to-end pipeline runner
│   └── rebuild_visual_cache.py    ← rebuild CLIP figure cache
│
├── corpus/                        ← place scientific PDFs here (sample papers included)
├── data/                          ← auto-generated (embeddings, vector store, cache)
│
├── benchmark/                     ← EVIRAG-Bench materials
│   ├── README.md
│   ├── queries/sample_queries.json
│   ├── prompts/system_prompts.md
│   └── annotation_guide/cda7_decision_tree.md
│
├── evirag/                        ← React frontend (Vercel deployment)
├── evirag_legacy/                 ← previous frontend version
│
├── hf_backend/                    ← Hugging Face Spaces backend
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── evirag-search/                 ← large-scale search backend (100k+ papers)
│   ├── search_backend.py
│   ├── fast_pipeline.py
│   └── pipeline/                  ← data pipeline scripts
│
├── paper/latex/                   ← LaTeX source for the research paper
│   ├── acl_latex.tex
│   ├── custom.bib
│   └── acl_latex.pdf
│
└── tests/                         ← test scripts
```

---

## Setup

### Requirements

- Python 3.9+
- [Ollama](https://ollama.com) (for local inference) **or** a Groq API key (for cloud inference)

### Install

```bash
git clone https://github.com/amethystani/evirag-search
cd evirag-search
pip install -r requirements.txt
```

### Add your corpus

Place scientific PDF files in the `corpus/` directory. The repository includes seven sample homework-effectiveness papers to get started.

```bash
ls corpus/    # seven sample papers included
```

### Configure the model backend

Copy `.env.example` to `.env` and set your preferred backend:

**Option A — Local Ollama (used in paper experiments):**
```bash
# Pull the model used in experiments
ollama pull qwen3.6:35b-a3b

# .env
OLLAMA_HOST=http://localhost:11434
OLLAMA_CLOUD_MODEL=qwen3.6:35b-a3b
```

**Option B — Groq API (faster, no local GPU required):**
```bash
# .env
GROQ_API_KEY=your_key_here
```

---

## Running EVIRAG

### Command-line

```bash
python run.py "Does homework improve academic achievement?"
```

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501), click **Initialize System**, then enter a query.

### REST API

```bash
uvicorn fastapi_service:app --reload
# POST http://localhost:8000/query
# {"query": "Do statins prevent cardiovascular events?", "mode": "evirag"}
```

### Python API

```python
from evirag_system import EVIRAGSystem, EVIRAGConfig

config = EVIRAGConfig(mode="evirag")
system = EVIRAGSystem(config)
system.initialize_corpus()

result = system.query("Does raising the minimum wage reduce employment?")

for view in result["answer"]["views"]:
    print(f"[{view['label']}] {view['summary']}")
    print(f"  Sources: {view['source_count']} | CDA-7: {view['cda7_attribution']}")
    print(f"  Confidence: {view['confidence']}")
```

---

## Evaluation

EVIRAG-Bench contains 250 queries across five contested scientific domains. A 25-query sample is included in `benchmark/queries/sample_queries.json`.

```bash
python evaluation_framework.py \
  --predictions your_output.json \
  --ground_truth benchmark/queries/sample_queries.json
```

### Metrics

| Metric | Description |
|--------|-------------|
| **CR** | ContradictionRecall — fraction of annotated contradictions recovered |
| **VC_emb** | ViewpointCoverage (dense encoder, all-MiniLM-L6-v2, θ=0.65) |
| **VC_kw** | ViewpointCoverage (keyword overlap) |
| **CCS** | Consensus Collapse Score — asymmetry in viewpoint coverage |
| **CCE** | Confidence Calibration Error |
| **FS** | FaithfulnessScore (FActScore-style NLI verification) |

### Main results (EVIRAG-Bench, 250 queries)

| System | CR↑ | VC_emb↑ | CCS↓ | FS↑ |
|--------|-----|---------|------|-----|
| Vanilla RAG | 0.572 | 0.423 | 0.571 | 0.824 |
| MADAM-RAG | 0.618 | 0.508 | 0.549 | 0.856 |
| MMR-RAG | 0.595 | 0.534 | 0.547 | 0.839 |
| PaperQA2 | 0.553 | 0.467 | 0.583 | **0.903** |
| **EVIRAG Full** | **0.742** | **0.847** | **0.518** | 0.891 |

Human evaluation (50 queries, 3 annotators, 5-point epistemic completeness scale):  
**EVIRAG Full: 4.3/5 vs Vanilla RAG: 2.1/5** (Krippendorff's α=0.72, p<0.001, Cohen's d=2.71)

---

## Baselines

All baseline implementations are included and use the same corpus and backbone model:

| System | Description |
|--------|-------------|
| Vanilla RAG | Dense retrieval + single-view synthesis |
| Single-agent | Precision agent only, single-view synthesis |
| EVIRAG-NoGraph | Adversarial retrieval without claim graph |
| EVIRAG-NoMultiView | Claim graph without multi-view output |
| MADAM-RAG | Multi-agent debate → resolved single answer |
| MMR-RAG | Maximal Marginal Relevance retrieval, single-view |
| PaperQA2 | Citation-grounded scientific QA |

Prompt templates for all systems are in `benchmark/prompts/system_prompts.md`.

---

## Offline Mode (Fast Path)

The claim graph can be precomputed offline over the full corpus (one-time cost). At query time, inference slices a retrieved subgraph rather than rerunning pairwise NLI.

```bash
# Precompute graph (run once)
python benchmark_fast_graph.py --precompute

# Query using cached graph (fast path)
python run.py "Does saturated fat increase cardiovascular risk?" --fast
```

Benchmark result: **0.43 s** warm-query latency vs **10.62 s** for vanilla RAG (25.6× speedup).

---

## Hardware Used in Paper Experiments

- GPU: NVIDIA RTX 4500 Ada (24 GB VRAM)
- Model: `qwen3.6:35b-a3b` via Ollama (`think=False`)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (d=384)
- Corpus: 75 papers (3,247 chunks, 4,831 atomic claims)

Cross-backbone validation on the education subset confirms consistent gains with Qwen2.5-7B and Llama-3.1-70B.

---

## System Constraints

| Setting | Value |
|---------|-------|
| Retrieval budget | 15 chunks (3+5+4+3 across agents) |
| Max claims per query | 60 (15 per agent) |
| Chunk size | 256 tokens, 32-token overlap |
| Community detection | Louvain on signed claim adjacency matrix |
| NLI threshold | 0.45 minimum edge confidence |

---

## License

MIT — see [LICENSE](LICENSE).
