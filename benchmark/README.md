# EVIRAG-Bench

**Benchmark for Epistemic Fidelity in Scientific Retrieval-Augmented Generation**

Companion repository for:  
> *Against Epistemic Collapse: Disagreement-Aware Scientific Retrieval-Augmented Generation*  
> EMNLP 2026 submission

---

## Contents

| Path | Description |
|------|-------------|
| `queries/sample_queries.json` | 25 sample queries (10% of full benchmark, 5 per domain) |
| `queries/ground_truth_sample.json` | Ground-truth viewpoints + CDA-7 labels for sample queries |
| `prompts/system_prompts.md` | All system prompts used for EVIRAG Full and all baselines |
| `annotation_guide/annotation_guide.md` | Full 30-page annotation protocol (condensed) |
| `annotation_guide/cda7_decision_tree.md` | CDA-7 taxonomy decision tree for annotators |
| `evaluation/eval_metrics.py` | CR, VC_emb, CCS, CCE, FS metric implementations |

## Full Benchmark

The complete 250-query benchmark (all 5 domains × 50 queries) with full ground-truth annotations, raw annotator judgments, and evaluation scripts will be released upon publication.

## Quick Start

```bash
git clone https://github.com/amethystani/evirag-search
cd evirag-search/benchmark
# Evaluate your system against sample queries
python evaluation/eval_metrics.py --predictions your_output.json --ground_truth queries/ground_truth_sample.json
```

## Domains

| Domain | Topic | Papers | Controversy class |
|--------|-------|--------|------------------|
| Education | Homework effectiveness | 15 | Stable |
| Biomedicine | Statin therapy in primary prevention | 15 | Stable |
| Economics | Minimum wage and employment | 15 | Polarized |
| Earth Sciences | Climate sensitivity estimates | 15 | Emerging |
| Nutrition | Dietary fat and cardiovascular disease | 15 | Stable |
