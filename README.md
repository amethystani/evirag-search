# EVIRAG: Epistemic-Fidelity-First Retrieval-Augmented Generation

> **Standard RAG commits epistemic collapse — compressing genuine scientific controversy into false consensus. EVIRAG is the first RAG framework that treats disagreement as the primary output signal, not noise to suppress.**

## The Problem: Epistemic Collapse in Scientific RAG

When asked *"Does homework improve academic achievement?"*, a standard RAG system says:
> "Research shows homework improves academic performance."

EVIRAG says:
> **Dominant view** (5 sources, high confidence): *No conclusive evidence of improvement since 1987.*
> **Alternative view** (3 sources): *Effects depend on design, grade level, and subject.*
> **Minority view** (2 sources): *Excess homework correlates with reduced wellbeing.*
> Disagreement density: 5.3% | Confidence: MEDIUM | Controversy class: stable

The first answer is fluent but epistemically false. The second is honest about the state of knowledge.

This gap — between what existing RAG produces and what scientific epistemology requires — is what EVIRAG addresses.

### Why existing approaches don't solve this

| System | Their framing | What they still do |
|---|---|---|
| MADAM-RAG (2025) | Ambiguity + misinformation | Debate to find best answer — consensus-seeking |
| ContraCrow / PaperQA2 | Contradiction detection | Binary label per claim — no multi-view synthesis |
| SciFact | Claim verification | SUPPORTS/REFUTES — adjudicates rather than surfaces |
| TruthfulRAG | Factual conflict | Suppresses conflicting sources |
| **EVIRAG** | **Scientific controversy** | **Structures and surfaces the disagreement** |

The distinction: existing systems treat conflict as *ambiguity* (multiple correct answers) or *noise* (misinformation). EVIRAG treats it as *controversy* — genuine expert disagreement that is epistemically premature to resolve.

## ✨ Key Features

### 1. Evidence Over Fluency
- Claims extracted as atomic, verifiable units
- Relationships explicitly modeled (support/contradict)
- Visual evidence aligned with textual claims using CLIP

### 2. Disagreement Over Consensus
- Multi-view answers instead of single confident response
- Disagreement graph construction
- Confidence calibrated from evidence structure

### 3. Verification Before Synthesis
- Explanation-first verification (X-IR)
- Hypothesis generation before retrieval
- Claims verified against retrieved evidence

### 4. Multi-Agent Deliberation
- **Precision Agent**: High-confidence support
- **Recall Agent**: Broad coverage
- **Skeptic Agent**: Find contradictions
- **Counterfactual Agent**: Alternative explanations

### 5. Visual Evidence Integration
- Figure extraction from PDFs
- CLIP-based visual-text alignment
- Visual evidence strength in confidence calibration

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI Layer                       │
│  (Multi-view answer | Graph | Claims | Visual | Metrics)    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    EVIRAG Core System                        │
├─────────────────────────────────────────────────────────────┤
│  1. Epistemic Intent Analysis (SLM: phi-3:mini)             │
│  2. Hypothesis Generation (LLM: llama3:latest)              │
│  3. Multi-Agent Retrieval                                    │
│     ├── Precision Agent (llama3)                            │
│     ├── Recall Agent (qwen2.5:4b)                           │
│     ├── Skeptic Agent (llama3)                              │
│     └── Counterfactual Agent (deepseek-r1:7b)              │
│  4. Claim Extraction (SLM filter → LLM extract)             │
│  5. Disagreement Graph Construction                         │
│  6. Visual Evidence Analysis (CLIP ViT-B/16)                │
│  7. Multi-View Synthesis                                     │
│  8. Confidence Calibration                                   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
├─────────────────────────────────────────────────────────────┤
│  • PDF Processing (PyMuPDF)                                 │
│  • Section-aware Chunking                                    │
│  • Embeddings (sentence-transformers/MiniLM)                │
│  • Vector Store (FAISS)                                      │
│  • Figure Extraction & CLIP Embeddings                       │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
evirag/
├── config.py                    # Centralized configuration
├── evirag_system.py            # Main EVIRAG orchestrator
├── requirements.txt             # Dependencies
├── README.md                    # This file
│
├── data/
│   └── data_layer.py           # PDF processing, chunking, indexing
│
├── models/
│   ├── epistemic_engine.py     # Intent analysis, hypothesis, claims, NLI
│   ├── vlm_module.py           # CLIP embedding & visual alignment
│   └── disagreement.py         # Graph, metrics, synthesis, calibration
│
├── agents/
│   └── multi_agent.py          # Multi-agent deliberative retrieval
│
├── ui/
│   └── streamlit_app.py        # Streamlit interface
│
└── corpus/                      # Place your PDF files here
```

## 🚀 Getting Started

### Prerequisites

1. **Ollama** with models:
   ```bash
   ollama pull phi3:mini
   ollama pull llama3:latest
   ollama pull qwen2.5:4b
   ollama pull deepseek-r1:7b
   ```

2. **Python 3.9+**

### Installation

1. Clone and navigate to project:
   ```bash
   cd evirag
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt --break-system-packages
   ```

3. Add your corpus:
   ```bash
   # Place PDF files in corpus/ directory
   mkdir -p corpus
   # Copy your scientific PDFs to corpus/
   ```

### Running EVIRAG

#### Option 1: Streamlit UI (Recommended)

```bash
streamlit run ui/streamlit_app.py
```

Then:
1. Click "Initialize System" in sidebar
2. Enter your scientific question
3. Click "Query" to see multi-view results

#### Option 2: Python API

```python
from evirag_system import EVIRAGSystem, EVIRAGConfig

# Initialize
config = EVIRAGConfig(
    mode="evirag",
    use_visual_grounding=True,
    depth_vs_speed="balanced"
)

system = EVIRAGSystem(config)
system.initialize_corpus()

# Query
result = system.query(
    "Is overparameterization necessary for generalization in deep neural networks?"
)

# Access results
print(result['answer']['dominant_view']['summary'])
print(f"Confidence: {result['answer']['overall_confidence']}")
```

## 🎮 Execution Modes

### 1. Vanilla RAG (Baseline)
Standard retrieval-augmented generation:
- Simple semantic search
- Direct synthesis  
- No disagreement modeling
- Fast baseline for comparison

### 2. EVIRAG (Full Pipeline)
Complete evidence-centric pipeline with all features:
- Epistemic intent analysis
- Hypothesis generation  
- Multi-agent deliberative retrieval
- Claim extraction and disagreement graph
- Visual evidence grounding
- Full confidence calibration

## 📊 Key Components

### Epistemic Intent Analyzer
Classifies queries along dimensions:
- Factual vs Disputed
- Empirical vs Theoretical
- Expected Disagreement Level

### Explanation-First Verification (X-IR)
Generates hypothesis before retrieval:
```json
{
  "central_hypothesis": "...",
  "supporting_claims": [...],
  "assumptions": [...],
  "expected_counterclaims": [...]
}
```

### Multi-Agent Deliberation
Each agent has specialized objectives:
- **Precision**: Strict semantic, high-confidence support
- **Recall**: Expansive semantic, broad coverage
- **Skeptic**: Adversarial semantic, find contradictions
- **Counterfactual**: Alternative explanations

### Disagreement Metrics
- **Disagreement Density**: Ratio of contradictory edges
- **Conflict Ratio**: Contradictions vs supports
- **Claim Entropy**: Diversity of claim stances
- **Conflict Centrality**: Claims central to conflicts
- **Visual-Text Mismatch**: Alignment score

### Confidence Calibration
Weighted factors:
- Claim agreement (30%)
- Source diversity (20%)
- Contradiction severity (25%)
- Unverified assumptions (15%)
- Visual alignment (10%)

## 🔬 Example Query Flow

**Query**: "Is overparameterization necessary for generalization in deep neural networks?"

1. **Intent Analysis**:
   - Factual vs Disputed: `highly_disputed`
   - Disagreement Level: `high`

2. **Hypothesis Generation**:
   - Central: "Overparameterization can improve generalization"
   - Supporting: ["Double descent phenomenon", "Implicit regularization"]
   - Expected counters: ["Overfitting risk", "Classical bias-variance"]

3. **Multi-Agent Retrieval**:
   - Precision: Finds 5 high-confidence papers on double descent
   - Recall: Retrieves 15 diverse papers
   - Skeptic: Finds 8 papers on overfitting concerns
   - Counterfactual: Retrieves 6 papers on alternative regularization

4. **Claim Extraction**: 47 atomic claims extracted

5. **Graph Construction**: 23 support, 15 contradict, 9 neutral edges

6. **Visual Analysis**: 12 figures aligned, 0.3 mismatch score

7. **Synthesis**:
   - **Dominant View**: "Evidence supports conditional benefits..."
   - **Alternative**: "Classical wisdom on bias-variance..."
   - **Minority**: "Data quality matters more..."

8. **Confidence**: `MEDIUM (0.62)` - Mixed evidence, active debate

## 🎯 Use Cases

- **Literature Review**: Surface competing viewpoints
- **Research Planning**: Identify gaps and controversies
- **Claim Verification**: Check evidence strength
- **Thesis Development**: Understand debate landscape
- **Meta-Analysis**: Aggregate conflicting findings

## ⚙️ Configuration

Edit `config.py` to customize:

- Model selection and parameters
- Embedding dimensions
- Agent retrieval strategies
- Confidence calibration weights
- Graph construction rules

## 📈 Evaluation

Metrics (planned):
- Hallucination rate
- Contradiction detection accuracy
- Viewpoint coverage
- Confidence calibration error
- Visual grounding impact

Baselines:
- Vanilla RAG
- Single-agent RAG

## 🔧 System Requirements

- **RAM**: 16 GB recommended (M1 Mac or equivalent)
- **Storage**: ~5 GB for models + corpus
- **GPU**: Optional (uses CPU/MPS)
- **Models**: All local via Ollama

## 🚨 Limitations

- Corpus size: Optimal for 1-2 GB PDFs (~500-1000 papers)
- Processing time: 30s-2min per query depending on mode
- CLIP: Alignment only, no language generation
- Graph: Max 500 nodes to prevent explosion

## 🛣️ Roadmap

- [ ] n8n orchestration workflows
- [ ] Batch evaluation framework
- [ ] Interactive graph visualization (PyVis)
- [ ] Export to citation managers
- [ ] Multi-domain extension
- [ ] Temporal disagreement tracking

## 📚 Citation

```bibtex
@software{evirag2025,
  title={EVIRAG: Evidence-Centric, Disagreement-Aware RAG},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/evirag}
}
```

## 📄 License

MIT License - See LICENSE file

## 🤝 Contributing

Contributions welcome! Areas:
- Agent strategies
- Disagreement metrics
- Visual grounding improvements
- Evaluation benchmarks
- UI/UX enhancements

## 📞 Contact

For questions or collaboration:
- GitHub Issues
- Email: your.email@domain.com

---

**EVIRAG**: *Because scientific truth lives in the disagreement, not the consensus.*
