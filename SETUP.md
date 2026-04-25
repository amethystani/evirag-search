# EVIRAG Setup Guide

## Complete Installation and Setup Instructions

### Prerequisites

#### 1. Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama service:
```bash
ollama serve
```

#### 2. Pull Required Models

```bash
# Small Language Models (SLMs)
ollama pull phi3:mini

# Large Language Models (LLMs)
ollama pull llama3:latest

# Agent Models
ollama pull qwen2.5:4b
ollama pull deepseek-r1:7b
```

Verify models:
```bash
ollama list
```

You should see:
- phi3:mini
- llama3:latest
- qwen2.5:4b
- deepseek-r1:7b

#### 3. Python Environment

Python 3.9 or higher required:
```bash
python3 --version
```

### Installation

#### 1. Navigate to Project Directory

```bash
cd /path/to/evirag
```

#### 2. Install Python Dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

Or with virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 3. Prepare Your Corpus

Create corpus directory and add PDF files:

```bash
mkdir -p corpus
# Copy your scientific PDFs to corpus/
cp /path/to/your/papers/*.pdf corpus/
```

**Corpus Guidelines:**
- Format: PDF only
- Size: 1-2 GB optimal (~500-1000 papers)
- Domain: AI/ML/Data Science (or customize)
- Quality: Prefer papers with figures and clear structure

### Running EVIRAG

#### Mode 1: Streamlit UI (Recommended for Exploration)

```bash
streamlit run ui/streamlit_app.py
```

Or using the run script:
```bash
python3 run.py --mode ui
```

Then:
1. Open browser to http://localhost:8501
2. Click "Initialize System" in sidebar (wait 1-2 min)
3. Enter your query
4. Click "Query" and explore results in tabs

#### Mode 2: CLI (Recommended for Development)

Interactive mode:
```bash
python3 run.py --mode cli
```

Single query:
```bash
python3 run.py --query "Is overparameterization necessary for generalization?"
```

With specific execution mode:
```bash
python3 run.py --query "Your question" --exec-mode evirag
```

Rebuild index:
```bash
python3 run.py --rebuild
```

#### Mode 3: FastAPI Service (Recommended for Production/n8n)

```bash
python3 orchestration/fastapi_service.py
```

Or with uvicorn directly:
```bash
uvicorn orchestration.fastapi_service:app --reload --host 0.0.0.0 --port 8000
```

API Endpoints:
- `GET /` - Service info
- `GET /health` - Health check
- `POST /api/process_query` - Main query endpoint
- `POST /api/batch_process` - Batch processing
- `GET /api/corpus/stats` - Corpus statistics

Test with curl:
```bash
curl -X POST "http://localhost:8000/api/process_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Is overparameterization necessary?",
    "config": {
      "mode": "evirag",
      "use_visual_grounding": true
    }
  }'
```

### Execution Modes Explained

#### 1. Vanilla RAG (Baseline)
```bash
# CLI
python3 run.py --exec-mode vanilla_rag --query "Your question"

# API
curl -X POST "http://localhost:8000/api/vanilla_rag?query=Your question"
```

**What it does:**
- Simple semantic search
- Direct synthesis
- No disagreement modeling
- Fast (~5 seconds)

#### 2. EVIRAG (Full Pipeline)
```bash
# CLI
python3 run.py --exec-mode evirag --query "Your question"

# API
curl -X POST "http://localhost:8000/api/process_query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question"}'
```

**What it does:**
- Epistemic intent analysis
- Hypothesis generation
- Multi-agent deliberative retrieval
- Claim extraction and disagreement graph  
- Visual evidence grounding
- Full confidence calibration
- Comprehensive but slower (~1-2 minutes)

### Configuration

#### Customize Agents

In Streamlit UI: Uncheck agents in sidebar

In API:
```json
{
  "query": "Your question",
  "config": {
    "mode": "evirag",
    "enabled_agents": ["precision", "skeptic"],
    "use_visual_grounding": true,
    "depth_vs_speed": "balanced"
  }
}
```

#### Adjust Depth vs Speed

- **fast**: Fewer retrieval iterations, SLMs only
- **balanced**: Default configuration
- **thorough**: More agents, LLM escalation, deeper analysis

#### Edit Core Config

Edit `config.py` to customize:

```python
# Model selection
SLM_MODELS["intent_analyzer"].name = "phi3:mini"

# Retrieval parameters
AGENT_CONFIG["precision"]["retrieval_k"] = 10

# Confidence weights
CONFIDENCE_CONFIG["factors"]["visual_alignment"] = 0.2
```

### Troubleshooting

#### "System not initialized"

**Cause:** Corpus not processed yet

**Solution:**
```bash
# CLI: System initializes automatically
python3 run.py --rebuild

# UI: Click "Initialize System" button
# API: Will initialize on first startup
```

#### "No PDF files found"

**Cause:** Empty corpus directory

**Solution:**
```bash
# Check corpus
ls corpus/

# Add PDFs
cp /path/to/papers/*.pdf corpus/
```

#### "Ollama connection failed"

**Cause:** Ollama not running or wrong port

**Solution:**
```bash
# Check Ollama is running
ollama list

# If not, start it
ollama serve

# Check endpoint (default: http://localhost:11434)
curl http://localhost:11434/api/tags
```

#### "Out of memory"

**Cause:** Too many chunks or large corpus

**Solution:**

1. Reduce chunk size in `config.py`:
```python
EMBEDDING_CONFIG["chunk_size"] = 256  # Default: 512
```

2. Limit corpus size:
```bash
# Keep only recent papers
ls -t corpus/*.pdf | tail -n +500 | xargs rm
```

3. Disable visual grounding:
```python
config = EVIRAGConfig(use_visual_grounding=False)
```

#### "Slow query processing"

**Solutions:**

1. Use faster mode:
```bash
python3 run.py --exec-mode vanilla_rag  # Fastest baseline
```

2. Reduce agents:
```python
config = EVIRAGConfig(enabled_agents=["precision", "recall"])
```

3. Adjust depth:
```python
config = EVIRAGConfig(depth_vs_speed="fast")
```

### Example Queries

**Disputed Claims:**
```
Is overparameterization necessary for generalization in deep neural networks?
```

**Comparative Questions:**
```
Which is more effective for language models: RLHF or DPO?
```

**Causal Questions:**
```
Does batch normalization improve convergence or just change the loss landscape?
```

**Methodology Debates:**
```
Should we optimize for BLEU score or human evaluation in NMT?
```

### Advanced Usage

#### Batch Processing

Create queries file:
```bash
cat > queries.txt << EOF
Is overparameterization necessary?
Does attention improve transformers?
What causes double descent?
EOF
```

Process batch:
```python
from evirag_system import EVIRAGSystem, EVIRAGConfig
import json

system = EVIRAGSystem()
system.initialize_corpus()

with open('queries.txt', 'r') as f:
    queries = [line.strip() for line in f if line.strip()]

results = []
for query in queries:
    result = system.query(query)
    results.append(result)

with open('batch_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

#### Custom Agent

Create custom agent in `agents/multi_agent.py`:

```python
class CustomAgent(EpistemicAgent):
    def generate_retrieval_query(self, hypothesis, state):
        # Your custom query generation logic
        return "custom query"
    
    def evaluate_stance(self, hypothesis, claims):
        # Your custom stance evaluation
        return "support", 0.8, "reasoning"
```

Add to orchestrator:
```python
orchestrator.agents.append(CustomAgent(...))
```

#### Export Results

```python
import json
import pandas as pd

result = system.query("Your question")

# Save JSON
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

# Convert to DataFrame
claims_data = []
for claim in result['claims']:
    claims_data.append({
        'text': claim['text'],
        'source': claim['source_doc_title'],
        'stance': claim.get('stance', 'neutral')
    })

df = pd.DataFrame(claims_data)
df.to_csv('claims.csv', index=False)
```

### Performance Benchmarks

**Corpus Size:** 1000 papers, ~1.5 GB

**Hardware:** M1 Mac, 16 GB RAM

| Mode | Processing Time | Claims | Relationships |
|------|----------------|--------|---------------|
| Vanilla RAG | 5s | 0 | 0 |
| EVIRAG (2 agents) | 45s | 35 | 60 |
| EVIRAG (4 agents) | 90s | 55 | 95 |

### Next Steps

1. **Explore Results**: Run example queries, explore UI tabs
2. **Customize Agents**: Edit agent strategies for your domain
3. **Tune Confidence**: Adjust calibration weights
4. **Scale Up**: Add more papers to corpus
5. **Integrate**: Use API for production workflows
6. **Evaluate**: Run systematic evaluation on test queries

### Getting Help

- Check `README.md` for architecture overview
- Read docstrings in source code
- Open GitHub issues
- Contact: your.email@domain.com

---

**Ready to explore scientific disagreement with EVIRAG!** 🚀
