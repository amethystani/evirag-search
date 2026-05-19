"""
EVIRAG smoke test — verifies the full pipeline works with qwen3:0.6b.
Run: python3 smoke_test.py
"""

import sys
import time
import json

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ── 1. Ollama reachable ──────────────────────────────────────────────────────
section("1/6  Ollama + qwen3:0.6b")
import requests
try:
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "qwen3:0.6b", "prompt": "Say READY", "stream": False,
              "options": {"num_predict": 300, "temperature": 0.0}},
        timeout=60,
    )
    r.raise_for_status()
    resp = r.json().get("response", "").strip()
    thinking = r.json().get("thinking", "")
    print(f"  response : {repr(resp[:80])}")
    print(f"  thinking : {len(thinking)} chars (discarded)")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    print("  → Is Ollama running?  Try: ollama serve")
    sys.exit(1)

# ── 2. Config loads ───────────────────────────────────────────────────────────
section("2/6  Config")
try:
    from config import SLM_MODELS, LLM_MODELS, AGENT_MODELS, GRAPH_CONFIG, SYSTEM_CONSTRAINTS
    model_name = SLM_MODELS["intent_analyzer"].name
    assert model_name == "qwen3:0.6b", f"Expected qwen3:0.6b, got {model_name}"
    print(f"  all models → {model_name}")
    print(f"  max_pairs  → {GRAPH_CONFIG['max_pairs']}")
    print(f"  max_claims → {SYSTEM_CONSTRAINTS['max_claims_per_query']}")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# ── 3. OllamaClient generate ──────────────────────────────────────────────────
section("3/6  OllamaClient (epistemic engine)")
try:
    from epistemic_engine import OllamaClient
    client = OllamaClient()
    t0 = time.time()
    result = client.generate(
        model="qwen3:0.6b",
        prompt='Return JSON only, no explanation:\n{"status": "ok", "value": 42}',
        temperature=0.0,
        max_tokens=80,
    )
    elapsed = time.time() - t0
    print(f"  response : {repr(result[:120])}")
    print(f"  latency  : {elapsed:.1f}s")
    if not result:
        print("  WARN: empty response — thinking budget may need to be larger")
    else:
        print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# ── 4. Sentence embeddings ────────────────────────────────────────────────────
section("4/6  Sentence embeddings (MiniLM)")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(["Test sentence"])
    print(f"  embedding dim: {emb.shape[1]}")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    print("  → pip install sentence-transformers")

# ── 5. Corpus check ───────────────────────────────────────────────────────────
section("5/6  Corpus")
from pathlib import Path
corpus = Path("corpus")
pdfs = list(corpus.glob("*.pdf")) if corpus.exists() else []
print(f"  PDFs found: {len(pdfs)}")
if pdfs:
    for p in pdfs[:3]:
        print(f"    {p.name}")
    if len(pdfs) > 3:
        print(f"    ... and {len(pdfs)-3} more")
    print("  PASS")
else:
    print("  WARN: no PDFs in corpus/ — add papers before running full pipeline")

# ── 6. EpistemicDivergence metric ────────────────────────────────────────────
section("6/6  EpistemicDivergence metric")
try:
    from epistemic_divergence import EpistemicDivergenceCalculator
    calc = EpistemicDivergenceCalculator()
    views = [
        {"name": "pro",  "summary": "Homework significantly improves academic achievement.",
         "claim_texts": ["Homework completion correlates with higher grades (r=0.25)",
                         "Structured homework improves study habits."]},
        {"name": "con",  "summary": "No conclusive evidence homework improves achievement.",
         "claim_texts": ["No significant effect found in controlled studies.",
                         "Homework causes stress and reduces wellbeing."]},
    ]
    result = calc.compute(viewpoints=views)
    print(f"  ED score              : {result.ed_score:.3f}")
    print(f"  Consensus collapse    : {result.consensus_collapse_score:.3f}")
    print(f"  Controversy class     : {result.controversy_class}")
    print(f"  Dom-minority gap      : {result.dominant_vs_minority_gap:.3f}")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback; traceback.print_exc()

# ── Summary ───────────────────────────────────────────────────────────────────
section("Summary")
print("  All core components checked.")
print("  To run a full query:")
print()
print('    python3 run.py --mode cli')
print()
print("  Or launch the UI:")
print()
print("    streamlit run streamlit_app.py")
