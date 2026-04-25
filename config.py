"""
EVIRAG Configuration
Evidence-Centric, Disagreement-Aware RAG for Scientific Literature
"""

import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
CORPUS_DIR = PROJECT_ROOT / "corpus"
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
FIGURES_DIR = DATA_DIR / "figures"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Create directories
for dir_path in [CORPUS_DIR, DATA_DIR, CACHE_DIR, VECTOR_STORE_DIR, 
                 FIGURES_DIR, EMBEDDINGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single model"""
    name: str
    type: str  # 'slm', 'llm', 'vlm'
    endpoint: str = "http://localhost:11434"  # Ollama default
    temperature: float = 0.7
    max_tokens: int = 2048
    
# ─────────────────────────────────────────────────────────────────────────────
# LOCAL MODEL: qwen3:0.6b
# qwen3:0.6b is a thinking model — it generates ~150 reasoning tokens BEFORE
# the actual response.  num_predict must be ≥ 400 or the response field is
# empty.  The OllamaClient reads response (not thinking), so all prompts work
# correctly — just set max_tokens high enough to clear the thinking budget.
# ─────────────────────────────────────────────────────────────────────────────
_LOCAL_MODEL = "qwen3:0.6b"

# Small Language Models (SLMs) — simpler classification tasks
SLM_MODELS = {
    "intent_analyzer": ModelConfig(
        name=_LOCAL_MODEL,
        type="slm",
        temperature=0.1,
        max_tokens=400,   # 150 thinking + 250 answer
    ),
    "claim_filter": ModelConfig(
        name=_LOCAL_MODEL,
        type="slm",
        temperature=0.1,
        max_tokens=500,
    ),
    "nli_prefilter": ModelConfig(
        name=_LOCAL_MODEL,
        type="slm",
        temperature=0.1,
        max_tokens=400,
    ),
    "synthesizer": ModelConfig(
        name=_LOCAL_MODEL,
        type="slm",
        temperature=0.3,
        max_tokens=600,
    ),
}

# Large Language Models (LLMs) — same model, more generous token budget
LLM_MODELS = {
    "hypothesis_generator": ModelConfig(
        name=_LOCAL_MODEL,
        type="llm",
        temperature=0.2,
        max_tokens=600,
    ),
    "claim_extractor": ModelConfig(
        name=_LOCAL_MODEL,
        type="llm",
        temperature=0.1,
        max_tokens=500,
    ),
    "nli_verifier": ModelConfig(
        name=_LOCAL_MODEL,
        type="llm",
        temperature=0.1,
        max_tokens=400,
    ),
    "synthesizer_escalation": ModelConfig(
        name=_LOCAL_MODEL,
        type="llm",
        temperature=0.3,
        max_tokens=700,
    ),
}

# Agent Models — all unified on the single available 0.6B model
AGENT_MODELS = {
    "precision": ModelConfig(
        name=_LOCAL_MODEL,
        type="llm",
        temperature=0.1,
        max_tokens=500,
    ),
    "recall": ModelConfig(
        name=_LOCAL_MODEL,
        type="llm",
        temperature=0.3,
        max_tokens=500,
    ),
    "skeptic": ModelConfig(
        name=_LOCAL_MODEL,
        type="llm",
        temperature=0.2,
        max_tokens=500,
    ),
    "counterfactual": ModelConfig(
        name=_LOCAL_MODEL,
        type="llm",
        temperature=0.3,
        max_tokens=500,
    ),
}

# ============================================================================
# EMBEDDING CONFIGURATIONS
# ============================================================================

EMBEDDING_CONFIG = {
    "text_model": "sentence-transformers/all-MiniLM-L6-v2",
    "clip_model": "openai/clip-vit-base-patch16",
    "chunk_size": 256,       # smaller chunks → shorter prompts → faster inference
    "chunk_overlap": 32,
    "embedding_dim": 384,
    "clip_dim": 512,
}

# ============================================================================
# VECTOR STORE CONFIGURATION
# ============================================================================

VECTOR_STORE_CONFIG = {
    "type": "faiss",  # or 'chroma'
    "index_type": "IndexFlatL2",
    "metric": "l2",
    "n_probes": 10
}

# ============================================================================
# PDF PROCESSING
# ============================================================================

PDF_CONFIG = {
    "extract_figures": True,
    "min_figure_size": (100, 100),  # pixels
    "max_figures_per_paper": 20,
    "section_headers": [
        "abstract", "introduction", "related work", "methodology", 
        "methods", "approach", "results", "experiments", "discussion",
        "conclusion", "references"
    ]
}

# ============================================================================
# EPISTEMIC INTENT DIMENSIONS
# ============================================================================

INTENT_DIMENSIONS = {
    "factual_vs_disputed": ["factual", "somewhat_disputed", "highly_disputed"],
    "empirical_vs_theoretical": ["empirical", "mixed", "theoretical"],
    "disagreement_level": ["low", "medium", "high"]
}

# ============================================================================
# AGENT CONFIGURATIONS
# ============================================================================

AGENT_CONFIG = {
    # retrieval_k kept low — each chunk becomes a prompt to qwen3:0.6b
    "precision": {
        "retrieval_k": 3,
        "confidence_threshold": 0.8,
        "objective": "high_confidence_support",
        "search_strategy": "strict_semantic",
    },
    "recall": {
        "retrieval_k": 5,
        "confidence_threshold": 0.5,
        "objective": "broad_coverage",
        "search_strategy": "expansive_semantic",
    },
    "skeptic": {
        "retrieval_k": 4,
        "confidence_threshold": 0.65,
        "objective": "find_contradictions",
        "search_strategy": "adversarial_semantic",
    },
    "counterfactual": {
        "retrieval_k": 3,
        "confidence_threshold": 0.6,
        "objective": "attempt_disproof",
        "search_strategy": "counterfactual_reasoning",
    },
}

# ============================================================================
# DISAGREEMENT GRAPH
# ============================================================================

GRAPH_CONFIG = {
    "edge_types": ["supports", "contradicts", "neutral"],
    "min_edge_confidence": 0.45,
    "max_nodes": 50,          # hard cap — prevents runaway on small model
    "max_pairs": 30,          # 30 NLI calls × ~8s each ≈ 4 min on 0.6B
    "max_pairs_cloud": 200,
    "community_detection": True,
    "fast_mode": True,
}

CLAIM_GRAPH_CONFIG = {
    "enabled": True,
    "max_claims_per_chunk": 3,
    "min_sentence_words": 8,
    "candidate_neighbors": 6,
    "query_seed_k": 8,
    "query_neighbor_expansion": 3,
    "max_query_edges": 40,
    "min_edge_similarity": 0.62,
    "min_support_similarity": 0.72,
    "min_contradiction_similarity": 0.68,
    "verify_edges": "local",  # local-first offline verification
    "verification_budget_local": 8,
    "verification_budget_cloud": 60,
    "verification_margin": 0.04,
    "always_verify_contradictions": True,
}

# ============================================================================
# DISAGREEMENT METRICS
# ============================================================================

METRICS_CONFIG = {
    "disagreement_density": True,
    "conflict_ratio": True,
    "claim_entropy": True,
    "conflict_centrality": True,
    "visual_text_mismatch": True
}

# ============================================================================
# CONFIDENCE CALIBRATION
# ============================================================================

CONFIDENCE_CONFIG = {
    "factors": {
        "claim_agreement": 0.3,
        "source_diversity": 0.2,
        "contradiction_severity": 0.25,
        "unverified_assumptions": 0.15,
        "visual_alignment": 0.1
    },
    "thresholds": {
        "high": 0.75,
        "medium": 0.5,
        "low": 0.25
    }
}

# ============================================================================
# EXECUTION MODES
# ============================================================================

EXECUTION_MODES = ["vanilla_rag", "evirag"]

# ============================================================================
# BACKEND MODES
# ============================================================================

BACKEND_MODES = ["local", "cloud"]

# Cloud Model Configurations (Groq API)
GROQ_SLM_MODELS = {
    "intent_analyzer": ModelConfig(
        name="llama-3.1-8b-instant",
        type="slm",
        endpoint="https://api.groq.com/openai/v1",
        temperature=0.1,
        max_tokens=512
    ),
    "claim_filter": ModelConfig(
        name="llama-3.1-8b-instant",
        type="slm",
        endpoint="https://api.groq.com/openai/v1",
        temperature=0.1,
        max_tokens=1024
    ),
    "nli_prefilter": ModelConfig(
        name="llama-3.1-8b-instant",
        type="slm",
        endpoint="https://api.groq.com/openai/v1",
        temperature=0.1,
        max_tokens=512
    ),
    "synthesizer": ModelConfig(
        name="llama-3.1-8b-instant",
        type="slm",
        endpoint="https://api.groq.com/openai/v1",
        temperature=0.3,
        max_tokens=2048
    )
}

GROQ_LLM_MODELS = {
    "hypothesis_generator": ModelConfig(
        name="llama-3.3-70b-versatile",
        type="llm",
        endpoint="https://api.groq.com/openai/v1",
        temperature=0.2,
        max_tokens=1024
    ),
    "claim_extractor": ModelConfig(
        name="llama-3.3-70b-versatile",
        type="llm",
        endpoint="https://api.groq.com/openai/v1",
        temperature=0.1,
        max_tokens=1024
    ),
    "nli_verifier": ModelConfig(
        name="llama-3.3-70b-versatile",
        type="llm",
        endpoint="https://api.groq.com/openai/v1",
        temperature=0.1,
        max_tokens=512
    ),
    "synthesizer_escalation": ModelConfig(
        name="llama-3.3-70b-versatile",
        type="llm",
        endpoint="https://api.groq.com/openai/v1",
        temperature=0.3,
        max_tokens=3072
    )
}

GROQ_AGENT_MODELS = {
    "precision": ModelConfig(
        name="llama-3.3-70b-versatile",
        type="llm",
        endpoint="https://api.groq.com/openai/v1",
        temperature=0.1,
        max_tokens=1024
    ),
    "recall": ModelConfig(
        name="llama-3.1-8b-instant",  # Fast for broad retrieval
        type="llm",
        endpoint="https://api.groq.com/openai/v1",
        temperature=0.5,
        max_tokens=1024
    ),
    "skeptic": ModelConfig(
        name="deepseek-r1-distill-llama-70b",
        type="llm",
        endpoint="https://api.groq.com/openai/v1",
        temperature=0.2,
        max_tokens=1024
    ),
    "counterfactual": ModelConfig(
        name="deepseek-r1-distill-llama-70b",
        type="llm",
        endpoint="https://api.groq.com/openai/v1",
        temperature=0.3,
        max_tokens=1024
    )
}

# ============================================================================
# STREAMLIT UI
# ============================================================================

UI_CONFIG = {
    "title": "EVIRAG: Evidence-Centric Disagreement-Aware RAG",
    "theme": "dark",
    "tabs": [
        "Multi-View Answer",
        "Disagreement Graph", 
        "Claims & Evidence",
        "Visual Evidence",
        "Confidence & Metrics"
    ],
    "sidebar_controls": [
        "execution_mode",
        "enable_agents",
        "depth_vs_speed",
        "visual_grounding"
    ]
}

# ============================================================================
# EVALUATION
# ============================================================================

EVAL_CONFIG = {
    "baselines": ["vanilla_rag", "single_agent_rag"],
    "metrics": [
        "hallucination_rate",
        "contradiction_detection_accuracy",
        "viewpoint_coverage",
        "confidence_calibration_error",
        "visual_grounding_impact"
    ],
    "test_queries": []  # populated during evaluation
}

# ============================================================================
# SYSTEM CONSTRAINTS
# ============================================================================

SYSTEM_CONSTRAINTS = {
    "max_memory_gb": 16,
    "max_concurrent_agents": 4,   # 4-agent mode runs role agents concurrently
    "max_retrieval_chunks": 15,
    "max_claims_per_query": 15,   # 15 claims × 4 agents = 60 max NLI inputs
    "max_claims_cloud": 60,
    "cache_embeddings": True,
    "use_gpu": False,
}

# ============================================================================
# PROMPTS LIBRARY
# ============================================================================

PROMPTS = {
    "intent_analysis": """Analyze the following scientific query and classify it along these dimensions:

Query: {query}

Provide classifications for:
1. Factual vs Disputed: Is this a settled fact or an area of active disagreement?
2. Empirical vs Theoretical: Does this require experimental evidence or theoretical reasoning?
3. Expected Disagreement Level: How much disagreement do you expect in the literature?

Respond in JSON format:
{{
    "factual_vs_disputed": "factual|somewhat_disputed|highly_disputed",
    "empirical_vs_theoretical": "empirical|mixed|theoretical",
    "disagreement_level": "low|medium|high",
    "reasoning": "brief explanation"
}}""",

    "hypothesis_generation": """Given this scientific query, generate a structured explanation hypothesis.

Query: {query}

Intent Analysis: {intent}

Generate a hypothesis that will be verified against scientific literature:

{{
    "central_hypothesis": "main explanatory claim",
    "supporting_claims": ["claim 1", "claim 2", ...],
    "assumptions": ["assumption 1", "assumption 2", ...],
    "expected_counterclaims": ["potential counter 1", "potential counter 2", ...]
}}

Be specific and verifiable. Each claim should be atomic and falsifiable.""",

    "claim_filtering": """Identify sentences from the following text that contain factual scientific claims.

Text: {text}

Source: {source}

Return only sentences that:
- Make specific factual assertions
- Are verifiable
- Are substantive (not just background)

Respond with a JSON list of claim sentences.""",

    "claim_extraction": """Extract atomic, normalized claims from these sentences.

Sentences: {sentences}

Source: {source}

Rules:
- One claim per output
- Normalize terminology
- Preserve attribution
- Make claims self-contained

Output format:
{{
    "claims": [
        {{
            "claim": "normalized claim text",
            "source": "source identifier",
            "context": "brief context if needed"
        }}
    ]
}}""",

    "nli_verification": """Determine the relationship between these two claims.

Claim 1: {claim1}
Source 1: {source1}

Claim 2: {claim2}
Source 2: {source2}

Relationship: Does Claim 2 entail, contradict, or remain neutral to Claim 1?

Respond with JSON (no extra text):
{{
    "relationship": "entailment|contradiction|neutral",
    "certainty": "certain|likely|uncertain",
    "reasoning": "brief explanation"
}}

Use "certain" only when the relationship is unambiguous. Use "uncertain" when the evidence is weak or mixed.""",

    "agent_retrieval": """You are the {agent_name} agent in a multi-agent scientific retrieval system.

Objective: {objective}

Hypothesis to evaluate: {hypothesis}

Current evidence state: {evidence_state}

Your retrieval strategy: {strategy}

Generate a retrieval query that serves your specific objective. Consider:
- What evidence would support/contradict this hypothesis?
- What gaps exist in current evidence?
- What alternative interpretations are possible?

Return: {{
    "query": "your retrieval query",
    "reasoning": "why this query serves your objective"
}}""",

    "synthesis": """Synthesize a multi-view answer from the following evidence.

Query: {query}

Claims Graph: {claims_graph}

Disagreement Metrics: {metrics}

Generate:
1. Dominant View: Most supported interpretation
2. Alternative Views: Other supported interpretations
3. Minority Views: Outlier but notable positions

For each view, include:
- Summary
- Supporting claim IDs
- Source count
- Known weaknesses
- Visual evidence strength (if applicable)

Format as structured JSON.""",

    "confidence_calibration": """Calculate calibrated confidence for this answer.

Supporting Claims: {support_count}
Contradicting Claims: {contradict_count}
Source Diversity: {source_diversity}
Unverified Assumptions: {assumptions}
Visual Evidence Alignment: {visual_alignment}

Use weights:
- Claim agreement: 0.3
- Source diversity: 0.2
- Contradiction severity: 0.25
- Unverified assumptions: 0.15
- Visual alignment: 0.1

Output:
{{
    "confidence": "high|medium|low",
    "score": 0.0-1.0,
    "reasoning": "factor breakdown"
}}"""
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_config(model_key: str, model_type: str = "slm", backend: str = "local") -> ModelConfig:
    """
    Get model configuration by key and backend
    
    Args:
        model_key: Model identifier (e.g., "intent_analyzer")
        model_type: "slm", "llm", or "agent"
        backend: "local" (Ollama) or "cloud" (Groq)
    
    Returns:
        ModelConfig or None if not found
    """
    if backend == "cloud":
        # Use Groq cloud models
        if model_type == "slm":
            return GROQ_SLM_MODELS.get(model_key)
        elif model_type == "llm":
            return GROQ_LLM_MODELS.get(model_key)
        elif model_type == "agent":
            return GROQ_AGENT_MODELS.get(model_key)
    else:
        # Use local Ollama models
        if model_type == "slm":
            return SLM_MODELS.get(model_key)
        elif model_type == "llm":
            return LLM_MODELS.get(model_key)
        elif model_type == "agent":
            return AGENT_MODELS.get(model_key)
    
    return None

def get_prompt(prompt_key: str, **kwargs) -> str:
    """Get and format a prompt template"""
    template = PROMPTS.get(prompt_key, "")
    return template.format(**kwargs)

def validate_corpus() -> bool:
    """Check if corpus directory exists and contains PDFs"""
    if not CORPUS_DIR.exists():
        return False
    pdf_files = list(CORPUS_DIR.glob("*.pdf"))
    return len(pdf_files) > 0

def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging"""
    return {
        "corpus_exists": CORPUS_DIR.exists(),
        "corpus_size": len(list(CORPUS_DIR.glob("*.pdf"))) if CORPUS_DIR.exists() else 0,
        "vector_store_exists": (VECTOR_STORE_DIR / "index.faiss").exists(),
        "cache_exists": CACHE_DIR.exists(),
        "models_configured": {
            "slm": len(SLM_MODELS),
            "llm": len(LLM_MODELS),
            "agents": len(AGENT_MODELS)
        }
    }

if __name__ == "__main__":
    print("EVIRAG Configuration")
    print("=" * 60)
    info = get_system_info()
    for key, value in info.items():
        print(f"{key}: {value}")
