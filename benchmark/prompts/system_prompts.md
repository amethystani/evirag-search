# EVIRAG System Prompts

All prompts used in the EVIRAG Full system and all baselines.
Model: `qwen3.6:35b-a3b` via Ollama. `think=False` for all structured outputs.

---

## Vanilla RAG

```
Based on the following passages, answer the question concisely and accurately.

Passages:
{passages}

Question: {query}

Answer:
```

---

## EVIRAG — Stage 2: Adversarial Retrieval Agent Prompts

### Precision Agent
```
Retrieve the most directly relevant, high-confidence evidence for the following question.
Focus on well-supported, peer-reviewed findings.

Query: {query}
```

### Recall Agent
```
Retrieve a broad range of evidence relevant to the following question.
Include both primary studies and review articles.

Query: {query}
```

### Skeptic Agent
```
Find passages that CHALLENGE, CONTRADICT, or QUALIFY the dominant view on the following question.
Focus specifically on dissenting findings, null results, and critical analyses.

Query: {query}
```

### Counterfactual Agent
```
Find passages presenting ALTERNATIVE EXPLANATIONS or MINORITY POSITIONS on the following question.
Look for different theoretical frameworks, different populations studied, or different outcome measures.

Query: {query}
```

---

## EVIRAG — Stage 3: Claim Extraction

```
Extract all atomic factual claims from the following passage.
Return a JSON list where each item has: {"claim": "...", "confidence": 0.0-1.0}
Each claim must be a single, self-contained factual statement.
Do not include opinions, recommendations, or methodological descriptions.

Passage:
{passage}
```

## EVIRAG — Stage 3: NLI Relation Labeling

```
Classify the relationship between the following two scientific claims.

Claim 1: {claim1}
Claim 2: {claim2}

Relationship options:
- SUPPORTS: Claim 2 provides evidence for or is consistent with Claim 1
- CONTRADICTS: Claim 2 directly conflicts with Claim 1
- NEUTRAL: No clear logical relationship

Return JSON: {"label": "SUPPORTS|CONTRADICTS|NEUTRAL", "confidence": 0.0-1.0, "reasoning": "..."}
```

---

## EVIRAG — Stage 4: CDA-7 Attribution

```
Classify the cause of disagreement between the following two conflicting scientific claims
using the CDA-7 taxonomy.

Claim 1: {claim1} (Source: {source1})
Claim 2: {claim2} (Source: {source2})

CDA-7 Classes:
1. METHODOLOGICAL - Different experimental designs or protocols
2. POPULATION - Different study populations or settings
3. TEMPORAL - Newer evidence supersedes older claims
4. OPERATIONAL - Different operationalizations of key terms
5. STATISTICAL - Conflicting effect sizes or p-value interpretations
6. THEORETICAL - Competing theoretical frameworks
7. REPLICATION - Failure to replicate original findings

Return JSON: {"cda7_class": "...", "confidence": 0.0-1.0, "reasoning": "..."}
Note: Multiple classes allowed as comma-separated string (e.g., "METHODOLOGICAL,POPULATION")
```

---

## EVIRAG — Stage 6: Multi-View Synthesis

```
You are synthesizing a multi-view scientific response. Do NOT resolve the controversy.
Do NOT pick a winner. Present each viewpoint faithfully.

Query: {query}
Controversy class: {controversy_class}
CDA-7 attribution: {cda7_attribution}

Viewpoint communities from claim graph:
{viewpoint_communities}

For each viewpoint, generate:
{
  "view_id": "V1|V2|...",
  "label": "one-phrase label",
  "summary": "2-3 sentence summary",
  "supporting_claim_ids": [...],
  "source_count": N,
  "stated_weaknesses": "...",
  "cda7_attribution": "..."
}

Return JSON array of viewpoints. The dominant view (most claims) goes first.
Do not add editorial judgment about which view is correct.
```

---

## MADAM-RAG (adapted from Wang et al. 2025)

### Proposer Agent
```
Answer the following question based on the retrieved passages.
Make a clear, supported argument for your answer.

Passages: {passages}
Question: {query}
```

### Opposer Agent
```
Challenge the following answer. Find evidence in the passages that contradicts or qualifies it.

Answer to challenge: {proposer_answer}
Passages: {passages}
Question: {query}
```

### Moderator Agent
```
Given the following debate, produce the best final answer that resolves the conflict.
Choose the most evidence-supported position.

Proposer: {proposer_answer}
Opposer: {opposer_answer}
Question: {query}
```

---

## MMR-RAG

Retrieval: Maximal Marginal Relevance with λ=0.5 (cosine similarity and diversity weighted equally).
Synthesis: Same prompt as Vanilla RAG (single-view).
