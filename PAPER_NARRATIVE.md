# EVIRAG: The Paper Narrative

## The Single Thesis

> **Standard RAG commits epistemic collapse — it compresses genuine scientific controversy into false consensus. EVIRAG is the first retrieval-augmented generation framework designed for epistemic fidelity: treating disagreement as the primary output signal, not noise to suppress.**

Everything in the paper is a lens on this single claim. There is no "contribution 2" or "contribution 3." There is one argument, shown from multiple angles.

---

## How to Position Against Prior Work

The clearest way to explain EVIRAG's novelty is the contrast table:

| Prior System | Their problem | Their solution | What they assume |
|---|---|---|---|
| MADAM-RAG (2025) | Ambiguous queries + misinformation | Multi-agent debate to find best answer(s) | A correct answer exists; conflict is noise |
| ContraCrow / PaperQA2 (2024) | Contradiction detection per paper | Binary contradiction label | Contradiction = error to be caught |
| SciFact / SciFact-Open | Claim verification | SUPPORTS / REFUTES / NEI label | Evidence is adjudicable |
| TruthfulRAG | Factual conflict in RAG | Suppress conflicting sources | One source is "the truth" |
| **EVIRAG** | **Scientific controversy in RAG** | **Structure and surface the disagreement** | **Disagreement IS the signal** |

The key distinction: MADAM-RAG handles *ambiguity* (e.g., "Who is the capital of the Netherlands?" — Amsterdam or The Hague depending on context). EVIRAG handles *controversy* (e.g., "Does homework improve achievement?" — experts genuinely disagree, resolution is premature). Resolving the second class with consensus-seeking is epistemically dishonest.

---

## The Story Arc (Section by Section)

### §1 Introduction — The Consensus Assumption

Open with the concrete example:
- A user asks: "Does homework improve academic achievement?"
- Vanilla RAG says: "Research shows homework improves academic performance."
- EVIRAG says: "Dominant view (5 sources): No conclusive evidence since 1987. Alternative view (3 sources): Effects depend on design and grade level. Minority view (2 sources): Excess homework harms wellbeing. Disagreement density: 5.3%. Confidence: MEDIUM."

Which answer would you trust? The first is fluent but epistemically false. The second is honest about the state of knowledge.

Claim: The first answer is the product of what we call **epistemic collapse** — the systematic compression of scientific controversy into a single confident response. We show this is not a flaw in implementation but a structural consequence of the consensus assumption embedded in standard RAG design.

### §2 Related Work — Why Existing Systems Don't Solve This

Three clusters of prior work, all addressing a different problem:

1. **Conflict resolution RAG** (MADAM-RAG, TruthfulRAG, DRAGged): Resolve conflicts → still collapse controversy. Wrong problem class.
2. **Claim verification** (SciFact, ContraCrow): Binary labels → doesn't structure multi-view controversy for the reader. Wrong output format.
3. **Calibration / epistemic integrity** (Epistemic Integrity in LLMs, 2024): Addresses linguistic assertiveness → individual model outputs, not retrieval systems. Different layer.

EVIRAG is the first system to target *epistemic fidelity in retrieval* — faithfully representing the structure of scientific disagreement as the primary output.

### §3 The Epistemic Collapse Problem (Formalized)

Define epistemic collapse formally using the ConsensusCollapseScore (CCS):

```
CCS = 1 - cosine_sim(single_view_centroid, multi_view_centroid_mean)
```

High CCS = high information loss from collapsing to one view.

Show empirically: vanilla RAG, MADAM-RAG, single-agent RAG all produce high CCS on contested scientific questions. EVIRAG produces low CCS.

This section converts the intuition into a measurable claim. The CCS is the "effect size" that motivates the whole paper.

### §4 EVIRAG: Epistemic-Fidelity-First Retrieval

EVIRAG is not a collection of modules. It is a single architectural answer to the epistemic collapse problem. Everything in the architecture serves this goal:

**Why adversarial multi-agent retrieval?**
Because a single retrieval agent — even a good one — will find evidence that confirms the hypothesis. You need agents with *opposing* epistemic objectives to escape the consensus attractor. The Skeptic agent's job is to prevent collapse.

**Why a disagreement graph?**
Because collapse happens at the synthesis step: if all you have is a bag of retrieved chunks, the LLM synthesizer will average them. The graph makes the structure of controversy explicit, so synthesis cannot silently collapse it.

**Why multi-view synthesis (not single-answer)?**
Because epistemic fidelity requires the output to mirror the structure of the evidence. If three camps of researchers exist in the literature, three views must appear in the output.

**Why confidence from evidence structure, not fluency?**
Because fluency-based confidence calibrates to how coherent the answer sounds, not how contested the evidence is. A highly contested question will always sound coherent if synthesized by a capable LLM.

### §5 Understanding the Collapse (Supporting Analysis)

These sections deepen the story — they are not separate contributions, they are evidence that the epistemic collapse problem is real, structured, and measurable.

**§5a Causal Attribution (Why collapse is premature)**
Using the CDA-7 taxonomy, we show that disagreements in scientific RAG are not noise: they are systematic, attributable to methodological differences, population differences, and theoretical paradigm conflicts. Resolving them by averaging is not just unhelpful — it is wrong. Methodological disagreements need methodological resolution, not synthesis.

**§5b Temporal Drift (When collapse is most harmful)**
We show that epistemic collapse is most damaging in *emerging controversies* — topics where disagreement is increasing. A system that resolves the debate prematurely is most harmful precisely when the science is most alive. Temporal tracking shows that for diverging trajectories, confidence should decrease — the opposite of what fluency-based calibration produces.

### §6 Evaluation

**Benchmark: EVIRAG-Bench**
A new benchmark specifically designed to measure epistemic fidelity. Standard IR benchmarks (NDCG, MAP) cannot measure this — they assume a single correct answer. EVIRAG-Bench measures:

- **ContradictionRecall**: Do we find known contradictions?
- **ViewpointCoverage**: Do we surface all known scientific views?
- **ConsensusCollapseScore**: How much viewpoint information do we lose vs. multi-view ground truth?
- **ConfidenceCalibrationError**: Is expressed confidence calibrated to actual controversy?

**Ablation table** (this is the key table for reviewers):

| System | CD-Recall | VC | CCS↓ | CCE↓ |
|---|---|---|---|---|
| Vanilla RAG | low | low | high | high |
| MADAM-RAG | medium | low | high | medium |
| EVIRAG -Skeptic | medium | medium | medium | medium |
| EVIRAG -Graph | medium | medium | medium | medium |
| EVIRAG -MultiView | high | low | medium | medium |
| **EVIRAG Full** | **high** | **high** | **low** | **low** |

Each ablation removes one piece of the architecture and shows collapse increases. This is the proof that every component serves the single thesis.

---

## What NOT to Put in the Paper

- Do NOT frame EpistemicDivergence, CausalAttribution, TemporalTracking as "contributions."
- Frame them as: "analytical tools we use to understand and measure the epistemic collapse problem."
- The one contribution is EVIRAG as a framework. Everything else is evidence, analysis, and evaluation.

---

## The One-Liner for Every Section Header

- Abstract: "RAG collapses scientific controversy. EVIRAG preserves it."
- §1: "The consensus assumption in RAG is epistemically harmful."
- §2: "Existing work resolves conflict — we preserve it."
- §3: "We measure epistemic collapse with ConsensusCollapseScore."
- §4: "Every EVIRAG component fights collapse."
- §5: "Understanding what's being collapsed: causal structure and temporal drift."
- §6: "Evaluating epistemic fidelity, not answer accuracy."

---

## Target Venue Positioning

**Best fit: EMNLP 2026 (main track) or ACL 2026**
- Scientific NLP is a core EMNLP strength
- The task formulation (epistemic fidelity) is novel and well-motivated
- The system paper format (new task + new system + new benchmark) is the standard for this venue

**Second choice: AAAI 2026**
- The epistemological framing fits AAAI's broader scope
- Multi-agent system is directly in scope

**Avoid: NeurIPS / ICML**
- They want theoretical guarantees or large-scale empirical results
- This is primarily a systems + task formulation paper

---

## Concrete Next Steps for the Paper

1. **Run the baseline experiments**: Get CCS for vanilla RAG and MADAM-RAG on the 3 EVIRAG-Bench instances. This is the empirical core of the paper.
2. **Write §3 first**: The formalization of epistemic collapse is the hardest section and the one reviewers will scrutinize most.
3. **Annotate 10-20 real contradictions** from your corpus manually — this is your EVIRAG-Bench. The 3 seeded ones are a start.
4. **Run the ablation**: Disable skeptic agent, then disable graph, then disable multi-view. Show CCS increases each time.
5. **Human evaluation**: Ask 3-5 people which answer (vanilla RAG vs EVIRAG) they find more epistemically trustworthy. This is the "user study" section.
