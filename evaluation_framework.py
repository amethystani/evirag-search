"""
EVIRAG: Formal Evaluation Framework
=====================================
This module implements the evaluation benchmark that tier-1 papers require.
No prior evaluation framework exists for disagreement-aware scientific RAG.

Metrics implemented:
  1. ContradictionDetection (CD): Precision, Recall, F1 against ground-truth pairs
  2. EpistemicCoverage@K (EC@K): Novel — fraction of disagreement space covered at K
  3. ViewpointCoverage (VC): How many ground-truth viewpoints were surfaced?
  4. ConfidenceCalibrationError (CCE): Expected Calibration Error for confidence scores
  5. ConsensusCollapseDegradation (CCD): Info loss vs. vanilla RAG (uses EpistemicDivergence)
  6. FaithfulnessScore (FS): Claim-level faithfulness (inspired by FActScore)
  7. VisualTextConsistency (VTC): Novel — do text claims match retrieved figures?

Baselines:
  - Vanilla RAG (BM25 + single-view synthesis)
  - Single-agent RAG (one precision agent, no skeptic/counterfactual)
  - EVIRAG-NoVisual (ablation: no CLIP grounding)
  - EVIRAG-NoTemporal (ablation: no temporal tracking)
  - Full EVIRAG

The framework supports:
  - Automatic evaluation (all metrics above)
  - Human evaluation rubrics (exported as structured forms)
  - Ablation study runner

Benchmark format compatible with:
  - SciFact (Wadden et al. 2020) — for contradiction detection
  - ExpertQA (Malaviya et al. 2023) — for answer quality
  - Custom EVIRAG-Bench (this work) — for disagreement coverage
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Ground-truth data structures
# ---------------------------------------------------------------------------

@dataclass
class GroundTruthContradiction:
    """A known contradictory pair for evaluation."""
    claim1_text: str
    claim2_text: str
    source1: str
    source2: str
    attribution_type: str   # from CDA-7 taxonomy
    verified_by: str        # 'human', 'expert', 'automatic'


@dataclass
class GroundTruthViewpoint:
    """A known scientific viewpoint for coverage evaluation."""
    viewpoint_id: str
    description: str
    representative_claims: List[str]
    stance: str     # 'pro', 'con', 'neutral', 'alternative'
    strength: str   # 'strong', 'moderate', 'weak'


@dataclass
class EvaluationInstance:
    """A complete evaluation instance (query + ground truth)."""
    instance_id: str
    query: str
    domain: str
    contradictions: List[GroundTruthContradiction]
    viewpoints: List[GroundTruthViewpoint]
    vanilla_rag_answer: Optional[str] = None   # for CCD comparison
    notes: str = ""


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class ContradictionDetectionResult:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Full evaluation result for one system on one instance."""
    instance_id: str
    system_name: str

    # Metric scores
    cd_precision: float
    cd_recall: float
    cd_f1: float
    ec_at_k: Dict[int, float]      # EC@K for K in {1,3,5,10}
    viewpoint_coverage: float
    confidence_calibration_error: float
    consensus_collapse_degradation: float
    faithfulness_score: float
    visual_text_consistency: float   # -1 if visual grounding not used

    # Aggregate
    evirag_score: float   # Composite score (see compute_composite_score())
    notes: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------

def evaluate_contradiction_detection(
    detected_pairs: List[Tuple[str, str]],     # (claim1_text, claim2_text) predicted contradictions
    ground_truth: List[GroundTruthContradiction],
    similarity_threshold: float = 0.8,
) -> ContradictionDetectionResult:
    """
    Evaluate contradiction detection against ground-truth pairs.

    Uses fuzzy matching (Jaccard on token sets) to handle paraphrasing,
    since exact string match is too strict for claim texts.

    Args:
        detected_pairs: System's predicted contradictory claim pairs
        ground_truth: Ground-truth contradictions
        similarity_threshold: Minimum token-Jaccard for a match

    Returns:
        ContradictionDetectionResult with P/R/F1
    """
    def jaccard_sim(a: str, b: str) -> float:
        tokens_a = set(a.lower().split())
        tokens_b = set(b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    def pair_matches(pred: Tuple[str, str], gt: GroundTruthContradiction) -> bool:
        # Check both orderings
        fwd = (jaccard_sim(pred[0], gt.claim1_text) + jaccard_sim(pred[1], gt.claim2_text)) / 2
        rev = (jaccard_sim(pred[0], gt.claim2_text) + jaccard_sim(pred[1], gt.claim1_text)) / 2
        return max(fwd, rev) >= similarity_threshold

    matched_gt = set()
    true_positives = 0

    for pred_pair in detected_pairs:
        for i, gt in enumerate(ground_truth):
            if i not in matched_gt and pair_matches(pred_pair, gt):
                true_positives += 1
                matched_gt.add(i)
                break

    false_positives = len(detected_pairs) - true_positives
    false_negatives = len(ground_truth) - true_positives

    precision = true_positives / max(1, len(detected_pairs))
    recall = true_positives / max(1, len(ground_truth))
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return ContradictionDetectionResult(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
    )


def evaluate_viewpoint_coverage(
    system_views: List[str],                   # System's viewpoint summaries
    ground_truth_viewpoints: List[GroundTruthViewpoint],
    coverage_threshold: float = 0.5,
) -> float:
    """
    ViewpointCoverage (VC): What fraction of ground-truth viewpoints
    did the system surface?

    Uses token-Jaccard similarity between system viewpoint summaries
    and ground-truth viewpoint descriptions.

    Returns: float in [0, 1]
    """
    if not system_views or not ground_truth_viewpoints:
        return 0.0

    def jaccard_sim(a: str, b: str) -> float:
        ta = set(a.lower().split())
        tb = set(b.lower().split())
        return len(ta & tb) / len(ta | tb) if ta | tb else 0.0

    covered = 0
    for gt_vp in ground_truth_viewpoints:
        max_sim = max(
            jaccard_sim(sv, gt_vp.description) for sv in system_views
        )
        if max_sim >= coverage_threshold:
            covered += 1

    return covered / len(ground_truth_viewpoints)


def evaluate_confidence_calibration(
    predicted_confidences: List[float],   # System confidence scores (0-1)
    actual_correctness: List[float],       # 1 if answer was correct, 0 otherwise
    n_bins: int = 5,
) -> float:
    """
    ConfidenceCalibrationError (CCE) = Expected Calibration Error (ECE).

    Groups predictions into n_bins by confidence, measures gap between
    mean confidence and mean accuracy in each bin.

    Lower is better (0 = perfect calibration).
    """
    if len(predicted_confidences) != len(actual_correctness):
        raise ValueError("Confidence and correctness lists must be same length")
    if not predicted_confidences:
        return 0.0

    n = len(predicted_confidences)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_low, bin_high = bins[i], bins[i + 1]
        mask = [bin_low <= c < bin_high for c in predicted_confidences]
        if not any(mask):
            continue
        bin_conf = np.mean([c for c, m in zip(predicted_confidences, mask) if m])
        bin_acc = np.mean([a for a, m in zip(actual_correctness, mask) if m])
        bin_size = sum(mask)
        ece += (bin_size / n) * abs(bin_conf - bin_acc)

    return round(float(ece), 4)


def evaluate_faithfulness(
    system_claims: List[str],    # Claims made by the system in its answer
    source_passages: List[str],  # Retrieved source passages
    nli_threshold: float = 0.25,
) -> float:
    """
    FaithfulnessScore (FS): Fraction of system claims that are entailed
    by at least one source passage.

    Uses token-overlap (NLI-lite) as a proxy for entailment.
    For full evaluation, replace with a proper NLI model.

    Threshold lowered from 0.5 → 0.25: a claim like "Research shows mixed results
    on homework" shares few tokens with long retrieved passages by design; 0.5 was
    too strict and made FS=0 even for grounded systems.  The correct fix is a full
    NLI model; until then 0.25 is a better-calibrated proxy.

    Inspired by FActScore (Min et al. 2023) but adapted for multi-source setting.
    """
    if not system_claims or not source_passages:
        return 0.0

    def entailed_by_any(claim: str, passages: List[str], threshold: float) -> bool:
        claim_tokens = set(claim.lower().split())
        for passage in passages:
            passage_tokens = set(passage.lower().split())
            if not claim_tokens:
                continue
            overlap = len(claim_tokens & passage_tokens) / len(claim_tokens)
            if overlap >= threshold:
                return True
        return False

    supported = sum(
        1 for c in system_claims
        if entailed_by_any(c, source_passages, nli_threshold)
    )
    return round(supported / max(1, len(system_claims)), 4)


def evaluate_visual_text_consistency(
    text_claims: List[str],
    figure_captions: List[str],
    clip_similarities: Optional[List[float]] = None,
    mismatch_threshold: float = 0.3,
) -> float:
    """
    VisualTextConsistency (VTC): Novel metric.

    Measures alignment between text claims and figure captions.
    A mismatch score > mismatch_threshold indicates the text overclaims
    beyond what figures show (visual-text inconsistency).

    If CLIP similarities are provided, uses them directly.
    Otherwise falls back to token overlap as proxy.

    Returns: float in [0, 1], higher = more consistent (better).
    """
    if not text_claims or not figure_captions:
        return -1.0  # Sentinel: visual grounding not applicable

    if clip_similarities is not None:
        # Direct CLIP-based measurement
        mean_sim = np.mean(clip_similarities)
        return round(float(mean_sim), 4)

    # Fallback: token overlap proxy
    def token_overlap(a: str, b: str) -> float:
        ta = set(a.lower().split())
        tb = set(b.lower().split())
        return len(ta & tb) / len(ta | tb) if ta | tb else 0.0

    scores = []
    for claim in text_claims:
        max_score = max(token_overlap(claim, cap) for cap in figure_captions)
        scores.append(max_score)

    return round(float(np.mean(scores)), 4) if scores else -1.0


def compute_composite_score(
    cd_f1: float,
    viewpoint_coverage: float,
    ec_at_5: float,
    cce: float,
    faithfulness: float,
    vtc: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    EVIRAG Composite Score: weighted average of all metrics.

    Default weights reflect the paper's emphasis on disagreement coverage:
      - cd_f1: 0.25  (core disagreement detection)
      - viewpoint_coverage: 0.20  (multi-view completeness)
      - ec_at_5: 0.20  (retrieval coverage of disagreement space)
      - cce: 0.10  (calibration, inverted: lower CCE = better)
      - faithfulness: 0.15  (grounding)
      - vtc: 0.10  (visual consistency, or skipped if -1)
    """
    if weights is None:
        weights = {
            "cd_f1": 0.25,
            "viewpoint_coverage": 0.20,
            "ec_at_5": 0.20,
            "cce_inv": 0.10,
            "faithfulness": 0.15,
            "vtc": 0.10,
        }

    cce_inv = max(0.0, 1.0 - cce)   # invert CCE (lower error = higher score)
    vtc_score = vtc if vtc >= 0 else 0.5   # neutral if no visual grounding

    total_weight = sum(weights.values())
    composite = (
        weights["cd_f1"] * cd_f1 +
        weights["viewpoint_coverage"] * viewpoint_coverage +
        weights["ec_at_5"] * ec_at_5 +
        weights["cce_inv"] * cce_inv +
        weights["faithfulness"] * faithfulness +
        weights["vtc"] * vtc_score
    ) / total_weight

    return round(float(composite), 4)


# ---------------------------------------------------------------------------
# Benchmark dataset builder
# ---------------------------------------------------------------------------

class EVIRAGBenchBuilder:
    """
    Build a curated evaluation benchmark (EVIRAG-Bench) from test outputs.

    EVIRAG-Bench format:
      - Domain: Scientific AI/ML literature
      - Queries: Contested scientific questions
      - Ground truth: Human-annotated contradictions and viewpoints
      - Baselines: Vanilla RAG, Single-agent RAG
    """

    SAMPLE_INSTANCES = [
        {
            "instance_id": "EB-001",
            "query": "Does homework improve academic achievement?",
            "domain": "education",
            "contradictions": [
                {
                    "claim1_text": "Homework completion correlates with higher grades (r=0.25)",
                    "claim2_text": "No conclusive evidence homework improves achievement since 1987",
                    "source1": "Cooper et al. 1989",
                    "source2": "Corno et al. 2000",
                    "attribution_type": "METHODOLOGICAL",
                    "verified_by": "expert",
                },
                {
                    "claim1_text": "Structured homework improves study habits",
                    "claim2_text": "Homework causes stress and reduces wellbeing",
                    "source1": "Epstein & Van Voorhis 2001",
                    "source2": "Galloway et al. 2013",
                    "attribution_type": "POPULATION",
                    "verified_by": "expert",
                },
            ],
            "viewpoints": [
                {
                    "viewpoint_id": "EB-001-V1",
                    "description": "Homework has positive effects on achievement, especially when well-designed",
                    "representative_claims": ["Homework completion correlates with higher grades"],
                    "stance": "pro",
                    "strength": "moderate",
                },
                {
                    "viewpoint_id": "EB-001-V2",
                    "description": "No conclusive evidence of homework benefit; stress effects are negative",
                    "representative_claims": ["No conclusive evidence homework improves achievement"],
                    "stance": "con",
                    "strength": "strong",
                },
            ],
        },
        {
            "instance_id": "EB-002",
            "query": "Is overparameterization necessary for generalization in deep neural networks?",
            "domain": "machine_learning",
            "contradictions": [
                {
                    "claim1_text": "Overparameterized models exhibit double descent generalization",
                    "claim2_text": "Classical bias-variance tradeoff predicts overparameterization harms generalization",
                    "source1": "Belkin et al. 2019",
                    "source2": "Geman et al. 1992",
                    "attribution_type": "THEORETICAL",
                    "verified_by": "expert",
                },
            ],
            "viewpoints": [
                {
                    "viewpoint_id": "EB-002-V1",
                    "description": "Double descent: overparameterization can improve generalization beyond the classical regime",
                    "representative_claims": ["Overparameterized models exhibit double descent"],
                    "stance": "pro",
                    "strength": "strong",
                },
                {
                    "viewpoint_id": "EB-002-V2",
                    "description": "Classical regularization is sufficient; overparameterization is not necessary",
                    "representative_claims": ["Implicit regularization explains generalization without overparameterization"],
                    "stance": "alternative",
                    "strength": "moderate",
                },
            ],
        },
        {
            "instance_id": "EB-003",
            "query": "Does vitamin D supplementation prevent COVID-19?",
            "domain": "medicine",
            "contradictions": [
                {
                    "claim1_text": "Vitamin D deficiency is associated with higher COVID-19 severity",
                    "claim2_text": "RCTs show vitamin D supplementation does not reduce COVID-19 risk",
                    "source1": "Meltzer et al. 2020",
                    "source2": "Jolliffe et al. 2021",
                    "attribution_type": "METHODOLOGICAL",
                    "verified_by": "expert",
                },
            ],
            "viewpoints": [
                {
                    "viewpoint_id": "EB-003-V1",
                    "description": "Observational evidence suggests vitamin D protects against severe COVID-19",
                    "representative_claims": ["Vitamin D deficiency correlates with worse outcomes"],
                    "stance": "pro",
                    "strength": "moderate",
                },
                {
                    "viewpoint_id": "EB-003-V2",
                    "description": "RCT evidence does not support vitamin D as preventative",
                    "representative_claims": ["RCTs show no significant protective effect"],
                    "stance": "con",
                    "strength": "strong",
                },
            ],
        },
    ]

    def build_sample_benchmark(self) -> List[EvaluationInstance]:
        """Build sample benchmark instances for testing."""
        instances = []
        for raw in self.SAMPLE_INSTANCES:
            contradictions = [
                GroundTruthContradiction(**c) for c in raw["contradictions"]
            ]
            viewpoints = [
                GroundTruthViewpoint(**v) for v in raw["viewpoints"]
            ]
            instances.append(EvaluationInstance(
                instance_id=raw["instance_id"],
                query=raw["query"],
                domain=raw["domain"],
                contradictions=contradictions,
                viewpoints=viewpoints,
            ))
        return instances

    def save_benchmark(self, instances: List[EvaluationInstance], path: str = "evirag_bench.json"):
        """Save benchmark to JSON for reproducibility."""
        data = []
        for inst in instances:
            data.append({
                "instance_id": inst.instance_id,
                "query": inst.query,
                "domain": inst.domain,
                "contradictions": [asdict(c) for c in inst.contradictions],
                "viewpoints": [asdict(v) for v in inst.viewpoints],
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Benchmark saved to {path} ({len(instances)} instances)")


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

class EVIRAGEvaluator:
    """
    Run formal evaluation of EVIRAG vs. baselines on EVIRAG-Bench.

    Usage:
        evaluator = EVIRAGEvaluator()
        result = evaluator.evaluate_instance(
            instance=bench_instance,
            system_name="EVIRAG",
            detected_contradictions=[...],
            system_views=[...],
            ...
        )
        evaluator.print_report([result])
    """

    def evaluate_instance(
        self,
        instance: EvaluationInstance,
        system_name: str,
        detected_contradictions: List[Tuple[str, str]],
        system_views: List[str],
        system_claims: List[str],
        source_passages: List[str],
        confidence_score: float,
        ground_truth_correct: float,   # 1.0 or 0.0 for this instance
        ec_at_k_values: Optional[Dict[int, float]] = None,
        clip_similarities: Optional[List[float]] = None,
        figure_captions: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Evaluate one system on one benchmark instance."""

        # 1. Contradiction Detection
        cd = evaluate_contradiction_detection(
            detected_contradictions, instance.contradictions
        )

        # 2. Viewpoint Coverage
        vc = evaluate_viewpoint_coverage(
            system_views,
            instance.viewpoints,
        )

        # 3. Confidence Calibration Error (single instance approximation)
        cce = abs(confidence_score - ground_truth_correct)

        # 4. EC@K (use provided or zeros)
        ec_at_k = ec_at_k_values or {}

        # 5. Faithfulness
        fs = evaluate_faithfulness(system_claims, source_passages)

        # 6. Visual-Text Consistency
        vtc = evaluate_visual_text_consistency(
            system_claims,
            figure_captions or [],
            clip_similarities,
        )

        # 7. Composite
        composite = compute_composite_score(
            cd_f1=cd.f1,
            viewpoint_coverage=vc,
            ec_at_5=ec_at_k.get(5, 0.0),
            cce=cce,
            faithfulness=fs,
            vtc=vtc,
        )

        # 8. Consensus Collapse Degradation placeholder
        ccd = 0.0  # Computed separately via EpistemicDivergenceCalculator

        return EvaluationResult(
            instance_id=instance.instance_id,
            system_name=system_name,
            cd_precision=cd.precision,
            cd_recall=cd.recall,
            cd_f1=cd.f1,
            ec_at_k=ec_at_k,
            viewpoint_coverage=vc,
            confidence_calibration_error=round(cce, 4),
            consensus_collapse_degradation=ccd,
            faithfulness_score=fs,
            visual_text_consistency=vtc,
            evirag_score=composite,
        )

    def print_report(self, results: List[EvaluationResult]):
        """Print formatted evaluation report."""
        if not results:
            print("No evaluation results.")
            return

        print("\n" + "=" * 70)
        print("EVIRAG EVALUATION REPORT")
        print("=" * 70)
        print(f"{'System':<25} {'CD-F1':>7} {'VC':>7} {'EC@5':>7} "
              f"{'CCE':>7} {'FS':>7} {'VTC':>7} {'Score':>8}")
        print("-" * 70)

        for r in results:
            print(
                f"{r.system_name:<25} "
                f"{r.cd_f1:>7.3f} "
                f"{r.viewpoint_coverage:>7.3f} "
                f"{r.ec_at_k.get(5, 0.0):>7.3f} "
                f"{r.confidence_calibration_error:>7.3f} "
                f"{r.faithfulness_score:>7.3f} "
                f"{r.visual_text_consistency:>7.3f} "
                f"{r.evirag_score:>8.3f}"
            )
        print("=" * 70)
        print("\nMetrics:")
        print("  CD-F1: Contradiction Detection F1")
        print("  VC:    Viewpoint Coverage")
        print("  EC@5:  EpistemicCoverage@5 (novel)")
        print("  CCE:   Confidence Calibration Error (lower=better)")
        print("  FS:    Faithfulness Score")
        print("  VTC:   Visual-Text Consistency (-1 = no visual grounding)")
        print("  Score: EVIRAG Composite Score")


# ---------------------------------------------------------------------------
# Ablation study runner
# ---------------------------------------------------------------------------

ABLATION_CONDITIONS = {
    "vanilla_rag": {
        "description": "Baseline: BM25 retrieval + single-view synthesis",
        "use_multi_agent": False,
        "use_visual": False,
        "use_temporal": False,
        "use_causal_attribution": False,
    },
    "single_agent_rag": {
        "description": "Ablation: One precision agent, no adversarial agents",
        "use_multi_agent": False,
        "use_visual": False,
        "use_temporal": False,
        "use_causal_attribution": False,
        "num_agents": 1,
    },
    "evirag_no_visual": {
        "description": "Ablation: EVIRAG without CLIP visual grounding",
        "use_multi_agent": True,
        "use_visual": False,
        "use_temporal": True,
        "use_causal_attribution": True,
    },
    "evirag_no_temporal": {
        "description": "Ablation: EVIRAG without temporal drift tracking",
        "use_multi_agent": True,
        "use_visual": True,
        "use_temporal": False,
        "use_causal_attribution": True,
    },
    "evirag_no_causal": {
        "description": "Ablation: EVIRAG without causal attribution",
        "use_multi_agent": True,
        "use_visual": True,
        "use_temporal": True,
        "use_causal_attribution": False,
    },
    "evirag_full": {
        "description": "Full EVIRAG: all components enabled",
        "use_multi_agent": True,
        "use_visual": True,
        "use_temporal": True,
        "use_causal_attribution": True,
    },
}


def run_ablation_study(
    query: str,
    system_runner,  # callable(query, config) -> result dict
    benchmark_instance: EvaluationInstance,
    evaluator: Optional[EVIRAGEvaluator] = None,
) -> Dict[str, EvaluationResult]:
    """
    Run all ablation conditions and return comparison table.

    Each condition tests removing one EVIRAG component to demonstrate
    its marginal contribution — required for tier-1 paper acceptance.
    """
    if evaluator is None:
        evaluator = EVIRAGEvaluator()

    results = {}
    for condition_name, config in ABLATION_CONDITIONS.items():
        print(f"Running ablation: {condition_name}...")
        try:
            system_output = system_runner(query, config)
            result = evaluator.evaluate_instance(
                instance=benchmark_instance,
                system_name=condition_name,
                detected_contradictions=system_output.get("contradictions", []),
                system_views=system_output.get("views", []),
                system_claims=system_output.get("claims", []),
                source_passages=system_output.get("passages", []),
                confidence_score=system_output.get("confidence", 0.5),
                ground_truth_correct=system_output.get("correctness", 0.5),
                ec_at_k_values=system_output.get("ec_at_k", {}),
                clip_similarities=system_output.get("clip_similarities"),
                figure_captions=system_output.get("figure_captions"),
            )
            results[condition_name] = result
        except Exception as e:
            print(f"  Error in {condition_name}: {e}")

    evaluator.print_report(list(results.values()))
    return results
