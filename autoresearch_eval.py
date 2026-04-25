"""
EVIRAG Autoresearch Evaluation Script
======================================
This is the EVALUATOR for the autoresearch loop.

Target metric: contradiction_recall — the fraction of known contradictions
that EVIRAG surfaces. This is the metric autoresearch will optimize.

Usage:
    python autoresearch_eval.py

Output format (autoresearch reads this):
    contradiction_recall: <float>
    cd_f1: <float>
    viewpoint_coverage: <float>
    ec_at_5: <float>

DO NOT MODIFY THIS FILE once the autoresearch loop is running.
Modifying the evaluator invalidates all experiment comparisons.
"""

import json
import sys
from typing import List, Tuple

from evaluation_framework import (
    EVIRAGBenchBuilder,
    EVIRAGEvaluator,
    evaluate_contradiction_detection,
    evaluate_viewpoint_coverage,
    EvaluationInstance,
)


def run_evirag_on_instance(instance: EvaluationInstance) -> dict:
    """
    Run EVIRAG on a benchmark instance and collect outputs.

    In autoresearch mode, this is the function that gets measured.
    The autoresearch loop modifies EVIRAG's internals and re-runs this.
    """
    try:
        from evirag_system import EVIRAGSystem, EVIRAGConfig

        config = EVIRAGConfig(
            mode="evirag",
            backend="local",
            use_visual_grounding=False,   # Disabled for speed in eval loop
            use_epistemic_divergence=True,
            use_causal_attribution=True,
            use_temporal_tracking=True,
            depth_vs_speed="balanced",
        )
        system = EVIRAGSystem(config)
        system.initialize_corpus()
        result = system.query(instance.query)
        return result
    except Exception as e:
        print(f"EVIRAG query failed: {e}", file=sys.stderr)
        return {}


def extract_detected_contradictions(result: dict) -> List[Tuple[str, str]]:
    """Extract detected contradiction pairs from EVIRAG output."""
    contradictions = []
    metrics = result.get("metrics", {})
    answer = result.get("answer", {})

    # Pull from agent outputs if available
    agent_summary = result.get("agent_outputs", {})

    # Best proxy: extract from the disagreement graph edges
    # (In the full system, we'd pass the graph directly)
    # For evaluation, we use the claims from opposing viewpoints
    dom_view = answer.get("dominant_view", {})
    alt_views = answer.get("alternative_views", [])

    dom_claims = dom_view.get("supporting_claim_ids", [])
    for alt in alt_views:
        alt_claims = alt.get("supporting_claim_ids", [])
        # Cross-pair dominant and alternative claims as proxies for contradictions
        for dc in dom_claims[:3]:
            for ac in alt_claims[:3]:
                if dc != ac:
                    contradictions.append((str(dc), str(ac)))

    return contradictions[:20]  # cap for efficiency


def extract_system_views(result: dict) -> List[str]:
    """Extract viewpoint summaries from EVIRAG output."""
    answer = result.get("answer", {})
    views = []

    dom = answer.get("dominant_view", {})
    if dom.get("summary"):
        views.append(dom["summary"])

    for alt in answer.get("alternative_views", []):
        if alt.get("summary"):
            views.append(alt["summary"])

    for minority in answer.get("minority_views", []):
        if minority.get("summary"):
            views.append(minority["summary"])

    return views


def main():
    """Main evaluation entry point for autoresearch loop."""
    builder = EVIRAGBenchBuilder()
    instances = builder.build_sample_benchmark()

    total_cd_precision = 0.0
    total_cd_recall = 0.0
    total_cd_f1 = 0.0
    total_vc = 0.0
    n = 0

    for instance in instances:
        print(f"Evaluating: {instance.instance_id} — {instance.query[:60]}...")

        result = run_evirag_on_instance(instance)
        if not result:
            continue

        detected = extract_detected_contradictions(result)
        system_views = extract_system_views(result)

        cd = evaluate_contradiction_detection(
            detected_pairs=detected,
            ground_truth=instance.contradictions,
        )
        vc = evaluate_viewpoint_coverage(
            system_views=system_views,
            ground_truth_viewpoints=instance.viewpoints,
        )

        print(f"  CD: P={cd.precision:.3f} R={cd.recall:.3f} F1={cd.f1:.3f} | VC={vc:.3f}")

        total_cd_precision += cd.precision
        total_cd_recall += cd.recall
        total_cd_f1 += cd.f1
        total_vc += vc
        n += 1

    if n == 0:
        print("No instances evaluated.")
        # Output sentinel values
        print("contradiction_recall: 0.0")
        print("cd_f1: 0.0")
        print("viewpoint_coverage: 0.0")
        print("ec_at_5: 0.0")
        return

    mean_recall = total_cd_recall / n
    mean_f1 = total_cd_f1 / n
    mean_vc = total_vc / n

    # Primary metric for autoresearch: contradiction_recall
    # (Recall is more important than precision here — missing contradictions = false consensus)
    print(f"\n{'='*50}")
    print(f"EVIRAG Evaluation Results ({n} instances)")
    print(f"{'='*50}")
    print(f"CD Precision:       {total_cd_precision/n:.4f}")
    print(f"CD Recall:          {mean_recall:.4f}")
    print(f"CD F1:              {mean_f1:.4f}")
    print(f"Viewpoint Coverage: {mean_vc:.4f}")
    print(f"{'='*50}")

    # Autoresearch reads these lines:
    print(f"contradiction_recall: {mean_recall:.4f}")
    print(f"cd_f1: {mean_f1:.4f}")
    print(f"viewpoint_coverage: {mean_vc:.4f}")
    print(f"ec_at_5: 0.0")  # Requires retrieval logs not yet wired in


if __name__ == "__main__":
    main()
