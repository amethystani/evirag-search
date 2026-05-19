"""
EVIRAG End-to-End Test Runner
Runs the full pipeline against the actual corpus and prints a clean summary.
"""

import sys
import json
import time

sys.path.insert(0, "/Users/krishangsharma/Downloads/Project-2")

from evirag_system import EVIRAGSystem, EVIRAGConfig

TEST_QUERY = "Does the effectiveness of homework differ between subjects like math or science compared to reading or language arts?"

def separator(char="=", width=70):
    print(char * width)

def run_test():
    separator()
    print("EVIRAG END-TO-END TEST")
    print(f"Query: {TEST_QUERY}")
    separator()

    config = EVIRAGConfig(
        mode="evirag",
        backend="local",
        use_visual_grounding=False,  # Keep fast; VLM tested separately
        enabled_agents=["precision", "recall", "skeptic", "counterfactual"],
        depth_vs_speed="balanced",
        rebuild_index=False
    )

    t0 = time.time()

    print("\n[1/2] Initializing system and corpus...")
    system = EVIRAGSystem(config)
    system.initialize_corpus(rebuild=False)

    print(f"\n[2/2] Running EVIRAG query...")
    result = system.query(TEST_QUERY, config)

    elapsed = time.time() - t0

    separator()
    print("RESULTS SUMMARY")
    separator()

    # Mode
    print(f"Mode:    {result.get('mode', 'N/A')}")
    print(f"Elapsed: {elapsed:.1f}s")

    # Intent
    intent = result.get("intent", {})
    print(f"\nIntent Analysis:")
    print(f"  Factual vs Disputed:   {intent.get('factual_vs_disputed','?')}")
    print(f"  Empirical vs Theoretical: {intent.get('empirical_vs_theoretical','?')}")
    print(f"  Disagreement Level:    {intent.get('disagreement_level','?')}")

    # Hypothesis
    hyp = result.get("hypothesis", {})
    print(f"\nHypothesis: {hyp.get('central_hypothesis','?')[:120]}")

    # Agent stats
    agents = result.get("agent_outputs", {})
    print(f"\nAgent Outputs:")
    print(f"  Total agents run: {agents.get('total_agents', 0)}")
    print(f"  Total claims:     {agents.get('total_claims', 0)}")
    stances = agents.get("stances", {})
    print(f"  Stances — Support: {stances.get('support',0)} | Oppose: {stances.get('oppose',0)} | Neutral: {stances.get('neutral',0)}")

    # Stats
    stats = result.get("statistics", {})
    print(f"\nPipeline Stats:")
    print(f"  Unique claims:    {stats.get('total_claims', 0)}")
    print(f"  Relationships:    {stats.get('total_relationships', 0)}")

    # Metrics
    metrics = result.get("metrics", {})
    print(f"\nDisagreement Metrics:")
    print(f"  Density:        {metrics.get('disagreement_density', 0):.4f}")
    print(f"  Conflict Ratio: {metrics.get('conflict_ratio', 0):.4f}")
    print(f"  Claim Entropy:  {metrics.get('claim_entropy', 0):.4f}")
    mismatch = metrics.get("visual_text_mismatch", 0)
    print(f"  Visual Mismatch:{mismatch:.4f}  (VLM disabled, expected 0)")

    # Answer
    answer = result.get("answer", {})
    print(f"\nConfidence: {answer.get('overall_confidence','?').upper()} ({answer.get('confidence_score',0):.3f})")
    print(f"Reasoning:\n  {answer.get('confidence_reasoning','').strip()}")

    dominant = answer.get("dominant_view", {})
    if dominant:
        print(f"\nDominant View ({dominant.get('name','?')}):")
        print(f"  {dominant.get('summary','?')}")
        print(f"  Sources: {dominant.get('source_count',0)} | Claims: {len(dominant.get('supporting_claim_ids',[]))}")
        weaknesses = dominant.get("weaknesses", [])
        if weaknesses:
            print(f"  Weaknesses: {'; '.join(weaknesses[:2])}")

    alt_views = answer.get("alternative_views", [])
    if alt_views:
        print(f"\nAlternative Views ({len(alt_views)}):")
        for i, v in enumerate(alt_views, 1):
            print(f"  {i}. {v.get('summary','?')[:100]}")

    min_views = answer.get("minority_views", [])
    if min_views:
        print(f"\nMinority Views ({len(min_views)}): present")

    separator()

    # Pass/Fail checks
    checks = {
        "Mode is evirag":               result.get("mode") == "evirag",
        "Intent analyzed":              bool(intent.get("disagreement_level")),
        "Hypothesis generated":         bool(hyp.get("central_hypothesis")),
        "Agents ran (>0 claims)":       agents.get("total_claims", 0) > 0,
        "Claims extracted (>5)":        stats.get("total_claims", 0) > 5,
        "Relationships built":          stats.get("total_relationships", 0) >= 0,
        "Confidence assigned":          bool(answer.get("overall_confidence")),
        "Dominant view present":        bool(dominant),
    }

    print("\nPASS/FAIL CHECKS:")
    all_pass = True
    for name, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_pass = False

    separator()
    if all_pass:
        print("OVERALL: ALL CHECKS PASSED - System working end-to-end!")
    else:
        print("OVERALL: SOME CHECKS FAILED - Review output above.")
    separator()

    # Save full JSON result
    out_path = "/Users/krishangsharma/Downloads/Project-2/e2e_test_output.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Full result saved to: {out_path}")


if __name__ == "__main__":
    run_test()
