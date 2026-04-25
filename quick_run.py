"""Quick monitored pipeline run — 2 agents, no visual grounding."""
import sys, time

print("Initializing...")
from evirag_system import EVIRAGSystem, EVIRAGConfig

config = EVIRAGConfig(
    mode="evirag",
    use_visual_grounding=False,
    use_epistemic_divergence=True,
    use_causal_attribution=False,  # skip LLM attribution for speed
    use_temporal_tracking=True,
    depth_vs_speed="balanced",
    enabled_agents=["precision", "skeptic"],  # 2 agents only
    rebuild_index=False,
)
system = EVIRAGSystem(config)
system.initialize_corpus()

print("\nRunning query...")
t0 = time.time()
result = system.query("Does homework improve academic achievement?")
elapsed = time.time() - t0

print(f"\n{'='*60}")
print(f"Done in {elapsed:.0f}s")
ans = result["answer"]
print(f"Confidence      : {ans['overall_confidence']} ({ans['confidence_score']:.2f})")
dom = ans["dominant_view"]
print(f"Dominant view   : {dom['summary'][:140]}")
print(f"Alt views       : {len(ans['alternative_views'])}")
stats = result["statistics"]
print(f"Claims / rels   : {stats['total_claims']} / {stats['total_relationships']}")
ed = result.get("epistemic_divergence")
if ed:
    print(f"ED score        : {ed['ed_score']}  class: {ed['controversy_class']}")
    print(f"Collapse score  : {ed['consensus_collapse_score']}")
td = result.get("temporal_drift")
if td:
    print(f"Temporal traj   : {td['trajectory']} (conf {td['trajectory_confidence']:.2f})")
