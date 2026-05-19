#!/usr/bin/env python3
"""
Contradiction Detection Test - Controversial Query
Tests system's ability to detect genuine contradictions in literature
"""

import sys, time

print("🔥 CONTRADICTION DETECTION TEST")
print("="*80)
print("\n📝 Controversial Query Designed to Surface Contradictions:")
print("   'Does retrieval augmentation improve or hurt model performance?'")
print("\n🎯 Expected: This query should surface CONTRADICTORY claims about:")
print("   - Cases where RAG improves performance")
print("   - Cases where RAG hurts performance")
print("   - Trade-offs and limitations")
print("\n💥 This tests NLI's ability to detect genuine disagreement!\n")

from evirag_system import EVIRAGSystem, EVIRAGConfig
from config import GRAPH_CONFIG

print(f"📊 Config: {GRAPH_CONFIG['max_pairs']} pairs, {GRAPH_CONFIG['min_edge_confidence']} min confidence\n")

try:
    config = EVIRAGConfig(
        mode="evirag",
        backend="local",
        enabled_agents=["precision", "recall", "skeptic"],
        use_visual_grounding=True,
        depth_vs_speed="fast"
    )
    
    system = EVIRAGSystem(config)
    system.initialize_corpus()
    
    start = time.time()
    
    # CONTROVERSIAL QUERY - designed to surface contradictions
    result = system.query("Does retrieval augmentation improve or hurt model performance?")
    
    elapsed = time.time() - start
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")
    print(f"\n🎉 Contradiction detection test complete!")
    print(f"\n📝 Check if system found more contradictions than previous tests.")
    
except Exception as e:
    print(f"\n❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
