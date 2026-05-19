#!/usr/bin/env python3
"""
Visual Grounding Test - Targeted Query for Step 5
Tests with query specifically designed to leverage figures/tables
"""

import sys, time

print("🔬 VISUAL GROUNDING TEST - Step 5 Demonstration")
print("="*80)
print("\n📝 Query: How do GraphRAG and traditional RAG compare in terms of correctness?")
print("\n🎯 Expected: Claims about GraphRAG vs RAG performance should align with")
print("   figures showing confusion matrices, performance metrics, and comparisons")
print("\n📊 This query targets papers with rich visual content (confusion matrices,")
print("   performance charts) to demonstrate visual grounding working effectively.\n")

from evirag_system import EVIRAGSystem, EVIRAGConfig
from config import GRAPH_CONFIG

print(f"📊 Config: {GRAPH_CONFIG['max_pairs']} pairs, {GRAPH_CONFIG['min_edge_confidence']} min confidence\n")

try:
    config = EVIRAGConfig(
        mode="evirag",
        backend="local",
        enabled_agents=["precision", "recall", "skeptic"],
        use_visual_grounding=True,  # CRITICAL: Visual grounding enabled
        depth_vs_speed="fast"
    )
    
    system = EVIRAGSystem(config)
    system.initialize_corpus()
    
    start = time.time()
    # Query designed to leverage figures with performance comparisons
    result = system.query("How do GraphRAG and traditional RAG compare in terms of correctness and accuracy?")
    elapsed = time.time() - start
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")
    print(f"\n🎉 Visual grounding test complete!")
    print(f"\n📝 All results displayed above by the EVIRAG system.")
    
except Exception as e:
    print(f"\n❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
