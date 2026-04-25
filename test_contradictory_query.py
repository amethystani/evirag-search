#!/usr/bin/env python3
"""
Contradictory Query Test - Borderline Nonsensical
Based on corpus papers: RAG, GraphRAG, LLMs, medical applications

Creates a query designed to force contradictions from the corpus
"""

import sys, time

# Query designed to be internally contradictory and pull opposing claims
CONTRADICTORY_QUERY = "Does using smaller models with less data produce better results than larger models with more data?"

print("🔥 CONTRADICTORY QUERY TEST - Borderline Nonsensical")
print("="*80)
print(f"\n📝 Query (designed to force contradictions):")
print(f"   '{CONTRADICTORY_QUERY}'")
print("\n🎯 Why This Should Work:")
print("   - Papers discuss both small vs large models")
print("   - Papers discuss data quantity effects")
print("   - Query asks if LESS is MORE (contradictory premise)")
print("   - Forces extraction of opposing comparative claims")
print("\n💥 Expected: Claims will contradict the premise in multiple ways")
print(f"\n📊 Config: 300 pairs (local), 0.35 min confidence\n")

from evirag_system import EVIRAGSystem, EVIRAGConfig

try:
    config = EVIRAGConfig(
        mode="evirag",
        backend="local",
        enabled_agents=["precision", "recall", "skeptic"],
        use_visual_grounding=True,
        depth_vs_speed="fast"
    )
    
    print("Initializing EVIRAG System (backend: local)...\n")
    system = EVIRAGSystem(config)
    system.initialize_corpus()
    
    start = time.time()
    result = system.query(CONTRADICTORY_QUERY)
    elapsed = time.time() - start
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")
    print(f"\n🎉 Contradictory query test complete!")
    print(f"\n📝 Check if system detected contradictions in opposing model/data claims.")
    
except Exception as e:
    print(f"\n❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
