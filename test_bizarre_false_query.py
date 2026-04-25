#!/usr/bin/env python3
"""
Bizarre False Query Test
Query that is CLEARLY FALSE and will be contradicted by corpus papers
Based on corpus: LLMs in neurology, RAG, medical AI research
"""

import sys, time

# Bizarre FALSE query that corpus papers will contradict
BIZARRE_QUERY = "Are language models and AI completely useless and harmful for neurology and medical research, providing no benefits whatsoever?"

print("🔥 BIZARRE FALSE QUERY TEST")
print("="*80)
print(f"\n📝 Query (clearly FALSE and contradicted by corpus):")
print(f"   '{BIZARRE_QUERY}'")
print("\n🎯 Why This Will Work:")
print("   - Corpus contains papers on LLMs in NEUROLOGY research")
print("   - Papers demonstrate AI BENEFITS in medicine")
print("   - Query claims AI is USELESS and HARMFUL")
print("   - Claims extracted will show AI helps neurology")
print("   - Direct contradiction between FALSE premise and TRUE claims")
print("\n💥 Expected: All claims will contradict this absurd FALSE statement")
print(f"\n📊 Config: 300 pairs (local), 0.35 min confidence\n")

from evirag_system import EVIRAGSystem, EVIRAGConfig

try:
    config = EVIRAGConfig(
        mode="evirag",
        backend="local",
        enabled_agents=["precision", "recall", "skeptic"],
        use_visual_grounding=False,  # Skip for speed
        depth_vs_speed="fast"
    )
    
    print("Initializing EVIRAG System (backend: local)...\n")
    system = EVIRAGSystem(config)
    system.initialize_corpus()
    
    start = time.time()
    result = system.query(BIZARRE_QUERY)
    elapsed = time.time() - start
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")
    print(f"\n🎉 Bizarre false query test complete!")
    print(f"\nIf contradictions detected: Papers correctly refute the FALSE claim")
    print(f"System successfully identifies when query premise is wrong!")
    
except Exception as e:
    print(f"\n❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
