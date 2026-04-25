#!/usr/bin/env python3
"""
Bizarre Absurd Query Test - Totally Contradicted by Corpus
Based on corpus: RAG papers, LLM papers, medical/neurology AI papers

Creates a query that is COMPLETELY WRONG about what the papers discuss
"""

import sys, time

# Absurd query that contradicts everything in the corpus
ABSURD_QUERY = "Should we abandon artificial intelligence and language models completely in favor of manual typewriters for medical research?"

print("💥 ABSURD CONTRADICTORY QUERY TEST")
print("="*80)
print(f"\n📝 Query (totally absurd and contradicted by corpus):")
print(f"   '{ABSURD_QUERY}'")
print("\n🎯 Why This Should Produce Contradictions:")
print("   - Corpus is ABOUT AI/LLMs (papers wouldn't exist if we should abandon them!)")
print("   - Papers demonstrate AI benefits in medicine/research")
print("   - Query suggests abandoning AI → Claims will support AI")
print("   - Direct contradiction between query premise and paper claims")
print("\n💥 Expected: Claims contradict the anti-AI premise")
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
    result = system.query(ABSURD_QUERY)
    elapsed = time.time() - start
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")
    print(f"\n🎉 Absurd query test complete!")
    print(f"\n📝 If system found contradictions, it proves claim extraction + NLI work correctly.")
    print(f"📝 All claims should contradict 'abandon AI' premise.")
    
except Exception as e:
    print(f"\n❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
