#!/usr/bin/env python3
"""
COMPLETE TEST WITH ALL 4 AGENTS ENABLED
This is the definitive test with Skeptic + Counterfactual agents
"""

import sys, time

# Test query
QUERY = "Does retrieval augmentation improve or hurt model performance?"

print("🎯 COMPLETE TEST - ALL 4 AGENTS ENABLED")
print("="*80)
print(f"\n📝 Query: {QUERY}")
print("\n✅ Configuration:")
print("   - Backend: Local (Llama3)")
print("   - Agents: ALL 4 (Precision + Recall + Skeptic + Counterfactual)")
print("   - Confidence: 0.4 (updated from 0.35)")
print("   - Pairs: 300")
print("   - Corpus: Full 8 papers (~214 chunks)")
print("\n💡 CRITICAL CHANGE:")
print("   - Skeptic agent: ENABLED (finds limitations/challenges)")
print("   - Counterfactual agent: ENABLED (finds alternatives)")
print("   - These will extract CRITICAL claims that contradict positive claims!")
print("="*80)

from evirag_system import EVIRAGSystem, EVIRAGConfig

try:
    config = EVIRAGConfig(
        mode="evirag",
        backend="local",
        enabled_agents=["precision", "recall", "skeptic", "counterfactual"],  # ALL 4 AGENTS!
        use_visual_grounding=False,
        depth_vs_speed="fast"
    )
    
    print("\nInitializing EVIRAG System with ALL agents...\n")
    system = EVIRAGSystem(config)
    system.initialize_corpus(rebuild=True)  # Force rebuild
    
    start = time.time()
    result = system.query(QUERY)
    elapsed = time.time() - start
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")
    print(f"\n🎉 Full test with all 4 agents complete!")
    print(f"\nExpected results:")
    print(f"  - Precision/Recall: Positive claims (RAG improves...)")
    print(f"  - Skeptic/Counterfactual: Critical claims (RAG has challenges...)")
    print(f"  - NLI: Should detect contradictions between positive vs critical!")
    
except Exception as e:
    print(f"\n❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
