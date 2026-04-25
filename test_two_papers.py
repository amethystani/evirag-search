#!/usr/bin/env python3
"""
Two-Paper Targeted Contradiction Test
Using only 2 papers with potentially opposing views on domain specificity
"""

import sys, time

# Targeted query designed for 2 papers:
# Paper 1: General AI/RAG (broad applicability)
# Paper 2: Neurology-specific (domain challenges/limitations)
TARGETED_QUERY = "Are general-purpose language models equally effective across all domains, or do specialized medical domains like neurology require significant domain adaptation and face unique challenges?"

print("🎯 TWO-PAPER TARGETED CONTRADICTION TEST")
print("="*80)
print(f"\n📚 Corpus: ONLY 2 papers")
print("   1. General LLM/RAG paper")
print("   2. Neurology-specific LLM paper")
print(f"\n📝 Targeted Query:")
print(f"   '{TARGETED_QUERY}'")
print("\n💡 Strategy:")
print("   - General paper likely says: LLMs work broadly")
print("   - Neurology paper likely says: Medical domains need adaptation")
print("   - These perspectives should contradict on 'domain-specificity'")
print(f"\n📊 Config: 300 pairs (local), 0.35 min confidence")
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
    
    print("\nInitializing EVIRAG System with 2-paper corpus...\n")
    system = EVIRAGSystem(config)
    system.initialize_corpus(rebuild=True)  # Force rebuild with 2 papers
    
    start = time.time()
    result = system.query(TARGETED_QUERY)
    elapsed = time.time() - start
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")
    print(f"\n🎯 Two-paper targeted test complete!")
    print(f"\nIf contradictions detected:")
    print(f"  → Papers disagree on domain-specificity of LLMs")
    print(f"  → System successfully found disagreement between papers!")
    print(f"\nIf only entailments:")
    print(f"  → Papers actually agree on this topic")
    print(f"  → May need different papers or different query angle")
    
except Exception as e:
    print(f"\n❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
