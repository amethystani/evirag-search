#!/usr/bin/env python3
"""
Homework Contradiction Test
Testing with papers that have opposing views on homework effectiveness
HW1: PRO-homework papers
HW2: ANTI/critical papers on homework
"""

import sys, time

# Targeted query about homework effectiveness
HOMEWORK_QUERY = """
Does assigning homework to students improve their academic performance and learning outcomes, 
or is homework ineffective or even harmful to student achievement and wellbeing?
"""

print("🎯 HOMEWORK CONTRADICTION TEST")
print("="*80)
print("\n📚 Corpus: 4 homework research papers")
print("   HW1 Folder: 2 papers supporting homework effectiveness")
print("   HW2 Folder: 2 papers critical of homework")
print(f"\n📝 Query: {HOMEWORK_QUERY.strip()}")
print("\n✅ Configuration:")
print("   - Backend: Local (Llama3)")
print("   - Agents: ALL 4 (Precision + Recall + Skeptic + Counterfactual)")
print("   - Confidence: 0.45")
print("   - Pairs: 300")
print("\n💡 Expected: CONTRADICTIONS between PRO and ANTI homework claims!")
print("="*80)

from evirag_system import EVIRAGSystem, EVIRAGConfig

try:
    config = EVIRAGConfig(
        mode="evirag",
        backend="local",
        enabled_agents=["precision", "recall", "skeptic", "counterfactual"],
        use_visual_grounding=False,
        depth_vs_speed="fast"
    )
    
    print("\nInitializing EVIRAG System with homework papers...\n")
    system = EVIRAGSystem(config)
    system.initialize_corpus(rebuild=True)  # Force rebuild with homework papers
    
    start = time.time()
    result = system.query(HOMEWORK_QUERY)
    elapsed = time.time() - start
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")
    print(f"\n🎯 Homework contradiction test complete!")
    print(f"\nIf contradictions detected:")
    print(f"  → System successfully found disagreement between PRO and ANTI papers!")
    print(f"  → Contradiction detection WORKS! ✅")
    print(f"\nIf only entailments:")
    print(f"  → Need to investigate why papers didn't contradict")
    
except Exception as e:
    print(f"\n❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
