#!/usr/bin/env python3
"""
Micro-Disagreement Test - Testing Specific Comparative Topic
Testing if papers disagree on: large general models vs small specialized models for medical domains
"""

import sys, time

# Highly specific comparative query targeting potential disagreement
COMPARATIVE_QUERY = """
For clinical neurology applications, which approach is superior:
(A) Large general-purpose language models (10B+ parameters) adapted via few-shot prompting, OR  
(B) Smaller domain-specific models (1-7B parameters) fine-tuned on curated medical/neurology data?

Consider: accuracy, safety, interpretability, data efficiency, and clinical deployability.
"""

print("🎯 MICRO-DISAGREEMENT TEST")
print("="*80)
print("\n📝 Testing Specific Comparative Topic:")
print("   Large General Models vs Small Specialized Models for Medical Domains")
print(f"\n💡 Query:\n{COMPARATIVE_QUERY}")
print("\n✅ Configuration:")
print("   - Corpus: All 8 papers")  
print("   - Agents: ALL 4 (Precision + Recall + Skeptic + Counterfactual)")
print("   - Confidence: 0.45")
print("   - Backend: Local")
print("\n🎯 Goal: Find if ANY papers take different positions on this trade-off")
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
    
    print("\nInitializing EVIRAG System...\n")
    system = EVIRAGSystem(config)
    system.initialize_corpus(rebuild=False)  # Use existing index
    
    start = time.time()
    result = system.query(COMPARATIVE_QUERY)
    elapsed = time.time() - start
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")
    
except Exception as e:
    print(f"\n❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
