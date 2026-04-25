#!/usr/bin/env python3
"""
Disagreement Test - Local Mode
Query: Which RAG architecture is more effective?
Expected: HIGH disagreement (competing papers)
"""

import sys, time

print("🚀 LOCAL Mode - Disagreement Detection Test")
print("="*80)
print("\n📝 Query: How does retrieval augmentation affect large language model performance?")
print("\n🎯 Expected: MEDIUM disagreement (Multiple papers discuss different aspects and effectiveness)\n")

from evirag_system import EVIRAGSystem, EVIRAGConfig
from config import GRAPH_CONFIG

print(f"📊 Config: {GRAPH_CONFIG['max_pairs']} pairs, {GRAPH_CONFIG['min_edge_confidence']} min confidence\n")

try:
    config = EVIRAGConfig(
        mode="evirag",
        backend="local",
        enabled_agents=["precision", "recall", "skeptic"],
        use_visual_grounding=True,  # Enable visual grounding to test Step 5
        depth_vs_speed="fast"
    )
    
    system = EVIRAGSystem(config)
    system.initialize_corpus()
    
    start = time.time()
    result = system.query("Which RAG architecture is more effective: traditional RAG or GraphRAG?")
    elapsed = time.time() - start
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")
    
    # Detailed results
    if 'claims' in result:
        print(f"\n📝 Claims: {len(result['claims'])}")
    
    if 'relationships' in result:
        rels = result['relationships']
        print(f"🔗 Relationships: {len(rels)}")
        if rels:
            supports = sum(1 for r in rels if r.relationship == "supports")
            contradicts = sum(1 for r in rels if r.relationship == "contradicts")
            print(f"   ✓ Supports: {supports}")
            print(f"   ✗ Contradicts: {contradicts}")
            
            if contradicts > 0:
                print(f"\n📋 Contradictions found:")
                for i, rel in enumerate([r for r in rels if r.relationship == "contradicts"][:2], 1):
                    print(f"{i}. Confidence: {rel.confidence:.2f}")
                    print(f"   {rel.reasoning[:120]}...")
    
    if 'disagreement_metrics' in result:
        metrics = result['disagreement_metrics']
        print(f"\n📊 Disagreement Metrics:")
        if metrics:
            for key, value in metrics.items():
                print(f"   {key}: {value}")
    
    if 'multi_view_answer' in result:
        answer = result['multi_view_answer']
        print(f"\n🎯 Multi-View Answer:")
        
        dom = answer.get('dominant_view', {})
        if dom:
            print(f"\n   DOMINANT VIEW:")
            print(f"   {dom.get('summary', '')[:150]}...")
        
        alts = answer.get('alternative_views', [])
        if alts:
            print(f"\n   ALTERNATIVE VIEWS: {len(alts)}")
            for i, alt in enumerate(alts[:2], 1):
                print(f"   {i}. {alt.get('summary', '')[:100]}...")
        
        print(f"\n   CONFIDENCE: {answer.get('overall_confidence', 'N/A')} ({answer.get('confidence_score', 0):.2f})")
    
    print(f"\n🎉 Test complete!")
    
except Exception as e:
    print(f"\n❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
