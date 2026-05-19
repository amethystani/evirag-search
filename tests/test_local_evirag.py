#!/usr/bin/env python3
"""
Local Mode EVIRAG Test - 15 pairs, 0.4 confidence
"""

import sys
import time

def main():
    print("🚀 Local Mode EVIRAG Test (15 pairs, 0.4 confidence)")
    print("="*80)
    
    from evirag_system import EVIRAGSystem, EVIRAGConfig
    from config import GRAPH_CONFIG
    
    print(f"\n📊 Local Limits:")
    print(f"   Max NLI pairs: {GRAPH_CONFIG['max_pairs']}")
    print(f"   Min confidence: {GRAPH_CONFIG['min_edge_confidence']}")
    print(f"   Agents: precision, recall")
    
    try:
        config = EVIRAGConfig(
            mode="evirag",
            backend="local",  # Use Ollama
            use_visual_grounding=False,
            enabled_agents=["precision", "recall"],
            depth_vs_speed="fast"
        )
        
        system = EVIRAGSystem(config)
        system.initialize_corpus()
        
        query = "What are the key techniques in RAG?"
        print(f"\n📝 Query: {query}")
        print("\n⏱️  Running full EVIRAG with Ollama...\n")
        
        start = time.time()
        result = system.query(query)
        elapsed = time.time() - start
        
        print(f"\n{'='*80}")
        print(f"✅ COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print(f"{'='*80}")
        
        # Results
        if 'claims' in result:
            print(f"\n📝 Claims extracted: {len(result['claims'])}")
        
        if 'relationships' in result:
            rels = result['relationships']
            print(f"🔗 Relationships found: {len(rels)}")
            
            if len(rels) > 0:
                supports = sum(1 for r in rels if r.relationship == "supports")
                contradicts = sum(1 for r in rels if r.relationship == "contradicts")
                print(f"   - Supports: {supports}")
                print(f"   - Contradicts: {contradicts}")
                
                print(f"\n📋 Sample relationships:")
                for i, rel in enumerate(rels[:3], 1):
                    print(f"{i}. {rel.relationship.upper()} (conf: {rel.confidence:.2f})")
                    print(f"   {rel.reasoning[:100]}...")
        
        if 'disagreement_metrics' in result:
            metrics = result['disagreement_metrics']
            print(f"\n📊 Disagreement Metrics:")
            print(f"   Density: {metrics.get('density', 'N/A')}")
            print(f"   Conflict ratio: {metrics.get('conflict_ratio', 'N/A')}")
        
        if 'multi_view_answer' in result:
            answer = result['multi_view_answer']
            print(f"\n🎯 Multi-View Answer:")
            dom = answer.get('dominant_view', {})
            if dom:
                print(f"   Dominant: {dom.get('summary', '')[:120]}...")
            print(f"   Confidence: {answer.get('overall_confidence', 'N/A')} ({answer.get('confidence_score', 0):.2f})")
            
            alts = answer.get('alternative_views', [])
            if alts:
                print(f"   Alternative views: {len(alts)}")
        
        print(f"\n🎉 Local mode EVIRAG complete!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
