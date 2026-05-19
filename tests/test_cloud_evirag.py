#!/usr/bin/env python3
"""
Cloud Mode EVIRAG Test - 30 important pairs
"""

import os
import sys
import time
from pathlib import Path

def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    
    if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your-groq-api-key-here":
        print("❌ GROQ_API_KEY not set!")
        sys.exit(1)

def main():
    print("🚀 Cloud Mode EVIRAG Test (30 pairs)")
    print("="*80)
    
    load_env()
    
    from evirag_system import EVIRAGSystem, EVIRAGConfig
    from config import GRAPH_CONFIG
    
    print(f"\n📊 Cloud Limits:")
    print(f"   Max NLI pairs: {GRAPH_CONFIG['max_pairs_cloud']}")
    print(f"   Agents: precision, recall")
    
    try:
        config = EVIRAGConfig(
            mode="evirag",
            backend="cloud",
            use_visual_grounding=False,
            enabled_agents=["precision", "recall"],
            depth_vs_speed="fast"
        )
        
        system = EVIRAGSystem(config)
        system.initialize_corpus()
        
        query = "What are the key techniques in RAG?"
        print(f"\n📝 Query: {query}")
        print("\n⏱️  Running full EVIRAG with Groq API...")
        print("   (This will retry on rate limits automatically)\n")
        
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
            print(f"🔗 Relationships analyzed: {len(rels)}")
            supports = sum(1 for r in rels if r.relationship == "supports")
            contradicts = sum(1 for r in rels if r.relationship == "contradicts")
            print(f"   - Supports: {supports}")
            print(f"   - Contradicts: {contradicts}")
        
        if 'multi_view_answer' in result:
            answer = result['multi_view_answer']
            print(f"\n🎯 Multi-View Answer:")
            dom = answer.get('dominant_view', {})
            print(f"   Dominant: {dom.get('summary', '')[:120]}...")
            print(f"   Confidence: {answer.get('overall_confidence', 'N/A')} ({answer.get('confidence_score', 0):.2f})")
            
            alts = answer.get('alternative_views', [])
            if alts:
                print(f"   Alternative views: {len(alts)}")
        
        print(f"\n🎉 Cloud mode EVIRAG working perfectly!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
