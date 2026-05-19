#!/usr/bin/env python3
"""
Full EVIRAG Test with Reduced Limits (Testable on both modes)
Tests complete EVIRAG pipeline with 25/50 NLI pairs
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

def test_full_evirag(backend="local"):
    """Test full EVIRAG pipeline with realistic limits"""
    print(f"\n{'='*80}")
    print(f"Testing Full EVIRAG ({backend.upper()} mode)")
    print(f"{'='*80}")
    
    from evirag_system import EVIRAGSystem, EVIRAGConfig
    from config import GRAPH_CONFIG, SYSTEM_CONSTRAINTS
    
    # Show limits
    max_pairs = GRAPH_CONFIG['max_pairs_cloud'] if backend == "cloud" else GRAPH_CONFIG['max_pairs']
    max_claims = SYSTEM_CONSTRAINTS['max_claims_cloud'] if backend == "cloud" else SYSTEM_CONSTRAINTS['max_claims_per_query']
    
    print(f"\n📊 Limits:")
    print(f"   Max NLI pairs: {max_pairs}")
    print(f"   Max claims: {max_claims}")
    
    try:
        config = EVIRAGConfig(
            mode="evirag",
            backend=backend,
            use_visual_grounding=False,
            enabled_agents=["precision", "recall"],  # 2 agents for speed
            depth_vs_speed="fast"
        )
        
        system = EVIRAGSystem(config)
        system.initialize_corpus()
        
        query = "What are the key techniques in RAG?"
        print(f"\n📝 Query: {query}")
        print("⏱️  Running full EVIRAG pipeline...")
        
        start = time.time()
        result = system.query(query)
        elapsed = time.time() - start
        
        print(f"\n✅ Completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")
        
        # Show results
        if 'claims' in result:
            print(f"\n📝 Claims: {len(result['claims'])}")
        
        if 'relationships' in result:
            print(f"🔗 Relationships: {len(result['relationships'])}")
        
        if 'multi_view_answer' in result:
            answer = result['multi_view_answer']
            print(f"\n🎯 Multi-view Answer:")
            print(f"   Dominant view: {answer.get('dominant_view', {}).get('summary', '')[:100]}...")
            print(f"   Confidence: {answer.get('overall_confidence', 'N/A')}")
        
        return True, elapsed
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def main():
    print("🚀 Full EVIRAG Test Suite (Reduced Limits)")
    print("="*80)
    
    load_env()
    
    # Test local mode
    print("\n\n1. LOCAL MODE TEST")
    print("="*80)
    local_success, local_time = test_full_evirag("local")
    
    # Test cloud mode (if API key available)
    cloud_success = None
    cloud_time = 0
    
    if os.getenv("GROQ_API_KEY") and os.getenv("GROQ_API_KEY") != "your-groq-api-key-here":
        print("\n\n2. CLOUD MODE TEST")
        print("="*80)
        cloud_success, cloud_time = test_full_evirag("cloud")
    else:
        print("\n\n2. CLOUD MODE TEST")
        print("="*80)
        print("⚠️  Skipped - No Groq API key found")
    
    # Summary
    print("\n\n" + "="*80)
    print("📋 Test Summary")
    print("="*80)
    
    if local_success:
        print(f"✅ Local mode: {local_time:.1f}s")
    else:
        print("❌ Local mode: Failed")
    
    if cloud_success is not None:
        if cloud_success:
            print(f"✅ Cloud mode: {cloud_time:.1f}s")
            if local_success and local_time > 0:
                speedup = local_time / cloud_time if cloud_time > 0 else 0
                print(f"\n⚡ Cloud is {speedup:.1f}x faster than local!")
        else:
            print("❌ Cloud mode: Failed")
    
    if local_success or cloud_success:
        print("\n🎉 Full EVIRAG pipeline working with reduced limits!")
        print("\n💡 These limits (25/50 pairs) make the system testable")
        print("   For production, increase max_pairs in config.py")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
