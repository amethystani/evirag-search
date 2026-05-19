#!/usr/bin/env python3
"""
Test EVIRAG Cloud Mode with Groq API
Requires GROQ_API_KEY in .env file
"""

import os
import sys
import time
from pathlib import Path

# Load environment variables from .env file
def load_env():
    """Load .env file if it exists"""
    env_path = Path(__file__).parent / ".env"
    
    if not env_path.exists():
        print("❌ .env file not found!")
        print(f"   Create {env_path} and add your GROQ_API_KEY")
        sys.exit(1)
    
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    
    if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your-groq-api-key-here":
        print("❌ GROQ_API_KEY not set in .env file!")
        print("   Edit .env and replace 'your-groq-api-key-here' with your actual API key")
        print("   Get your key at: https://console.groq.com/keys")
        sys.exit(1)
    
    print("✅ Loaded GROQ_API_KEY from .env")


def test_groq_client():
    """Test Groq client connection"""
    print("\n" + "="*80)
    print("1. Testing Groq Client Connection")
    print("="*80)
    
    from groq_client import GroqClient
    
    try:
        client = GroqClient()
        print("✓ Client initialized")
        
        # Test simple call
        response = client.generate(
            model="llama-3.1-8b-instant",
            prompt="Say 'Cloud mode is working!' if you can read this.",
            temperature=0.1,
            max_tokens=20
        )
        
        print(f"✓ Response: {response}")
        print("✅ Groq client working!\n")
        return True
        
    except Exception as e:
        print(f"❌ Groq client failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cloud_evirag_simple():
    """Test EVIRAG cloud mode with simple query"""
    print("\n" + "="*80)
    print("2. Testing EVIRAG Cloud Mode - Simple Query")
    print("="*80)
    
    from evirag_system import EVIRAGSystem, EVIRAGConfig
    
    try:
        # Cloud config
        config = EVIRAGConfig(
            mode="evirag",
            backend="cloud",
            use_visual_grounding=False,
            enabled_agents=["precision", "recall"],
            depth_vs_speed="fast"
        )
        
        print(f"\n📋 Config:")
        print(f"   Backend: {config.backend}")
        print(f"   Mode: {config.mode}")
        print(f"   Agents: {config.enabled_agents}")
        
        # Initialize system
        system = EVIRAGSystem(config)
        system.initialize_corpus()
        
        # Test query
        query = "What is RAG?"
        print(f"\n📝 Query: {query}")
        print("⏱️  Starting cloud mode query...")
        
        start = time.time()
        result = system.query(query)
        elapsed = time.time() - start
        
        print(f"\n✅ Query completed in {elapsed:.1f}s")
        
        # Show results
        print(f"\n📊 Results:")
        for key in result.keys():
            print(f"   - {key}")
        
        if 'claims' in result:
            print(f"\n📝 Claims extracted: {len(result['claims'])}")
        
        if 'multi_view_answer' in result:
            print(f"\n🎯 Multi-view answer generated")
        
        print(f"\n⚡ Cloud mode is {elapsed/180:.0%} faster than expected local mode (30min)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Cloud EVIRAG failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cloud_vs_local_comparison():
    """Compare cloud and local mode initialization"""
    print("\n" + "="*80)
    print("3. Cloud vs Local Comparison")
    print("="*80)
    
    from evirag_system import EVIRAGSystem, EVIRAGConfig
    from config import GRAPH_CONFIG, SYSTEM_CONSTRAINTS
    
    print(f"\n📊 Configuration Differences:")
    print(f"   Max pairs (local):  {GRAPH_CONFIG['max_pairs']}")
    print(f"   Max pairs (cloud):  {GRAPH_CONFIG['max_pairs_cloud']}")
    print(f"   Max claims (local): {SYSTEM_CONSTRAINTS['max_claims_per_query']}")
    print(f"   Max claims (cloud): {SYSTEM_CONSTRAINTS['max_claims_cloud']}")
    
    print(f"\n🌐 Models:")
    print(f"   Cloud SLM:     llama-3.1-8b-instant")
    print(f"   Cloud LLM:     llama-3.3-70b-versatile")
    print(f"   Cloud Skeptic: deepseek-r1-distill-llama-70b")
    
    return True


def main():
    """Run all cloud mode tests"""
    print("🚀 EVIRAG Cloud Mode Test Suite")
    print("="*80)
    
    # Load API key
    load_env()
    
    # Run tests
    tests = [
        ("Groq Client", test_groq_client),
        ("Cloud EVIRAG", test_cloud_evirag_simple),
        ("Cloud vs Local", test_cloud_vs_local_comparison),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*80)
    print("📋 Test Summary")
    print("="*80)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! Cloud mode is ready.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
