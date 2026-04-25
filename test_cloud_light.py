#!/usr/bin/env python3
"""
Lightweight Cloud Mode Test - Respects Rate Limits
Tests cloud mode with minimal API calls
"""

import os
import sys
import time
from pathlib import Path

# Load .env
def load_env():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        print("❌ .env file not found!")
        sys.exit(1)
    
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    
    if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your-groq-api-key-here":
        print("❌ GROQ_API_KEY not set!")
        sys.exit(1)
    
    print("✅ Loaded API key from .env\n")

def test_groq_models():
    """Test all Groq models"""
    print("="*80)
    print("1. Testing Groq Models")
    print("="*80)
    
    from groq_client import GroqClient
    
    client = GroqClient()
    
    models = [
        ("llama-3.1-8b-instant", "SLM"),
        ("llama-3.3-70b-versatile", "LLM"),
        ("deepseek-r1-distill-llama-70b", "Skeptic")
    ]
    
    for model_name, role in models:
        try:
            response = client.generate(
                model=model_name,
                prompt=f"Say 'Working' if you can read this.",
                max_tokens=10,
                temperature=0.1
            )
            print(f"✅ {role:10} ({model_name}): {response[:50]}")
        except Exception as e:
            print(f"❌ {role:10} ({model_name}): {e}")
            return False
    
    return True

def test_vanilla_rag_cloud():
    """Test vanilla RAG with cloud backend (minimal calls)"""
    print("\n" + "="*80)
    print("2. Testing Vanilla RAG (Cloud Backend)")
    print("="*80)
    
    from evirag_system import EVIRAGSystem, EVIRAGConfig
    
    try:
        config = EVIRAGConfig(
            mode="vanilla_rag",  # Simple mode
            backend="cloud"
        )
        
        system = EVIRAGSystem(config)
        system.initialize_corpus()
        
        query = "What is RAG?"
        print(f"\n📝 Query: {query}")
        
        start = time.time()
        result = system.query(query)
        elapsed = time.time() - start
        
        print(f"✅ Completed in {elapsed:.1f}s")
        print(f"   Answer preview: {result.get('answer', '')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def test_cloud_components():
    """Test individual cloud components"""
    print("\n" + "="*80)
    print("3. Testing EVIRAG Cloud Components")
    print("="*80)
    
    from evirag_system import EVIRAGSystem, EVIRAGConfig
    from epistemic_engine import EpistemicIntent
    
    try:
        config = EVIRAGConfig(mode="evirag", backend="cloud", use_visual_grounding=False)
        system = EVIRAGSystem(config)
        
        # Test intent analysis
        print("\n  a) Intent Analysis...")
        query = "What is RAG?"
        intent = system.epistemic_engine.intent_analyzer.analyze(query)
        print(f"     ✅ Intent: {intent.disagreement_level} disagreement")
        
        # Test hypothesis generation
        print("\n  b) Hypothesis Generation...")
        hypothesis = system.epistemic_engine.hypothesis_generator.generate(query, intent, use_llm=True)
        print(f"     ✅ Hypothesis: {hypothesis.central_hypothesis[:80]}...")
        
        print("\n✅ All components working!")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 EVIRAG Cloud Mode - Lightweight Test")
    print("="*80 + "\n")
    
    load_env()
    
    tests = [
        ("Groq Models", test_groq_models),
        ("Vanilla RAG", test_vanilla_rag_cloud),
        ("Cloud Components", test_cloud_components),
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
    print("📋 Summary")
    print("="*80)
    
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
    
    if all(results.values()):
        print("\n🎉 Cloud mode verified! All systems operational.")
        print("\n💡 Note: Full EVIRAG with disagreement analysis uses many API calls")
        print("   and may hit Groq rate limits. Use vanilla_rag for simple queries.")
        return 0
    else:
        print("\n⚠️  Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
