#!/usr/bin/env python3
"""
Quick Cloud Test - Verbose Debugging
"""
import os, sys
from pathlib import Path

# Load env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

print("🔍 Quick Cloud Test - Verbose Mode\n")

from groq_client import GroqClient

# Test 1: Direct API call
print("1. Testing direct Groq API call...")
try:
    client = GroqClient()
    response = client.generate(
        model="llama-3.1-8b-instant",
        prompt="Say 'hello'",
        max_tokens=5,
        temperature=0.1
    )
    print(f"   ✅ Response: {response}\n")
except Exception as e:
    print(f"   ❌ Failed: {e}\n")
    sys.exit(1)

# Test 2: Multi-agent with cloud backend
print("2. Testing multi-agent with cloud backend...")
try:
    from multi_agent import PrecisionAgent
    
    agent = PrecisionAgent(
        name="precision",
        objective="Test",
        backend="cloud"
    )
    print(f"   ✅ Agent created: {agent.model_config.name}\n")
    
    # Test agent query generation
    print("3. Testing agent query generation...")
    from epistemic_engine import ExplanationHypothesis, EpistemicIntent
    
    hypothesis = ExplanationHypothesis("Test query", [], [], [])
    
    # Mock state
    from multi_agent import AgentState
    state = AgentState(hypothesis, [], [], "low")
    
    query = agent.generate_retrieval_query(hypothesis, state)
    print(f"   ✅ Generated query: {query[:50]}...\n")
    
    print("✅ All agent tests passed!")
    
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
