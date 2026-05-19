#!/usr/bin/env python3
"""
Cloud Backend Test - Groq with 500 Pairs
Tests contradiction detection with cloud models for better calibration
"""

import sys, time, os

# Load GROQ_API_KEY from .env file
if os.path.exists('.env'):
    with open('.env') as f:
        for line in f:
            if line.startswith('GROQ_API_KEY='):
                os.environ['GROQ_API_KEY'] = line.split('=', 1)[1].strip()

print("☁️  CLOUD BACKEND TEST (Groq)")
print("="*80)
print("\n📝 Query: Does retrieval augmentation improve or hurt model performance?")
print("\n🎯 Testing with cloud backend to validate proper contradiction detection")
print("\n💡 Cloud models should detect 10-15% contradictions (vs. 0-1% local)")
print(f"\n📊 Config: 500 pairs (cloud), 0.35 min confidence\n")

from evirag_system import EVIRAGSystem, EVIRAGConfig

try:
    config = EVIRAGConfig(
        mode="evirag",
        backend="cloud",  # Using Groq cloud backend
        enabled_agents=["precision", "recall", "skeptic"],
        use_visual_grounding=True,
        depth_vs_speed="fast"
    )
    
    print("Initializing EVIRAG System (backend: cloud - Groq)...\n")
    system = EVIRAGSystem(config)
    system.initialize_corpus()
    
    start = time.time()
    result = system.query("Does retrieval augmentation improve or hurt model performance?")
    elapsed = time.time() - start
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")
    print(f"\n🎉 Cloud test complete!")
    print(f"\n📝 Check if contradiction detection improved with cloud models.")
    
except Exception as e:
    print(f"\n❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
