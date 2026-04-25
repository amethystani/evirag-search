#!/usr/bin/env python3
"""
NLI Diagnostic Test - Check if NLI is detecting contradictions
"""

from epistemic_engine import NLIReasoner, AtomicClaim

print("🔬 NLI CONTRADICTION DETECTION DIAGNOSTIC")
print("="*80)

# Test with local backend (Llama3)
print("\n📝 Testing with CLEAR contradictory claims:\n")

nli = NLIReasoner('local')

# Create test claims with clear contradiction
claim1 = AtomicClaim(
    claim_id='test1',
    text='Retrieval augmentation significantly improves model performance',
    source_chunk_id='chunk1',
    source_doc_id='doc1',
    source_doc_title='Paper A'
)

claim2 = AtomicClaim(
    claim_id='test2',
    text='Retrieval augmentation hurts model performance and reduces accuracy',
    source_chunk_id='chunk2',
    source_doc_id='doc2',
    source_doc_title='Paper B'
)

print(f"Claim 1: {claim1.text}")
print(f"Claim 2: {claim2.text}")
print(f"\n{'='*80}")
print("Expected: contradiction with high confidence")
print(f"{'='*80}\n")

# Test without prefilter to see raw NLI output
result = nli.analyze_relationship(claim1, claim2, use_prefilter=False)

print(f"✅ Result: {result.relationship}")
print(f"✅ Confidence: {result.confidence}")  
print(f"✅ Reasoning: {result.reasoning}")

if result.relationship == "contradiction":
    print(f"\n🎉 SUCCESS: NLI detected contradiction correctly!")
else:
    print(f"\n❌ PROBLEM: NLI failed to detect clear contradiction")
    print(f"This confirms the issue is in the NLI prompt/model, not in other parts of the system")
