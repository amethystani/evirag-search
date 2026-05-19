#!/usr/bin/env python3
"""
Debug NLI in Full Pipeline - Check what NLI is actually returning
"""

from epistemic_engine import NLIReasoner, AtomicClaim
import sys

print("🔬 DEBUGGING NLI IN FULL PIPELINE")
print("="*80)

# Create NLI reasoner
nli = NLIReasoner('local')

# Test 1: Simple opposing claims about model size
claim1 = AtomicClaim(
    claim_id='test1',
    text='Larger language models demonstrate superior performance across benchmarks',
    source_chunk_id='chunk1',
    source_doc_id='doc1',
    source_doc_title='Paper A'
)

claim2 = AtomicClaim(
    claim_id='test2',
    text='Smaller models achieve better results with limited computational resources',
    source_chunk_id='chunk2',
    source_doc_id='doc2',
    source_doc_title='Paper B'
)

print("\n📝 Test 1: Larger vs Smaller Models")
print(f"Claim 1: {claim1.text}")
print(f"Claim 2: {claim2.text}")
print()

# Call using same path as pipeline (use_prefilter=False)
result = nli.analyze_relationship(claim1, claim2, use_prefilter=False)

print(f"✅ Relationship: {result.relationship}")
print(f"✅ Confidence: {result.confidence}")
print(f"✅ Reasoning: {result.reasoning}")

if result.relationship != "contradiction":
    print("\n❌ PROBLEM: NLI classified opposing claims as", result.relationship)
    print("This is the bug - NLI is not detecting contradictions even with clear opposing statements")
    
    # Try to see raw response
    print("\n🔍 Testing raw LLM response...")
    prompt = f"""Determine the relationship between these two claims.

Claim 1: {claim1.text}
Source 1: {claim1.source_doc_title}

Claim 2: {claim2.text}
Source 2: {claim2.source_doc_title}

Relationship: Does Claim 2 entail, contradict, or remain neutral to Claim 1?

Respond:
{{
    "relationship": "entailment|contradiction|neutral",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

    system_prompt = """You are an expert at determining logical relationships between scientific claims.

IMPORTANT: Be sensitive to contradictions - if claims present genuinely opposing views, conflicting results, or contradictory findings (even partially), classify as 'contradiction'.

Examples of contradictions to detect:
- "RAG improves performance" vs "RAG hurts performance"  
- "Model X outperforms Y" vs "Model Y outperforms X"
- "Method shows significant gains" vs "Method shows no improvement"
- "Approach is effective" vs "Approach is ineffective"

Classify as:
- entailment: claim2 supports, agrees with, or reinforces claim1
- contradiction: claims present opposing, conflicting, or contradictory information
- neutral: claims are unrelated or neither support nor oppose

Be precise and provide reasoning. Always return valid JSON."""

    response = nli.client.generate(
        model="llama3:latest",
        prompt=prompt,
        temperature=0.1,
        max_tokens=512,
        system=system_prompt
    )
    
    print("\n📄 Raw LLM Response:")
    print(response)
    print("\n" + "="*80)
    
    extracted = nli.client.extract_json(response)
    print("\n📦 Extracted JSON:")
    print(extracted)
    
else:
    print("\n🎉 SUCCESS: NLI correctly detected contradiction!")
