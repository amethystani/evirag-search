#!/usr/bin/env python3
"""
FOCUSED NLI PROMPT TEST
Tests if the NLI prompt itself has a bias towards entailment
"""

from epistemic_engine import OllamaClient
import json

print("🔬 TESTING NLI PROMPT BIAS")
print("="*80)

client = OllamaClient()

# Create 5 test pairs with CLEAR contradictions
test_pairs = [
    {
        "claim1": "Large language models significantly improve performance on benchmark tasks",
        "claim2": "Large language models hurt performance and reduce accuracy on benchmark tasks"
    },
    {
        "claim1": "Retrieval augmentation enhances model capabilities",
        "claim2": "Retrieval augmentation degrades model capabilities"  
    },
    {
        "claim1": "AI systems are effective for medical diagnosis",
        "claim2": "AI systems are ineffective for medical diagnosis"
    },
    {
        "claim1": "Smaller models outperform larger models",
        "claim2": "Larger models outperform smaller models"
    },
    {
        "claim1": "Manual data processing is superior to automated methods",
        "claim2": "Automated methods are superior to manual data processing"
    }
]

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

results = []

for i, pair in enumerate(test_pairs, 1):
    print(f"\n{'='*80}")
    print(f"TEST PAIR {i}:")
    print(f"Claim 1: {pair['claim1']}")
    print(f"Claim 2: {pair['claim2']}")
    print(f"{'='*80}")
    
    prompt = f"""Determine the relationship between these two claims.

Claim 1: {pair['claim1']}
Source 1: Paper A

Claim 2: {pair['claim2']}
Source 2: Paper B

Relationship: Does Claim 2 entail, contradict, or remain neutral to Claim 1?

Respond:
{{
    "relationship": "entailment|contradiction|neutral",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""
    
    response = client.generate(
        model="llama3:latest",
        prompt=prompt,
        temperature=0.1,
        max_tokens=512,
        system=system_prompt
    )
    
    print(f"\nRaw Response:")
    print(response[:300])
    print("...")
    
    # Try to extract JSON
    try:
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            relationship = result.get("relationship", "unknown")
            confidence = result.get("confidence", 0.0)
            reasoning = result.get("reasoning", "")[:100]
            
            print(f"\n✅ Relationship: {relationship}")
            print(f"✅ Confidence: {confidence}")
            print(f"✅ Reasoning: {reasoning}...")
            
            results.append({
                "pair": i,
                "relationship": relationship,
                "confidence": confidence
            })
            
            if relationship != "contradiction":
                print(f"\n❌ ERROR: Clear contradiction classified as {relationship}!")
        else:
            print("\n❌ Failed to extract JSON")
            results.append({"pair": i, "relationship": "parse_failed", "confidence": 0})
    except Exception as e:
        print(f"\n❌ Error: {e}")
        results.append({"pair": i, "relationship": "error", "confidence": 0})

print(f"\n{'='*80}")
print("FINAL RESULTS:")
print(f"{'='*80}")

contradictions = sum(1 for r in results if r['relationship'] == 'contradiction')
entailments = sum(1 for r in results if r['relationship'] == 'entailment')
other = len(results) - contradictions - entailments

print(f"Total tests: {len(results)}")
print(f"Contradictions: {contradictions} ({contradictions/len(results)*100:.1f}%)")
print(f"Entailments: {entailments} ({entailments/len(results)*100:.1f}%)")
print(f"Other: {other}")

if contradictions == len(results):
    print("\n🎉 SUCCESS: Model correctly classifies all clear contradictions!")
elif contradictions < len(results) / 2:
    print("\n❌ MAJOR ISSUE: Model has severe entailment bias!")
    print("This confirms Llama3 itself is the problem, not the pipeline code.")
else:
    print("\n⚠️  WARNING: Model has some entailment bias but detects most contradictions")
