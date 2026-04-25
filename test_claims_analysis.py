#!/usr/bin/env python3
"""
Claims Analysis - Check if corpus has contradictory claims
"""

from evirag_system import EVIRAGSystem, EVIRAGConfig

print("🔬 CLAIMS ANALYSIS - Checking for Contradictions in Corpus")
print("="*80)

config = EVIRAGConfig(
    mode="evirag",
    backend="local",
    enabled_agents=["precision", "recall", "skeptic"],
    use_visual_grounding=False,  # Skip visual for speed
    depth_vs_speed="fast"
)

print("\nInitializing system...\n")
system = EVIRAGSystem(config)
system.initialize_corpus()

print("\n📊 Extracting claims for query:")
query = "Does retrieval augmentation improve or hurt model performance?"
print(f"'{query}'\n")

# Get claims from Step 2-3
from epistemic_engine import EpistemicEngine

engine = EpistemicEngine(backend='local')
claims = engine.extract_atomic_claims([query], system.corpus.get_all_chunks())

print(f"\n✅ Extracted {len(claims)} claims\n")
print("="*80)
print("SAMPLE CLAIMS (first 20):")
print("="*80)

for i, claim in enumerate(claims[:20]):
    print(f"\n{i+1}. {claim.text}")
    print(f"   Source: {claim.source_doc_title}")

# Look for obvious contradictions manually
print(f"\n\n{'='*80}")
print("🔍 MANUAL CONTRADICTION SCAN:")
print("="*80)

contradictory_keywords = [
    ("improve", "hurt"),
    ("effective", "ineffective"),
    ("increase", "decrease"),
    ("better", "worse"),
    ("outperform", "underperform"),
    ("advantage", "disadvantage"),
    ("enhance", "degrade"),
    ("successful", "unsuccessful"),
]

claim_texts = [c.text.lower() for c in claims]

found_contradictions = []
for i, text1 in enumerate(claim_texts):
    for k1, k2 in contradictory_keywords:
        if k1 in text1:
            # Look for opposing claim
            for j, text2 in enumerate(claim_texts):
                if i != j and k2 in text2:
                    # Check if about same topic (simple heuristic)
                    words1 = set(text1.split())
                    words2 = set(text2.split())
                    overlap = len(words1 & words2)
                    if overlap > 3:
                        found_contradictions.append((i, j, claims[i].text, claims[j].text))

print(f"\n✅ Found {len(found_contradictions)} potential contradictory pairs:")

for i, (idx1, idx2, text1, text2) in enumerate(found_contradictions[:5]):
    print(f"\n{i+1}. PAIR ({idx1}, {idx2}):")
    print(f"   A: {text1}")
    print(f"   B: {text2}")

if len(found_contradictions) == 0:
    print("\n❌ NO contradictory claims found in extracted claims!")
    print(f"This explains why NLI finds only entailments - the claims genuinely don't contradict each other.")
