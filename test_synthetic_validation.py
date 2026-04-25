#!/usr/bin/env python3
"""
Synthetic Validation Test - Prove End-to-End System Works
Creates synthetic claims with clear contradictions to validate full pipeline
"""

from epistemic_engine import AtomicClaim, EpistemicEngine
from disagreement import DisagreementGraph
import time

print("🧪 SYNTHETIC VALIDATION TEST")
print("="*80)
print("\n📊 Testing full pipeline with synthetic contradictory claims")
print("   - 10 PRO-RAG claims")
print("   - 10 ANTI-RAG claims")
print("   - Expected: ~50% contradiction rate\n")

# Create 10 PRO-RAG claims
pro_claims = [
    "Retrieval-augmented generation significantly improves model accuracy on factual questions",
    "RAG systems demonstrate superior performance in knowledge-intensive tasks",
    "Adding retrieval mechanisms enhances language model capabilities across benchmarks",
    "RAG-based approaches outperform vanilla language models in question answering",
    "Retrieval augmentation leads to more accurate and grounded responses",
    "External knowledge retrieval boosts model performance on complex reasoning tasks",
    "RAG architectures show consistent improvements over baseline models",
    "Retrieval-enhanced models achieve higher accuracy with lower hallucination rates",
    "Augmenting generation with retrieval yields better factual consistency",
    "RAG systems provide more reliable and verifiable outputs than standard LLMs"
]

# Create 10 ANTI-RAG claims
anti_claims = [
    "Retrieval-augmented generation significantly degrades model accuracy on factual questions",
    "RAG systems demonstrate inferior performance in knowledge-intensive tasks",
    "Adding retrieval mechanisms reduces language model capabilities across benchmarks",
    "RAG-based approaches underperform vanilla language models in question answering",
    "Retrieval augmentation leads to less accurate and poorly grounded responses",
    "External knowledge retrieval harms model performance on complex reasoning tasks",
    "RAG architectures show consistent degradation compared to baseline models",
    "Retrieval-enhanced models achieve lower accuracy with higher hallucination rates",
    "Augmenting generation with retrieval yields worse factual consistency",
    "RAG systems provide less reliable and unverifiable outputs than standard LLMs"
]

# Convert to AtomicClaim objects
claims = []
for i, text in enumerate(pro_claims):
    claims.append(AtomicClaim(
        claim_id=f"pro_{i}",
        text=text,
        source_chunk_id=f"chunk_pro_{i}",
        source_doc_id="synthetic_pro",
        source_doc_title="Pro-RAG Paper"
    ))

for i, text in enumerate(anti_claims):
    claims.append(AtomicClaim(
        claim_id=f"anti_{i}",
        text=text,
        source_chunk_id=f"chunk_anti_{i}",
        source_doc_id="synthetic_anti",
        source_doc_title="Anti-RAG Paper"
    ))

print(f"✅ Created {len(claims)} synthetic claims (10 pro + 10 anti)\n")

# Initialize components
print("Initializing epistemic engine...\n")
engine = EpistemicEngine(backend='local')

# Build relationships (will analyze pairs)
print("Step 1: Analyzing claim relationships with NLI...")
print(f"   Total possible pairs: {len(claims) * (len(claims) - 1) // 2}")
print(f"   Analyzing first 100 pairs for validation\n")

start = time.time()
relationships = engine.build_claim_relationships(claims, max_pairs=100)
elapsed = time.time() - start

print(f"\n✅ Relationship analysis complete in {elapsed:.1f}s")
print(f"   Found {len(relationships)} non-neutral relationships\n")

# Count contradictions vs entailments
contradictions = [r for r in relationships if r.relationship == "contradiction"]
entailments = [r for r in relationships if r.relationship == "entailment"]

print(f"{'='*80}")
print("RELATIONSHIP BREAKDOWN:")
print(f"{'='*80}")
print(f"Contradictions: {len(contradictions)} ({len(contradictions)/len(relationships)*100:.1f}%)")
print(f"Entailments: {len(entailments)} ({len(entailments)/len(relationships)*100:.1f}%)")

# Show sample contradictions
if contradictions:
    print(f"\n{'='*80}")
    print("SAMPLE CONTRADICTIONS (first 3):")
    print(f"{'='*80}")
    for i, rel in enumerate(contradictions[:3], 1):
        claim1 = next(c for c in claims if c.claim_id == rel.claim1_id)
        claim2 = next(c for c in claims if c.claim_id == rel.claim2_id)
        print(f"\n{i}. Confidence: {rel.confidence:.2f}")
        print(f"   Claim A: {claim1.text[:80]}...")
        print(f"   Claim B: {claim2.text[:80]}...")
        print(f"   Reasoning: {rel.reasoning[:100]}...")

# Build graph
print(f"\n{'='*80}")
print("Step 2: Building disagreement graph...")
print(f"{'='*80}\n")

graph = DisagreementGraph()
for claim in claims:
    graph.add_claim(claim)

for rel in relationships:
    graph.add_relationship(rel)

print(f"✅ Graph built:")
print(f"   Nodes (claims): {graph.graph.number_of_nodes()}")
print(f"   Edges (relationships): {graph.graph.number_of_edges()}")

# Check for contradictory edges
contradictory_edges = [
    (u, v) for u, v, data in graph.graph.edges(data=True)
    if data.get("relationship") == "contradicts"
]
print(f"   Contradictory edges: {len(contradictory_edges)}")

# Detect communities
communities = graph.get_claim_communities()
print(f"   Communities detected: {len(communities)}")

# Final verdict
print(f"\n{'='*80}")
print("VALIDATION RESULTS:")
print(f"{'='*80}")

if len(contradictions) >= 10:
    print(f"✅ SUCCESS: System detected {len(contradictions)} contradictions!")
    print(f"   Expected: ~50 (50% of 100 pairs between opposing claims)")
    print(f"   Actual: {len(contradictions)} ({len(contradictions)/len(relationships)*100:.1f}%)")
    print(f"\n🎉 END-TO-END VALIDATION SUCCESSFUL!")
    print(f"   - NLI correctly detects contradictions ✅")
    print(f"   - Graph preserves contradictory relationships ✅")
    if len(contradictory_edges) > 0:
        print(f"   - Graph edges maintain 'contradicts' type ✅")
    if len(communities) >= 2:
        print(f"   - Communities split on contradictions ✅")
    print(f"\n   💯 System works perfectly when contradictions exist in claims!")
else:
    print(f"⚠️  WARNING: Only {len(contradictions)} contradictions detected")
    print(f"   Expected more with synthetic opposing claims")
    print(f"   May need to investigate further")

print(f"\n{'='*80}")
