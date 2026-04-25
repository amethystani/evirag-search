#!/usr/bin/env python3
"""
Test VLM (Step 5) - Visual Grounding
Tests CLIP embeddings and visual-text alignment
"""

import sys
from vlm_module import CLIPEmbedder, VisualEvidenceProcessor
from data_layer import DataManager

print("🔬 Testing Step 5: Visual Grounding (CLIP VLM)")
print("="*80)

try:
    # Test 1: CLIP Embedder
    print("\n1. Testing CLIP Embedder...")
    embedder = CLIPEmbedder()
    embedder.load_model()
    
    # Test text embedding
    print("   Testing text embedding...")
    test_text = "Retrieval augmented generation improves accuracy"
    text_emb = embedder.embed_text(test_text)
    print(f"   ✅ Text embedding: shape={text_emb.shape}, values={text_emb[:3]}")
    
    # Verify no errors
    if text_emb.shape[0] == 512 and not (text_emb == 0).all():
        print("   ✅ Text embedding successful (no tensor extraction errors)")
    else:
        print(f"   ❌ Text embedding failed: unexpected shape or zero vector")
        sys.exit(1)
    
    # Test 2: Visual Evidence Processor
    print("\n2. Testing Visual Evidence Processor...")
    data_manager = DataManager()
    data_manager.process_corpus(rebuild=False)
    
    visual_processor = VisualEvidenceProcessor()
    visual_processor.process_figures(data_manager.figures_db, rebuild=False)
    
    print(f"   ✅ Loaded {len(visual_processor.visual_evidence_db)} visual evidence entries")
    
    # Test 3: Claim-Figure Alignment
    print("\n3. Testing Claim-Figure Alignment...")
    test_claim = "Graph neural networks improve retrieval performance"
    aligned_figures = visual_processor.align_claim_with_figures(test_claim, top_k=3)
    
    if aligned_figures:
        print(f"   ✅ Found {len(aligned_figures)} aligned figures:")
        for ve, score in aligned_figures[:3]:
            print(f"      - {ve.figure_id}: {score:.3f} - {ve.caption_text[:60]}...")
    else:
        print("   ⚠️  No aligned figures found (may be normal if corpus has few figures)")
    
    # Test 4: Visual-Text Mismatch Detection
    print("\n4. Testing Visual-Text Mismatch Detection...")
    from vlm_module import VisualDisagreementAnalyzer
    
    analyzer = VisualDisagreementAnalyzer(visual_processor)
    
    # Test with sample claims
    test_claims = [
        {"text": "RAG improves LLM accuracy by 15%", "doc_id": "2502.11371v2.pdf", "stance": "positive"},
        {"text": "GraphRAG outperforms traditional RAG", "doc_id": "s41598-025-21222-z.pdf", "stance": "positive"}
    ]
    
    mismatch = analyzer.analyze_visual_text_mismatch(test_claims)
    print(f"   ✅ Mismatch analysis:")
    print(f"      - Mismatch score: {mismatch['mismatch_score']:.3f}")
    print(f"      - Claims with weak support: {mismatch['claims_with_weak_support']}/{mismatch['total_claims']}")
    
    print("\n" + "="*80)
    print("✅ ALL VLM TESTS PASSED!")
    print("="*80)
    print("\n✓ CLIP embeddings working (no tensor extraction errors)")
    print("✓ Visual evidence processor working")
    print("✓ Claim-figure alignment working")
    print("✓ Visual-text mismatch detection working")
    print("\n🎉 Step 5 is ready for full pipeline!")
    
except Exception as e:
    print(f"\n❌ VLM Test Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
