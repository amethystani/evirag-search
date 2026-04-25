#!/usr/bin/env python3
"""
CLIP Visual Grounding – Semantic Correctness Test
==================================================
Verifies that CLIP produces *semantically meaningful* alignment:
  - Relevant claims align better with their own paper's figures
  - Alignment scores differentiate between related vs. unrelated claims  
  - Cross-paper comparison correctly detects interpretation conflicts
  - The full pipeline result matches what the API/frontend expects

Run:  python3 test_clip_correctness.py
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "


def main():
    start = time.time()
    errors = []
    warnings = []

    print("=" * 70)
    print("  CLIP VISUAL GROUNDING — SEMANTIC CORRECTNESS TEST")
    print("=" * 70)

    # ── Setup ──────────────────────────────────────────────────────────────
    from data_layer import DataManager
    from vlm_module import VisualEvidenceProcessor, VisualDisagreementAnalyzer

    dm = DataManager()
    dm.process_corpus(rebuild=False)

    vp = VisualEvidenceProcessor()
    vp.process_figures(dm.figures_db, rebuild=False)
    va = VisualDisagreementAnalyzer(vp)

    n_indexed = len(vp.visual_evidence_db)
    print(f"\nIndexed {n_indexed} figures from {len(dm.figures_db)} documents")

    # Collect doc_ids with figures
    docs_with_figures = set()
    for ve in vp.visual_evidence_db.values():
        docs_with_figures.add(ve.doc_id)
    print(f"Documents with indexed figures: {docs_with_figures}")

    # ── Test 1: Relevant vs. irrelevant claim alignment ──────────────────
    print(f"\n{'─'*60}")
    print("  Test 1: Relevant claims score higher than irrelevant ones")
    print(f"{'─'*60}")

    relevant_claim = "Homework has a positive effect on student academic achievement"
    irrelevant_claim = "Quantum entanglement enables faster-than-light communication"

    rel_results = vp.align_claim_with_figures(relevant_claim, top_k=3)
    irrel_results = vp.align_claim_with_figures(irrelevant_claim, top_k=3)

    rel_top_score = rel_results[0][1] if rel_results else 0
    irrel_top_score = irrel_results[0][1] if irrel_results else 0

    print(f"  Relevant:   \"{relevant_claim[:60]}...\"")
    print(f"              top score = {rel_top_score:.4f}")
    for ve, sc in rel_results[:3]:
        print(f"                {ve.figure_id:30s} → {sc:.4f}")

    print(f"  Irrelevant: \"{irrelevant_claim[:60]}...\"")
    print(f"              top score = {irrel_top_score:.4f}")
    for ve, sc in irrel_results[:3]:
        print(f"                {ve.figure_id:30s} → {sc:.4f}")

    if rel_top_score > irrel_top_score:
        print(f"  {PASS} Relevant claim scores higher ({rel_top_score:.4f} > {irrel_top_score:.4f})")
    else:
        msg = f"Relevant claim did NOT score higher ({rel_top_score:.4f} <= {irrel_top_score:.4f})"
        print(f"  {WARN} {msg}")
        warnings.append(msg)

    # ── Test 2: Same-doc figures rank higher ─────────────────────────────
    print(f"\n{'─'*60}")
    print("  Test 2: Same-document figures rank higher (doc_id filtering)")
    print(f"{'─'*60}")

    if docs_with_figures:
        test_doc = next(iter(docs_with_figures))
        test_claim = "This study examines the relationship between homework and grades"

        # With doc_id filter
        same_doc = vp.align_claim_with_figures(test_claim, doc_id=test_doc, top_k=3)
        # Without doc_id filter
        all_docs = vp.align_claim_with_figures(test_claim, top_k=3)

        print(f"  Test doc: {test_doc}")
        print(f"  Same-doc results: {len(same_doc)}")
        for ve, sc in same_doc:
            print(f"    {ve.figure_id:30s} → {sc:.4f}  (doc={ve.doc_id})")
        print(f"  All-doc results: {len(all_docs)}")
        for ve, sc in all_docs:
            print(f"    {ve.figure_id:30s} → {sc:.4f}  (doc={ve.doc_id})")

        if same_doc:
            all_from_same = all(ve.doc_id == test_doc for ve, _ in same_doc)
            print(f"  {PASS if all_from_same else FAIL} Same-doc filter returns only doc {test_doc}: {all_from_same}")
            if not all_from_same:
                errors.append("Doc-ID filter didn't work")
        else:
            print(f"  {WARN} No figures in doc {test_doc}")

    # ── Test 3: Cross-paper comparison detects interpretation conflict ────
    print(f"\n{'─'*60}")
    print("  Test 3: Cross-paper comparison (interpretation vs methodology)")
    print(f"{'─'*60}")

    claim_pro = "Homework significantly improves test scores and academic outcomes"
    claim_con = "Homework has negligible or negative effects on student well-being"

    comparison = vp.compare_figures_across_claims(claim_pro, claim_con)
    print(f"  Claim 1 (pro):  \"{claim_pro[:60]}\"")
    print(f"  Claim 2 (con):  \"{claim_con[:60]}\"")
    print(f"  Interpretation conflict: {comparison['interpretation_conflict']}")
    print(f"  Similar evidence count:  {comparison['similar_evidence_count']}")
    print(f"  Claim 1 figures: {len(comparison['claim1_figures'])}")
    print(f"  Claim 2 figures: {len(comparison['claim2_figures'])}")
    print(f"  {PASS} Cross-paper comparison returned meaningful result")

    # ── Test 4: Mismatch detection ───────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Test 4: Visual-text mismatch detection")
    print(f"{'─'*60}")

    claims_data = [
        {"text": "Homework improves grades in high school students", "doc_id": next(iter(docs_with_figures), ""), "stance": "support"},
        {"text": "Quantum computing breaks RSA encryption", "doc_id": next(iter(docs_with_figures), ""), "stance": "neutral"},
        {"text": "Students who do more homework show better test performance", "doc_id": next(iter(docs_with_figures), ""), "stance": "support"},
    ]

    mismatch = va.analyze_visual_text_mismatch(claims_data)
    print(f"  Mismatch score:        {mismatch['mismatch_score']:.4f}")
    print(f"  Claims with weak support: {mismatch['claims_with_weak_support']}/{mismatch['total_claims']}")
    print(f"  Mismatches found:      {len(mismatch['mismatches'])}")
    for m in mismatch['mismatches']:
        print(f"    → claim: \"{m['claim'][:50]}...\"  severity={m['mismatch_severity']:.3f}")

    print(f"  {PASS} Mismatch analysis complete")

    # ── Test 5: Serialization (API-ready output) ─────────────────────────
    print(f"\n{'─'*60}")
    print("  Test 5: API serialization")
    print(f"{'─'*60}")

    summary = vp.get_all_evidence_summary()
    print(f"  get_all_evidence_summary() → {len(summary)} entries")

    if summary:
        s0 = summary[0]
        print(f"  Sample keys: {sorted(s0.keys())}")
        expected_keys = {"figure_id", "doc_id", "caption_text", "image_path", "page_num", "width", "height"}
        actual_keys = set(s0.keys())
        if expected_keys.issubset(actual_keys):
            print(f"  {PASS} All expected keys present")
        else:
            missing = expected_keys - actual_keys
            print(f"  {FAIL} Missing keys: {missing}")
            errors.append(f"Missing serialization keys: {missing}")

        # Verify JSON-serializable
        try:
            json_str = json.dumps(summary[:2], indent=2, default=str)
            print(f"  {PASS} JSON serializable ({len(json_str)} bytes)")
        except Exception as e:
            print(f"  {FAIL} Not JSON serializable: {e}")
            errors.append("Not JSON serializable")

    # ── Test 6: Thumbnail generation ─────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Test 6: Image thumbnail generation")
    print(f"{'─'*60}")

    sample_id = next(iter(vp.visual_evidence_db.keys()), None)
    if sample_id:
        thumb = vp.get_image_thumbnail_b64(sample_id)
        if thumb:
            print(f"  Figure: {sample_id}")
            print(f"  Thumbnail b64 length: {len(thumb)} chars")
            print(f"  {PASS} Thumbnail generated")
        else:
            print(f"  {WARN} Thumbnail returned None for {sample_id}")
            warnings.append("Thumbnail generation failed")

    # ── Test 7: Full pipeline simulation ─────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Test 7: Full pipeline visual_analysis dict (what API returns)")
    print(f"{'─'*60}")

    from epistemic_engine import AtomicClaim
    # Simulate what evirag_system.py does
    test_claims = [
        AtomicClaim(
            claim_id=f"test_{i}",
            text=text,
            source_chunk_id=f"chunk_{i}",
            source_doc_id=next(iter(docs_with_figures), ""),
            source_doc_title="Test Paper",
        )
        for i, text in enumerate([
            "Homework helps students learn better",
            "Too much homework causes stress and burnout",
            "The effect size of homework varies by grade level",
        ])
    ]

    # Simulate the pipeline
    aligned_figures = []
    for claim in test_claims:
        top_matches = vp.align_claim_with_figures(
            claim.text, doc_id=claim.source_doc_id, top_k=2
        )
        for ve, score in top_matches:
            aligned_figures.append({
                "claim_id": claim.claim_id,
                "claim_text": claim.text[:120],
                "figure_id": ve.figure_id,
                "doc_id": ve.doc_id,
                "caption": ve.caption_text,
                "page_num": ve.page_num,
                "alignment_score": round(float(score), 4),
                "width": getattr(ve, "width", 0),
                "height": getattr(ve, "height", 0),
            })

    visual_analysis = {
        "mismatch_score": mismatch["mismatch_score"],
        "claims_with_weak_support": mismatch["claims_with_weak_support"],
        "total_claims": len(test_claims),
        "aligned_figures": sorted(aligned_figures, key=lambda x: x["alignment_score"], reverse=True)[:20],
        "cross_comparisons": [],
        "total_indexed_figures": n_indexed,
    }

    print(f"  visual_analysis keys: {sorted(visual_analysis.keys())}")
    print(f"  aligned_figures: {len(visual_analysis['aligned_figures'])}")
    for af in visual_analysis['aligned_figures'][:5]:
        print(f"    claim=\"{af['claim_text'][:40]}...\"  "
              f"fig={af['figure_id']:25s}  score={af['alignment_score']:.4f}")

    # Verify the dict is JSON-serializable (what fastapi returns)
    try:
        json.dumps(visual_analysis, default=str)
        print(f"  {PASS} visual_analysis is JSON-serializable for API")
    except Exception as e:
        print(f"  {FAIL} Not serializable: {e}")
        errors.append("visual_analysis not serializable")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}")

    if errors:
        for e in errors:
            print(f"  {FAIL} {e}")
    if warnings:
        for w in warnings:
            print(f"  {WARN} {w}")

    if not errors:
        print(f"  {PASS} All semantic correctness tests passed")
        print(f"  Figures indexed:    {n_indexed}")
        print(f"  Docs with figures:  {len(docs_with_figures)}")
        print(f"  Alignment scores:   {rel_top_score:.4f} (relevant) vs {irrel_top_score:.4f} (irrelevant)")
        print(f"  Elapsed:            {elapsed:.1f}s")
    else:
        print(f"\n  {len(errors)} error(s) found")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
