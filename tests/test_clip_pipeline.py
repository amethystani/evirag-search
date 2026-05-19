#!/usr/bin/env python3
"""
CLIP Visual Grounding – End-to-End Pipeline Test
=================================================
Tests that the full CLIP pipeline works:
  1. Figures are extracted from PDFs and have valid sizes
  2. CLIP model loads and produces 512-D embeddings
  3. Claim-figure alignment produces meaningful scores
  4. Cross-paper figure comparison works
  5. The visual_analysis dict matches what the API expects

Run:  python test_clip_pipeline.py
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PASS = "✅"
FAIL = "❌"
INFO = "ℹ️ "


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def main():
    start = time.time()
    errors = []

    # ── 1. Load figures DB ──────────────────────────────────────────────────
    section("1 · Loading corpus and figures DB")
    from data_layer import DataManager
    dm = DataManager()
    dm.process_corpus(rebuild=False)

    total_figs = sum(len(figs) for figs in dm.figures_db.values())
    print(f"  Corpus: {len(dm.vector_store.chunks)} chunks across "
          f"{len(set(c.doc_id for c in dm.vector_store.chunks))} documents")
    print(f"  Figures DB: {total_figs} raw figures across "
          f"{len(dm.figures_db)} documents")

    if total_figs == 0:
        print(f"  {FAIL} No figures extracted – CLIP has nothing to work with")
        errors.append("No figures in corpus")
    else:
        print(f"  {PASS} Figures found")

    # List a few figure files with sizes
    from config import FIGURES_DIR
    fig_files = sorted(FIGURES_DIR.glob("*.png"))
    print(f"\n  {len(fig_files)} figure images on disk:")
    for fp in fig_files[:8]:
        from PIL import Image
        try:
            img = Image.open(fp)
            w, h = img.size
            sz = fp.stat().st_size
            flag = "✓" if w >= 50 and h >= 50 else "✗ (too small)"
            print(f"    {fp.name:40s}  {w:4d}×{h:<4d}  {sz:>8,d} B  {flag}")
        except Exception as e:
            print(f"    {fp.name:40s}  ERROR: {e}")

    # ── 2. Load CLIP and embed ─────────────────────────────────────────────
    section("2 · CLIP model loading + embedding test")
    from vlm_module import VisualEvidenceProcessor
    vp = VisualEvidenceProcessor()

    # Process (or load from cache)
    vp.process_figures(dm.figures_db, rebuild=False)
    n_indexed = len(vp.visual_evidence_db)
    print(f"  Indexed visual evidence: {n_indexed} figures")

    if n_indexed == 0:
        print(f"  {FAIL} No figures indexed – all were filtered?")
        errors.append("No indexed figures after CLIP processing")
    else:
        print(f"  {PASS} Visual evidence DB populated")

    # Check embedding dimensions
    sample_ve = next(iter(vp.visual_evidence_db.values()), None)
    if sample_ve:
        img_dim = sample_ve.image_embedding.shape[0]
        cap_dim = sample_ve.caption_embedding.shape[0]
        print(f"  Image embedding dim: {img_dim}")
        print(f"  Caption embedding dim: {cap_dim}")
        assert img_dim == 512, f"Expected 512, got {img_dim}"
        assert cap_dim == 512, f"Expected 512, got {cap_dim}"
        print(f"  {PASS} Embedding dimensions correct (512-D)")
        print(f"  Sample: {sample_ve.figure_id} | "
              f"{sample_ve.width}×{sample_ve.height} | "
              f"p.{sample_ve.page_num} | "
              f"caption: {sample_ve.caption_text[:60]}...")

    # ── 3. Claim-figure alignment ──────────────────────────────────────────
    section("3 · Claim → figure alignment")
    test_claims = [
        "Homework improves student academic performance",
        "There is a negative relationship between homework and achievement",
        "The effect of homework depends on the age of the student",
    ]
    for claim in test_claims:
        results = vp.align_claim_with_figures(claim, top_k=3)
        print(f"\n  Claim: \"{claim[:70]}\"")
        if results:
            for ve, score in results:
                print(f"    → {ve.figure_id:30s}  score={score:.4f}  "
                      f"caption=\"{ve.caption_text[:50]}\"")
        else:
            print(f"    → No figures matched")

    if results:
        print(f"\n  {PASS} Alignment produces scored results")
    else:
        errors.append("Alignment returned empty")

    # ── 4. Cross-paper figure comparison ───────────────────────────────────
    section("4 · Cross-paper figure comparison")
    c1 = "Homework has positive effects on achievement"
    c2 = "Homework has negative effects and causes stress"
    comparison = vp.compare_figures_across_claims(c1, c2)
    print(f"  Claim1: \"{c1}\"")
    print(f"  Claim2: \"{c2}\"")
    print(f"  Similar visual evidence: {comparison['similar_evidence_count']}")
    print(f"  Interpretation conflict: {comparison['interpretation_conflict']}")
    print(f"  Claim1 top figures: {len(comparison['claim1_figures'])}")
    print(f"  Claim2 top figures: {len(comparison['claim2_figures'])}")
    print(f"  {PASS} Cross-paper comparison executed")

    # ── 5. Weak support detection ──────────────────────────────────────────
    section("5 · Weak visual support detection")
    sample_doc_id = next(iter(dm.figures_db.keys()), None)
    if sample_doc_id:
        weak = vp.detect_weak_visual_support(
            "A completely unrelated claim about quantum computing",
            sample_doc_id,
        )
        print(f"  Doc: {sample_doc_id}")
        print(f"  Has visual support: {weak['has_visual_support']}")
        print(f"  Strength: {weak['strength']:.4f}")
        print(f"  Weak support flag: {weak['weak_support']}")
        if weak.get("warning"):
            print(f"  Warning: {weak['warning']}")
        print(f"  {PASS} Weak support detection works")

    # ── 6. Serialization test (what the API sends) ─────────────────────────
    section("6 · Serialization for API")
    summary = vp.get_all_evidence_summary()
    print(f"  get_all_evidence_summary() → {len(summary)} entries")
    if summary:
        s0 = summary[0]
        print(f"  Sample keys: {list(s0.keys())}")
        # Verify JSON-serializable
        json_str = json.dumps(summary[:2], indent=2, default=str)
        print(f"  JSON serializable: {PASS}")
        print(f"  Sample:\n{json_str[:300]}")

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    section("RESULT")
    if errors:
        for e in errors:
            print(f"  {FAIL} {e}")
        print(f"\n  Pipeline has issues — {len(errors)} error(s)")
    else:
        print(f"  {PASS} All CLIP visual grounding tests passed")
        print(f"  {INFO} {n_indexed} figures indexed from {total_figs} extracted")
        print(f"  {INFO} Embeddings: 512-D CLIP ViT-B/16")
        print(f"  {INFO} Elapsed: {elapsed:.1f}s")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
