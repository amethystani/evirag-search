#!/usr/bin/env python3
"""
Quick test to rebuild visual cache with progress output
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from vlm_module import VisualEvidenceProcessor
from data_layer import DataManager
import numpy as np

print('🔧 Rebuilding Visual Evidence Cache')
print('='*80)

dm = DataManager()
dm.process_corpus(rebuild=False)

print('\nProcessing figures and generating CLIP embeddings...')

vp = VisualEvidenceProcessor()
vp.process_figures(dm.figures_db, rebuild=True)  # REBUILD!!!

print('\n✅ Cache rebuilt! Verifying embeddings...')

# Verify first few
valid_count = 0
zero_count = 0

for i, (fig_id, ve) in enumerate(list(vp.visual_evidence_db.items())[:5]):
    cap_norm = np.linalg.norm(ve.caption_embedding)
    img_norm = np.linalg.norm(ve.image_embedding)
    
    if cap_norm > 0 and img_norm > 0:
        valid_count += 1
        print(f'✅ {fig_id}: cap_norm={cap_norm:.4f}, img_norm={img_norm:.4f}')
    else:
        zero_count += 1
        print(f'❌ {fig_id}: ZERO EMBEDDINGS!')

print(f'\n📊 Results: {valid_count} valid, {zero_count} invalid')

if valid_count > 0:
    print('\n✅ Visual evidence cache successfully rebuilt!')
else:
    print('\n❌ Cache rebuild failed - still have zero embeddings!')
