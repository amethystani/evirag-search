"""
EVIRAG Visual Evidence Module
Handles figure extraction, CLIP embeddings, and visual-text alignment.

CLIP (Contrastive Language-Image Pre-training) maps images and text into a
shared 512-D embedding space.  This module uses CLIP **only** for alignment
measurement — it never generates text.  The key capabilities are:

  1. Embed all figures extracted from the PDF corpus into CLIP space.
  2. Embed textual claims into the same space.
  3. Compute cosine similarity to find which figures best support which claims.
  4. Detect visual–text mismatch (a claim whose paper's figures don't match it).
  5. Cross-compare figures across papers to spot interpretation conflicts.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from dataclasses import dataclass, asdict
import pickle
import base64
import io

from config import EMBEDDING_CONFIG, CACHE_DIR, EMBEDDINGS_DIR

# Minimum image dimensions to consider (skip tiny icons, 1-pixel spacers etc.)
MIN_FIGURE_WIDTH = 50
MIN_FIGURE_HEIGHT = 50


@dataclass
class VisualEvidence:
    """Represents visual evidence with embeddings"""
    figure_id: str
    doc_id: str
    image_embedding: np.ndarray
    caption_embedding: np.ndarray
    caption_text: str
    image_path: Path
    page_num: int
    width: int = 0
    height: int = 0

    def to_dict(self) -> Dict:
        """Serialise for API / frontend (embeddings excluded)."""
        return {
            "figure_id": self.figure_id,
            "doc_id": self.doc_id,
            "caption_text": self.caption_text,
            "image_path": str(self.image_path),
            "page_num": self.page_num,
            "width": self.width,
            "height": self.height,
        }


class CLIPEmbedder:
    """Generate CLIP embeddings for images and text"""
    
    def __init__(self):
        self.model_name = EMBEDDING_CONFIG["clip_model"]
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.embedding_dim = EMBEDDING_CONFIG["clip_dim"]
    
    def load_model(self):
        """Lazy load CLIP model"""
        if self.model is None:
            print(f"Loading CLIP model: {self.model_name}")
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
    
    def embed_image(self, image_path: Path) -> Tuple[np.ndarray, int, int]:
        """Generate CLIP embedding for an image.

        Returns:
            (embedding, width, height).  On error returns (zeros, 0, 0).
        """
        self.load_model()
        
        try:
            image = Image.open(image_path).convert("RGB")
            w, h = image.size

            # Skip tiny / corrupt images
            if w < MIN_FIGURE_WIDTH or h < MIN_FIGURE_HEIGHT:
                return np.zeros(self.embedding_dim), w, h
            
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Extract tensor from BaseModelOutputWithPooling if needed
                if hasattr(image_features, 'pooler_output'):
                    image_features = image_features.pooler_output
                # Normalize using torch.linalg.norm to avoid attribute errors
                image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)
            
            return image_features.cpu().numpy()[0], w, h
        
        except Exception as e:
            print(f"Error embedding image {image_path}: {e}")
            return np.zeros(self.embedding_dim), 0, 0
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate CLIP embedding for text"""
        self.load_model()
        
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                # Extract tensor from BaseModelOutputWithPooling if needed
                if hasattr(text_features, 'pooler_output'):
                    text_features = text_features.pooler_output
                # Normalize using torch.linalg.norm to avoid attribute errors
                text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)
            
            return text_features.cpu().numpy()[0]
        
        except Exception as e:
            print(f"Error embedding text: {e}")
            return np.zeros(self.embedding_dim)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        # Both embeddings should already be normalized
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)


class VisualEvidenceProcessor:
    """Process figures and create visual evidence database"""
    
    def __init__(self):
        self.clip_embedder = CLIPEmbedder()
        self.visual_evidence_db = {}  # figure_id -> VisualEvidence
        self.cache_path = EMBEDDINGS_DIR / "visual_evidence.pkl"
    
    def process_figures(self, figures_db: Dict, rebuild: bool = False):
        """Process all figures and generate embeddings"""
        
        # Load cached if available
        if not rebuild and self.load_cache():
            print("Using cached visual embeddings")
            return
        
        print("Processing visual evidence...")
        
        all_figures = []
        for doc_id, figures in figures_db.items():
            all_figures.extend(figures)
        
        print(f"Generating CLIP embeddings for {len(all_figures)} figures...")
        
        skipped = 0
        for figure in all_figures:
            # Generate image embedding (with size check)
            image_emb, w, h = self.clip_embedder.embed_image(figure.image_path)

            # Skip figures that are too small or corrupt
            if w < MIN_FIGURE_WIDTH or h < MIN_FIGURE_HEIGHT:
                skipped += 1
                continue
            
            # Generate caption embedding
            caption_emb = self.clip_embedder.embed_text(figure.caption)
            
            # Create visual evidence
            visual_evidence = VisualEvidence(
                figure_id=figure.figure_id,
                doc_id=figure.doc_id,
                image_embedding=image_emb,
                caption_embedding=caption_emb,
                caption_text=figure.caption,
                image_path=figure.image_path,
                page_num=figure.page_num,
                width=w,
                height=h,
            )
            
            self.visual_evidence_db[figure.figure_id] = visual_evidence
        
        # Save cache
        self.save_cache()
        if skipped:
            print(f"  Skipped {skipped} tiny/corrupt images")
        print(f"Processed {len(self.visual_evidence_db)} usable figures")
    
    def align_claim_with_figures(
        self, 
        claim_text: str, 
        doc_id: Optional[str] = None,
        top_k: int = 5
    ) -> List[Tuple[VisualEvidence, float]]:
        """
        Find figures that align with a textual claim
        Returns top-k figures with alignment scores
        """
        # Generate claim embedding
        claim_embedding = self.clip_embedder.embed_text(claim_text)
        
        # Filter figures by doc_id if provided
        candidates = self.visual_evidence_db.values()
        if doc_id is not None:
            candidates = [ve for ve in candidates if ve.doc_id == doc_id]
        
        # Compute similarities
        results = []
        for visual_evidence in candidates:
            # Compute similarity with image
            img_sim = self.clip_embedder.compute_similarity(
                claim_embedding, 
                visual_evidence.image_embedding
            )
            
            # Compute similarity with caption
            cap_sim = self.clip_embedder.compute_similarity(
                claim_embedding,
                visual_evidence.caption_embedding
            )
            
            # Combined score (weighted average)
            combined_score = 0.6 * img_sim + 0.4 * cap_sim
            
            results.append((visual_evidence, combined_score))
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def compare_figures_across_claims(
        self,
        claim1: str,
        claim2: str,
        doc_id1: Optional[str] = None,
        doc_id2: Optional[str] = None
    ) -> Dict:
        """
        Compare visual evidence supporting two different claims
        Useful for detecting contradictory interpretations of similar evidence
        """
        # Get figures for each claim
        figures1 = self.align_claim_with_figures(claim1, doc_id1, top_k=3)
        figures2 = self.align_claim_with_figures(claim2, doc_id2, top_k=3)
        
        # Compute cross-similarities between figure sets
        cross_similarities = []
        for ve1, score1 in figures1:
            for ve2, score2 in figures2:
                # Compare the images directly
                fig_similarity = self.clip_embedder.compute_similarity(
                    ve1.image_embedding,
                    ve2.image_embedding
                )
                cross_similarities.append({
                    "figure1": ve1.figure_id,
                    "figure2": ve2.figure_id,
                    "visual_similarity": fig_similarity,
                    "claim1_alignment": score1,
                    "claim2_alignment": score2
                })
        
        # Identify cases of similar figures supporting contradictory claims
        similar_evidence = [
            cs for cs in cross_similarities 
            if cs["visual_similarity"] > 0.7  # High visual similarity threshold
        ]
        
        return {
            "claim1_figures": [(ve.figure_id, score) for ve, score in figures1],
            "claim2_figures": [(ve.figure_id, score) for ve, score in figures2],
            "similar_evidence_count": len(similar_evidence),
            "similar_evidence": similar_evidence,
            "interpretation_conflict": len(similar_evidence) > 0
        }
    
    def detect_weak_visual_support(
        self,
        claim: str,
        doc_id: str,
        threshold: float = 0.5
    ) -> Dict:
        """
        Detect when a claim has weak visual support in its source document
        This helps identify overstated textual conclusions
        """
        aligned_figures = self.align_claim_with_figures(claim, doc_id, top_k=5)
        
        if not aligned_figures:
            return {
                "has_visual_support": False,
                "strength": 0.0,
                "weak_support": True,  # No visual evidence = weak support
                "warning": "No visual evidence found in source document"
            }
        
        best_score = aligned_figures[0][1]
        avg_score = np.mean([score for _, score in aligned_figures])
        
        weak_support = best_score < threshold
        
        return {
            "has_visual_support": True,
            "strength": avg_score,
            "best_alignment": best_score,
            "weak_support": weak_support,
            "warning": "Weak visual support - claim may be overstated" if weak_support else None,
            "supporting_figures": [
                {
                    "figure_id": ve.figure_id,
                    "score": score,
                    "caption": ve.caption_text
                }
                for ve, score in aligned_figures
            ]
        }
    
    def get_visual_evidence_for_claim(
        self,
        claim: str,
        doc_id: Optional[str] = None
    ) -> Optional[VisualEvidence]:
        """Get the best visual evidence for a claim"""
        aligned = self.align_claim_with_figures(claim, doc_id, top_k=1)
        if aligned:
            return aligned[0][0]
        return None
    
    def save_cache(self):
        """Save visual evidence database to cache"""
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.visual_evidence_db, f)
    
    def load_cache(self) -> bool:
        """Load visual evidence database from cache"""
        if not self.cache_path.exists():
            return False
        
        try:
            with open(self.cache_path, 'rb') as f:
                self.visual_evidence_db = pickle.load(f)
            print(f"Loaded {len(self.visual_evidence_db)} visual evidence entries from cache")
            return True
        except Exception as e:
            print(f"Error loading visual evidence cache: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────
    # Utility: build a lightweight JSON-ready summary of all visual evidence
    # ──────────────────────────────────────────────────────────────────────

    def get_all_evidence_summary(self) -> List[Dict]:
        """Return a JSON-serialisable list of all indexed visual evidence."""
        results = []
        for ve in self.visual_evidence_db.values():
            if hasattr(ve, 'to_dict'):
                results.append(ve.to_dict())
            else:
                results.append({
                    "figure_id": ve.figure_id,
                    "doc_id": ve.doc_id,
                    "caption_text": ve.caption_text,
                    "image_path": str(ve.image_path),
                    "page_num": ve.page_num,
                    "width": getattr(ve, 'width', 0),
                    "height": getattr(ve, 'height', 0),
                })
        return results

    def get_image_thumbnail_b64(self, figure_id: str, max_px: int = 200) -> Optional[str]:
        """Return a base64-encoded JPEG thumbnail for a figure."""
        ve = self.visual_evidence_db.get(figure_id)
        if ve is None:
            return None
        try:
            img = Image.open(ve.image_path).convert("RGB")
            img.thumbnail((max_px, max_px))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return None


class VisualDisagreementAnalyzer:
    """
    Analyze disagreement using visual evidence
    CLIP is used ONLY for alignment measurement, never for generation
    """
    
    def __init__(self, visual_processor: VisualEvidenceProcessor):
        self.visual_processor = visual_processor
    
    def analyze_visual_text_mismatch(
        self,
        claims: List[Dict],  # Each dict has 'text', 'doc_id', 'stance'
    ) -> Dict:
        """
        Measure visual-text mismatch across competing claims
        Returns metrics for confidence calibration
        """
        mismatches = []
        
        for claim_data in claims:
            claim_text = claim_data["text"]
            doc_id = claim_data["doc_id"]
            stance = claim_data.get("stance", "neutral")
            
            # Check visual support
            visual_support = self.visual_processor.detect_weak_visual_support(
                claim_text, doc_id
            )
            
            if visual_support["weak_support"]:
                mismatches.append({
                    "claim": claim_text,
                    "doc_id": doc_id,
                    "stance": stance,
                    "visual_strength": visual_support["strength"],
                    "mismatch_severity": 1.0 - visual_support["strength"]
                })
        
        # Compute overall mismatch score
        if mismatches:
            avg_mismatch = np.mean([m["mismatch_severity"] for m in mismatches])
        else:
            avg_mismatch = 0.0
        
        return {
            "mismatch_score": avg_mismatch,
            "claims_with_weak_support": len(mismatches),
            "total_claims": len(claims),
            "mismatches": mismatches
        }
    
    def compute_visual_evidence_strength(
        self,
        supporting_claims: List[Dict],
        contradicting_claims: List[Dict]
    ) -> Dict:
        """
        Compute visual evidence strength for competing viewpoints
        Used in confidence calibration
        """
        support_strengths = []
        for claim_data in supporting_claims:
            visual_support = self.visual_processor.detect_weak_visual_support(
                claim_data["text"],
                claim_data["doc_id"]
            )
            support_strengths.append(visual_support["strength"])
        
        contradict_strengths = []
        for claim_data in contradicting_claims:
            visual_support = self.visual_processor.detect_weak_visual_support(
                claim_data["text"],
                claim_data["doc_id"]
            )
            contradict_strengths.append(visual_support["strength"])
        
        return {
            "support_visual_strength": np.mean(support_strengths) if support_strengths else 0.0,
            "contradict_visual_strength": np.mean(contradict_strengths) if contradict_strengths else 0.0,
            "visual_evidence_balance": abs(
                np.mean(support_strengths) - np.mean(contradict_strengths)
            ) if support_strengths and contradict_strengths else 0.0
        }




if __name__ == "__main__":
    # Test visual evidence processing
    from data_layer import DataManager
    
    print("Testing Visual Evidence Module")
    print("=" * 60)
    
    # Load data
    data_manager = DataManager()
    data_manager.process_corpus(rebuild=False)
    
    # Process visual evidence
    visual_processor = VisualEvidenceProcessor()
    visual_processor.process_figures(data_manager.figures_db, rebuild=False)
    
    # Test claim-figure alignment
    test_claim = "Neural networks show improved performance with increased depth"
    print(f"\nTest claim: {test_claim}")
    
    aligned_figures = visual_processor.align_claim_with_figures(test_claim, top_k=3)
    print(f"\nTop aligned figures:")
    for ve, score in aligned_figures:
        print(f"  - {ve.figure_id}: {score:.3f} - {ve.caption_text[:100]}")
