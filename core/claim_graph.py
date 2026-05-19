"""
Offline claim graph for fast EVIRAG query-time retrieval.

This module builds a sparse claim graph once at index time, then serves
query-time subgraphs without running pairwise NLI online.
"""

import json
import os
import pickle
import re
import hashlib
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from config import CLAIM_GRAPH_CONFIG, SYSTEM_CONSTRAINTS, VECTOR_STORE_DIR
from epistemic_engine import AtomicClaim, ClaimRelationship, NLIReasoner


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "this", "to", "was", "were", "with", "we", "our", "can",
    "may", "might", "than", "into", "using", "use", "used", "these", "those",
}

_POSITIVE_CUES = {
    "benefit", "beneficial", "better", "effective", "gain", "gains", "higher",
    "improve", "improved", "improves", "increase", "increases", "outperform",
    "positive", "robust", "significant", "strong", "superior", "supports",
}

_NEGATIVE_CUES = {
    "decrease", "decreases", "drop", "drops", "fail", "fails", "failure",
    "harm", "harms", "ineffective", "inferior", "insignificant", "lower",
    "negative", "no", "not", "poor", "reduce", "reduced", "reduces", "weak",
    "worse", "without",
}

_OPPOSITE_CUE_PAIRS = [
    ("increase", "decrease"),
    ("improve", "harm"),
    ("effective", "ineffective"),
    ("higher", "lower"),
    ("outperform", "inferior"),
    ("significant", "insignificant"),
    ("positive", "negative"),
]


def _can_render_live_progress() -> bool:
    stream = getattr(sys, "stderr", None)
    try:
        return bool(stream) and stream.isatty()
    except Exception:
        return False


class ClaimGraphStore:
    """Persistent sparse graph over extracted claim nodes."""

    def __init__(self, embedding_engine, backend: str = "local"):
        self.embedding_engine = embedding_engine
        self.backend = backend
        self.index = None
        self.claims: List[AtomicClaim] = []
        self.relationships: List[ClaimRelationship] = []
        self.claim_embeddings: Optional[np.ndarray] = None
        self.chunk_to_claim_ids: Dict[str, List[str]] = {}
        self.claim_id_to_idx: Dict[str, int] = {}
        self.claim_map: Dict[str, AtomicClaim] = {}
        self.adjacency: Dict[str, List[ClaimRelationship]] = defaultdict(list)
        self._pair_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._verifier = None
        self.embedding_dim = embedding_engine.embedding_dim
        self.index_path = VECTOR_STORE_DIR / "claim_index.faiss"
        self.claims_path = VECTOR_STORE_DIR / "claims.pkl"
        self.relationships_path = VECTOR_STORE_DIR / "claim_relationships.pkl"
        self.claim_embeddings_path = VECTOR_STORE_DIR / "claim_embeddings.npy"
        self.chunk_map_path = VECTOR_STORE_DIR / "chunk_to_claims.json"
        self.pair_cache_path = VECTOR_STORE_DIR / "claim_pair_cache.json"
        self.metadata_path = VECTOR_STORE_DIR / "claim_graph_metadata.json"

    def is_ready(self) -> bool:
        return self.index is not None and len(self.claims) > 0

    def load(self) -> bool:
        """Load an existing claim graph if present."""
        required = [
            self.index_path,
            self.claims_path,
            self.relationships_path,
            self.claim_embeddings_path,
            self.chunk_map_path,
            self.metadata_path,
        ]
        if not all(path.exists() for path in required):
            return False

        try:
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)
            if metadata.get("config") != CLAIM_GRAPH_CONFIG:
                print("Claim graph config changed; rebuilding offline graph.")
                return False

            self.index = faiss.read_index(str(self.index_path))
            with open(self.claims_path, "rb") as f:
                self.claims = pickle.load(f)
            with open(self.relationships_path, "rb") as f:
                self.relationships = pickle.load(f)
            self.claim_embeddings = np.load(self.claim_embeddings_path)
            with open(self.chunk_map_path, "r") as f:
                self.chunk_to_claim_ids = json.load(f)
            if self.pair_cache_path.exists():
                with open(self.pair_cache_path, "r") as f:
                    self._pair_cache = json.load(f)
            self._rebuild_runtime_maps()
            print(
                f"Loaded claim graph: {len(self.claims)} claims, "
                f"{len(self.relationships)} edges"
            )
            return True
        except Exception as exc:
            print(f"Error loading claim graph: {exc}")
            return False

    def build_from_chunks(self, chunks: List[Any], rebuild: bool = False):
        """Build a sparse claim graph from chunk text."""
        if not rebuild and self.load():
            return

        print("Building offline claim graph...")
        self.embedding_engine.load_model()
        self._load_pair_cache()
        claims: List[AtomicClaim] = []
        chunk_to_claim_ids: Dict[str, List[str]] = {}
        seen_claims = set()

        for chunk in chunks:
            chunk_claims = self._extract_claims_from_chunk(chunk)
            retained_ids = []
            for claim in chunk_claims:
                dedup_key = (claim.source_doc_id, self._normalize_text(claim.text))
                if dedup_key in seen_claims:
                    continue
                seen_claims.add(dedup_key)
                claims.append(claim)
                retained_ids.append(claim.claim_id)
            chunk_to_claim_ids[chunk.chunk_id] = retained_ids

        if not claims:
            print("Claim graph skipped: no claims extracted")
            self.index = None
            self.claims = []
            self.relationships = []
            self.claim_embeddings = None
            self.chunk_to_claim_ids = {}
            return

        texts = [claim.text for claim in claims]
        claim_embeddings = self.embedding_engine.model.encode(
            texts,
            show_progress_bar=_can_render_live_progress(),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(claim_embeddings.astype("float32"))

        self.claims = claims
        self.claim_embeddings = claim_embeddings.astype("float32")
        self.chunk_to_claim_ids = chunk_to_claim_ids
        self.relationships = self._build_sparse_relationships(claim_embeddings)
        self._rebuild_runtime_maps()
        self.save()
        print(
            f"Claim graph built: {len(self.claims)} claims, "
            f"{len(self.relationships)} edges"
        )

    def save(self):
        """Persist graph artifacts to disk."""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.claims_path, "wb") as f:
            pickle.dump(self.claims, f)
        with open(self.relationships_path, "wb") as f:
            pickle.dump(self.relationships, f)
        np.save(self.claim_embeddings_path, self.claim_embeddings)
        with open(self.chunk_map_path, "w") as f:
            json.dump(self.chunk_to_claim_ids, f, indent=2)
        with open(self.pair_cache_path, "w") as f:
            json.dump(self._pair_cache, f, indent=2)
        metadata = {
            "num_claims": len(self.claims),
            "num_edges": len(self.relationships),
            "embedding_dim": self.embedding_dim,
            "config": CLAIM_GRAPH_CONFIG,
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def query_subgraph(
        self,
        query: str,
        max_claims: Optional[int] = None,
    ) -> Tuple[List[AtomicClaim], List[ClaimRelationship], Dict[str, Any]]:
        """Retrieve a small claim-centric subgraph for a query."""
        if not self.is_ready():
            return [], [], {"mode": "claim_graph_missing"}

        max_claims = max_claims or SYSTEM_CONSTRAINTS["max_claims_per_query"]
        max_claims = min(max_claims, SYSTEM_CONSTRAINTS["max_claims_cloud"])
        seed_k = min(CLAIM_GRAPH_CONFIG["query_seed_k"], len(self.claims))

        query_embedding = self.embedding_engine.embed_query(query).reshape(1, -1).astype("float32")
        distances, indices = self.index.search(query_embedding, seed_k)

        selected_scores: Dict[str, float] = {}
        ordered_ids: List[str] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.claims):
                continue
            claim = self.claims[idx]
            score = float(1.0 / (1.0 + dist))
            ordered_ids.append(claim.claim_id)
            selected_scores[claim.claim_id] = score

        if not ordered_ids:
            return [], [], {"mode": "claim_graph_empty"}

        expanded_ids = list(ordered_ids)
        for claim_id in list(ordered_ids):
            neighbors = sorted(
                self.adjacency.get(claim_id, []),
                key=lambda rel: rel.confidence,
                reverse=True,
            )[:CLAIM_GRAPH_CONFIG["query_neighbor_expansion"]]
            for rel in neighbors:
                other_id = rel.claim2_id if rel.claim1_id == claim_id else rel.claim1_id
                if other_id not in selected_scores:
                    seed_boost = selected_scores[claim_id] * rel.confidence
                    selected_scores[other_id] = seed_boost
                    expanded_ids.append(other_id)
                if len(expanded_ids) >= max_claims:
                    break
            if len(expanded_ids) >= max_claims:
                break

        ranked_ids = sorted(
            set(expanded_ids),
            key=lambda cid: selected_scores.get(cid, 0.0),
            reverse=True,
        )[:max_claims]
        ranked_set = set(ranked_ids)

        claims = [self.claim_map[cid] for cid in ranked_ids if cid in self.claim_map]
        relationships = [
            rel for rel in self.relationships
            if rel.claim1_id in ranked_set and rel.claim2_id in ranked_set
        ][:CLAIM_GRAPH_CONFIG["max_query_edges"]]

        support_edges = len([rel for rel in relationships if rel.relationship == "supports"])
        contradiction_edges = len([rel for rel in relationships if rel.relationship == "contradicts"])

        return claims, relationships, {
            "mode": "fast_claim_graph",
            "seed_claims": len(ordered_ids),
            "selected_claims": len(claims),
            "support_edges": support_edges,
            "contradiction_edges": contradiction_edges,
        }

    def get_embedding_lookup(self, claim_ids: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Return persisted claim embeddings keyed by claim id."""
        if self.claim_embeddings is None:
            return {}
        if claim_ids is None:
            claim_ids = [claim.claim_id for claim in self.claims]
        result = {}
        for claim_id in claim_ids:
            idx = self.claim_id_to_idx.get(claim_id)
            if idx is not None and idx < len(self.claim_embeddings):
                result[claim_id] = self.claim_embeddings[idx]
        return result

    def _rebuild_runtime_maps(self):
        self.claim_id_to_idx = {
            claim.claim_id: idx for idx, claim in enumerate(self.claims)
        }
        self.claim_map = {claim.claim_id: claim for claim in self.claims}
        self.adjacency = defaultdict(list)
        for rel in self.relationships:
            self.adjacency[rel.claim1_id].append(rel)
            self.adjacency[rel.claim2_id].append(rel)

    def _extract_claims_from_chunk(self, chunk: Any) -> List[AtomicClaim]:
        if getattr(chunk, "section", "") == "references":
            return []

        sentence_candidates = self._split_sentences(chunk.text)
        scored = []
        for sentence in sentence_candidates:
            cleaned = self._clean_sentence(sentence)
            if not self._looks_like_claim(cleaned):
                continue
            scored.append((self._sentence_score(cleaned), cleaned))

        if not scored:
            fallback = [
                self._clean_sentence(sentence)
                for sentence in sentence_candidates
                if len(sentence.split()) >= CLAIM_GRAPH_CONFIG["min_sentence_words"]
            ]
            scored = [(len(sentence.split()), sentence) for sentence in fallback]

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[:CLAIM_GRAPH_CONFIG["max_claims_per_chunk"]]
        claims = []
        for idx, (_, sentence) in enumerate(selected):
            claim_id = self._stable_claim_id(chunk.doc_id, chunk.chunk_id, idx, sentence)
            claims.append(
                AtomicClaim(
                    claim_id=claim_id,
                    text=sentence,
                    source_chunk_id=chunk.chunk_id,
                    source_doc_id=chunk.doc_id,
                    source_doc_title=chunk.metadata.get("title", "Unknown"),
                    context=chunk.text[:400],
                    section=getattr(chunk, "section", None),
                    source_path=chunk.metadata.get("path"),
                    page_num=getattr(chunk, "page_num", None),
                )
            )
        return claims

    def _build_sparse_relationships(self, claim_embeddings: np.ndarray) -> List[ClaimRelationship]:
        k = min(CLAIM_GRAPH_CONFIG["candidate_neighbors"] + 1, len(self.claims))
        if k <= 1:
            return []
        relationships: List[ClaimRelationship] = []
        seen_pairs = set()
        verifier = self._get_verifier()
        verification_budget = self._get_verification_budget()

        for src_idx, src_embedding in enumerate(claim_embeddings):
            src_claim = self.claims[src_idx]
            distances, indices = self.index.search(
                src_embedding.reshape(1, -1).astype("float32"),
                k,
            )
            for dist, dst_idx in zip(distances[0], indices[0]):
                if dst_idx < 0 or dst_idx == src_idx or dst_idx >= len(self.claims):
                    continue
                pair_key = tuple(sorted((src_idx, dst_idx)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                dst_claim = self.claims[dst_idx]
                similarity = float(1.0 / (1.0 + dist))
                relationship = self._classify_relationship(src_claim.text, dst_claim.text, similarity)
                if not relationship:
                    continue

                rel_type, confidence, reasoning = relationship
                cache_key = self._pair_cache_key(src_claim, dst_claim)
                if self._should_verify_candidate(rel_type, similarity, confidence) and verifier is not None and verification_budget > 0:
                    cached = self._pair_cache.get(cache_key)
                    if cached is not None:
                        verified_rel = self._relationship_from_cache(cached, src_claim, dst_claim)
                    else:
                        verified_rel = verifier.analyze_relationship(
                            src_claim,
                            dst_claim,
                            use_prefilter=False,
                        )
                        self._pair_cache[cache_key] = verified_rel.to_dict() if verified_rel else None
                        verification_budget -= 1

                    if verified_rel is not None:
                        rel_type = self._map_relationship_type(verified_rel.relationship)
                        confidence = verified_rel.confidence
                        reasoning = f"Offline NLI verified: {verified_rel.reasoning}"
                    if rel_type == "neutral":
                        continue

                relationships.append(
                    ClaimRelationship(
                        claim1_id=src_claim.claim_id,
                        claim2_id=dst_claim.claim_id,
                        relationship=rel_type,
                        confidence=confidence,
                        reasoning=reasoning,
                    )
                )

        return relationships

    def _classify_relationship(
        self,
        text1: str,
        text2: str,
        similarity: float,
    ) -> Optional[Tuple[str, float, str]]:
        if similarity < CLAIM_GRAPH_CONFIG["min_edge_similarity"]:
            return None

        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)
        if not tokens1 or not tokens2:
            return None

        overlap = len(tokens1 & tokens2) / max(1, min(len(tokens1), len(tokens2)))
        has_opposite_polarity = self._has_opposite_polarity(text1, text2)

        if has_opposite_polarity and similarity >= CLAIM_GRAPH_CONFIG["min_contradiction_similarity"]:
            confidence = round(min(0.92, 0.50 + 0.35 * similarity + 0.20 * overlap), 3)
            return "contradicts", confidence, "Sparse claim graph polarity conflict"

        if overlap >= 0.35 or similarity >= CLAIM_GRAPH_CONFIG["min_support_similarity"]:
            confidence = round(min(0.95, 0.52 + 0.30 * similarity + 0.20 * overlap), 3)
            return "supports", confidence, "Sparse claim graph semantic alignment"

        return None

    def _should_verify_candidate(self, rel_type: str, similarity: float, confidence: float) -> bool:
        mode = CLAIM_GRAPH_CONFIG.get("verify_edges", "auto")
        if mode == "none":
            return False
        if rel_type == "contradicts" and CLAIM_GRAPH_CONFIG.get("always_verify_contradictions", True):
            return True

        margin = CLAIM_GRAPH_CONFIG.get("verification_margin", 0.04)
        near_boundary = similarity <= (CLAIM_GRAPH_CONFIG["min_support_similarity"] + margin)
        low_confidence = confidence < 0.72
        return rel_type == "supports" and (near_boundary or low_confidence)

    def _get_verifier(self):
        if self._verifier is not None:
            return self._verifier

        backend = self._resolve_verification_backend()
        if backend is None:
            return None

        try:
            self._verifier = NLIReasoner(backend=backend)
        except Exception as exc:
            print(f"Claim graph verifier unavailable ({backend}): {exc}")
            self._verifier = None
        return self._verifier

    def _resolve_verification_backend(self) -> Optional[str]:
        mode = CLAIM_GRAPH_CONFIG.get("verify_edges", "auto")
        if mode == "none":
            return None
        if mode == "local":
            return "local"
        if mode == "cloud":
            return "cloud" if os.getenv("GROQ_API_KEY") else None
        if self.backend == "cloud" and os.getenv("GROQ_API_KEY"):
            return "cloud"
        if os.getenv("GROQ_API_KEY"):
            return "cloud"
        if CLAIM_GRAPH_CONFIG.get("verification_budget_local", 0) > 0:
            return "local"
        return None

    def _get_verification_budget(self) -> int:
        backend = self._resolve_verification_backend()
        if backend == "cloud":
            return int(CLAIM_GRAPH_CONFIG.get("verification_budget_cloud", 0))
        if backend == "local":
            return int(CLAIM_GRAPH_CONFIG.get("verification_budget_local", 0))
        return 0

    def _pair_cache_key(self, claim1: AtomicClaim, claim2: AtomicClaim) -> str:
        pair = sorted([claim1.claim_id, claim2.claim_id])
        return f"{pair[0]}::{pair[1]}"

    def _relationship_from_cache(
        self,
        cached: Optional[Dict[str, Any]],
        claim1: AtomicClaim,
        claim2: AtomicClaim,
    ) -> Optional[ClaimRelationship]:
        if not cached:
            return None
        return ClaimRelationship(
            claim1_id=claim1.claim_id,
            claim2_id=claim2.claim_id,
            relationship=cached.get("relationship", "neutral"),
            confidence=float(cached.get("confidence", 0.5)),
            reasoning=cached.get("reasoning", "Cached offline verifier result"),
        )

    def _map_relationship_type(self, rel_type: str) -> str:
        rel = (rel_type or "neutral").lower()
        mapping = {
            "entailment": "supports",
            "partial entailment": "supports",
            "partially entailment": "supports",
            "contradiction": "contradicts",
            "partial contradiction": "contradicts",
            "neutral": "neutral",
            "supports": "supports",
            "contradicts": "contradicts",
        }
        return mapping.get(rel, rel)

    def _load_pair_cache(self):
        if self.pair_cache_path.exists():
            try:
                with open(self.pair_cache_path, "r") as f:
                    self._pair_cache = json.load(f)
            except Exception:
                self._pair_cache = {}
        else:
            self._pair_cache = {}

    def _has_opposite_polarity(self, text1: str, text2: str) -> bool:
        score1 = self._polarity_score(text1)
        score2 = self._polarity_score(text2)
        if score1 == 0 or score2 == 0:
            for left, right in _OPPOSITE_CUE_PAIRS:
                left_in_1 = left in text1.lower()
                right_in_1 = right in text1.lower()
                left_in_2 = left in text2.lower()
                right_in_2 = right in text2.lower()
                if (left_in_1 and right_in_2) or (right_in_1 and left_in_2):
                    return True
            return False
        return score1 * score2 < 0

    def _polarity_score(self, text: str) -> int:
        tokens = self._tokenize(text)
        positive = len(tokens & _POSITIVE_CUES)
        negative = len(tokens & _NEGATIVE_CUES)
        return positive - negative

    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _clean_sentence(self, text: str) -> str:
        text = re.sub(r"\[[0-9,\s]+\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:500]

    def _looks_like_claim(self, text: str) -> bool:
        word_count = len(text.split())
        if word_count < CLAIM_GRAPH_CONFIG["min_sentence_words"]:
            return False
        if text.endswith("?"):
            return False
        lowered = text.lower()
        cue_hit = bool(self._tokenize(lowered) & (_POSITIVE_CUES | _NEGATIVE_CUES))
        has_measurement = bool(re.search(r"\d", text))
        has_copula = any(token in lowered for token in [" is ", " are ", " was ", " were ", " show", " suggests", " indicate"])
        return cue_hit or has_measurement or has_copula

    def _sentence_score(self, text: str) -> float:
        tokens = self._tokenize(text)
        cue_score = len(tokens & (_POSITIVE_CUES | _NEGATIVE_CUES))
        measurement_score = 1.0 if re.search(r"\d", text) else 0.0
        return len(text.split()) + cue_score * 3 + measurement_score * 2

    def _tokenize(self, text: str) -> set:
        tokens = {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower())
            if token not in _STOPWORDS and len(token) > 2
        }
        return tokens

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _stable_claim_id(self, doc_id: str, chunk_id: str, idx: int, text: str) -> str:
        digest = hashlib.md5(f"{doc_id}:{chunk_id}:{idx}:{text}".encode("utf-8")).hexdigest()
        return f"claim_{digest[:12]}"
