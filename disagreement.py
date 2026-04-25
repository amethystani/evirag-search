"""
EVIRAG Disagreement Reasoning
Claim graph construction, disagreement metrics, synthesis, and confidence calibration
"""

import json
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict

from config import (
    GRAPH_CONFIG, METRICS_CONFIG, CONFIDENCE_CONFIG,
    get_prompt, get_model_config
)
from epistemic_engine import (
    AtomicClaim, ClaimRelationship, ExplanationHypothesis, OllamaClient
)


@dataclass
class DisagreementMetrics:
    """Computed disagreement metrics"""
    disagreement_density: float
    conflict_ratio: float
    claim_entropy: float
    conflict_centrality: Dict[str, float]
    visual_text_mismatch: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Viewpoint:
    """A distinct viewpoint/interpretation"""
    name: str
    summary: str
    supporting_claim_ids: List[str]
    source_count: int
    visual_strength: float
    weaknesses: List[str]
    confidence: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MultiViewAnswer:
    """Multi-view answer with disagreement awareness"""
    query: str
    dominant_view: Viewpoint
    alternative_views: List[Viewpoint]
    minority_views: List[Viewpoint]
    overall_confidence: str
    confidence_score: float
    confidence_reasoning: str
    direct_answer: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "direct_answer": self.direct_answer,
            "dominant_view": self.dominant_view.to_dict(),
            "alternative_views": [v.to_dict() for v in self.alternative_views],
            "minority_views": [v.to_dict() for v in self.minority_views],
            "overall_confidence": self.overall_confidence,
            "confidence_score": self.confidence_score,
            "confidence_reasoning": self.confidence_reasoning
        }


class DisagreementGraph:
    """Graph representation of claim relationships"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.claim_map = {}  # claim_id -> AtomicClaim
    
    def build(
        self,
        claims: List[AtomicClaim],
        relationships: List[ClaimRelationship],
        min_confidence: float = None
    ):
        """Build disagreement graph from claims and relationships"""

        # Reset graph state for this query — without this, stale claims from
        # previous queries accumulate and corrupt metrics on every subsequent call.
        self.graph = nx.Graph()
        self.claim_map = {}

        min_conf = min_confidence or GRAPH_CONFIG["min_edge_confidence"]

        # Add nodes
        for claim in claims:
            self.graph.add_node(
                claim.claim_id,
                text=claim.text,
                doc_id=claim.source_doc_id,
                doc_title=claim.source_doc_title
            )
            self.claim_map[claim.claim_id] = claim
        
        # Add edges with type mapping
        # NLI returns: "entailment", "contradiction", "partially entailment"
        # Graph expects: "supports", "contradicts", "neutral"
        type_mapping = {
            "entailment": "supports",
            "partially entailment": "supports",
            "partial entailment": "supports",
            "contradiction": "contradicts",
            "partial contradiction": "contradicts",  # Treat partial contradiction as contradiction
            "partially contradicts": "contradicts",  # Handle variant
            "contradition": "contradicts",  # Handle common NLI typo
            "neutral": "neutral"
        }
        
        edges_added = 0
        edges_filtered_confidence = 0
        edges_filtered_type = 0
        
        for rel in relationships:
            if rel.confidence >= min_conf:
                # Map NLI relationship type to graph edge type
                nli_type = rel.relationship.lower()
                edge_type = type_mapping.get(nli_type, nli_type)
                
                if edge_type in GRAPH_CONFIG["edge_types"]:
                    self.graph.add_edge(
                        rel.claim1_id,
                        rel.claim2_id,
                        relationship=edge_type,
                        confidence=rel.confidence,
                        reasoning=rel.reasoning
                    )
                    edges_added += 1
                else:
                    edges_filtered_type += 1
                    print(f"  WARNING: Unknown edge type '{nli_type}' (mapped to '{edge_type}')")
            else:
                edges_filtered_confidence += 1
        
        print(f"Graph built: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        if edges_filtered_confidence > 0:
            print(f"  Filtered {edges_filtered_confidence} relationships (confidence < {min_conf})")
        if edges_filtered_type > 0:
            print(f"  Filtered {edges_filtered_type} relationships (invalid type)")
    
    def get_supporting_claims(self, claim_id: str) -> List[str]:
        """Get claims that support this claim"""
        supporting = []
        for neighbor in self.graph.neighbors(claim_id):
            edge_data = self.graph[claim_id][neighbor]
            if edge_data.get("relationship") == "supports":
                supporting.append(neighbor)
        return supporting
    
    def get_contradicting_claims(self, claim_id: str) -> List[str]:
        """Get claims that contradict this claim"""
        contradicting = []
        for neighbor in self.graph.neighbors(claim_id):
            edge_data = self.graph[claim_id][neighbor]
            if edge_data.get("relationship") == "contradicts":
                contradicting.append(neighbor)
        return contradicting
    
    def get_claim_communities(self) -> List[List[str]]:
        """Detect communities of related claims"""
        if not GRAPH_CONFIG["community_detection"]:
            return []
        if len(self.graph.nodes) < 2:
            # greedy_modularity_communities requires at least 1 node with edges
            return [[n] for n in self.graph.nodes]
        try:
            from networkx.algorithms import community
            communities = community.greedy_modularity_communities(self.graph)
            return [list(c) for c in communities]
        except (ValueError, ZeroDivisionError):
            return [[n] for n in self.graph.nodes]
    
    def to_dict(self) -> Dict:
        """Export graph to dict for visualization"""
        return {
            "nodes": [
                {
                    "id": node,
                    "text": data.get("text", ""),
                    "doc_title": data.get("doc_title", "")
                }
                for node, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "relationship": data.get("relationship", "neutral"),
                    "confidence": data.get("confidence", 0.5)
                }
                for u, v, data in self.graph.edges(data=True)
            ]
        }


class DisagreementMetricsCalculator:
    """Calculate disagreement metrics from claim graph"""
    
    def __init__(self, graph: DisagreementGraph):
        self.graph = graph
    
    def calculate_all(self, visual_mismatch: float = 0.0) -> DisagreementMetrics:
        """Calculate all disagreement metrics"""
        
        return DisagreementMetrics(
            disagreement_density=self._disagreement_density(),
            conflict_ratio=self._conflict_ratio(),
            claim_entropy=self._claim_entropy(),
            conflict_centrality=self._conflict_centrality(),
            visual_text_mismatch=visual_mismatch
        )
    
    def _disagreement_density(self) -> float:
        """Measure density of contradictory edges"""
        if len(self.graph.graph.edges) == 0:
            return 0.0
        
        contradiction_count = sum(
            1 for u, v, data in self.graph.graph.edges(data=True)
            if data.get("relationship") == "contradicts"
        )
        
        return contradiction_count / len(self.graph.graph.edges)
    
    def _conflict_ratio(self) -> float:
        """Ratio of contradicting edges to supporting edges.

        Formula: |contradicts| / |supports|
        - 0.0  = pure consensus, no contradictions
        - 1.0  = equal number of contra and support edges
        - >1.0 = more contradictions than agreements (high controversy)
        Capped at 5.0 to remain JSON-safe if supports is very small.
        """
        contradicts = 0
        supports = 0

        for u, v, data in self.graph.graph.edges(data=True):
            rel = data.get("relationship")
            if rel == "contradicts":
                contradicts += 1
            elif rel == "supports":
                supports += 1

        if supports == 0:
            # All edges are contradictions (or no edges) — maximum conflict
            return min(float(contradicts), 5.0) if contradicts > 0 else 0.0

        return min(contradicts / supports, 5.0)
    
    def _claim_entropy(self) -> float:
        """Measure diversity of claim stances"""
        # Group claims by document
        doc_groups = defaultdict(list)
        for claim_id in self.graph.graph.nodes:
            claim = self.graph.claim_map[claim_id]
            doc_groups[claim.source_doc_id].append(claim_id)
        
        if len(doc_groups) <= 1:
            return 0.0
        
        # Calculate entropy based on document diversity
        total_claims = len(self.graph.graph.nodes)
        probs = [len(claims) / total_claims for claims in doc_groups.values()]

        entropy = -sum(p * float(np.log(p)) for p in probs if p > 0)

        # Normalize — cast to Python float so FastAPI can JSON-serialise it
        max_entropy = float(np.log(len(doc_groups)))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0
    
    def _conflict_centrality(self) -> Dict[str, float]:
        """Identify claims central to conflicts"""
        centrality = {}
        
        for claim_id in self.graph.graph.nodes:
            # Count contradictory edges
            conflict_degree = len(self.graph.get_contradicting_claims(claim_id))
            centrality[claim_id] = conflict_degree
        
        # Normalize
        max_conflict = max(centrality.values()) if centrality else 1
        if max_conflict > 0:
            centrality = {k: v / max_conflict for k, v in centrality.items()}
        
        return centrality


class MultiViewSynthesizer:
    """Synthesize multi-view answers from disagreement graph"""
    
    def __init__(self, backend="local"):
        self.backend = backend
        self.slm_config = get_model_config("synthesizer", "slm", backend)
        self.llm_config = get_model_config("synthesizer_escalation", "llm", backend)
        
        # Auto-select client
        config_to_check = self.llm_config or self.slm_config
        if config_to_check and "groq.com" in config_to_check.endpoint:
            from groq_client import GroqClient
            self.client = GroqClient()
        else:
            self.client = OllamaClient()
    
    def synthesize(
        self,
        query: str,
        graph: DisagreementGraph,
        metrics: DisagreementMetrics,
        use_llm: bool = False
    ) -> Dict[str, List[Viewpoint]]:
        """Synthesize viewpoints from claim graph"""

        # Guard: empty graph — return a minimal placeholder answer
        if len(graph.graph.nodes) == 0:
            placeholder = Viewpoint(
                name="Dominant View",
                summary=(
                    f"Insufficient evidence extracted from corpus to synthesize "
                    f"a multi-view answer for: '{query}'. "
                    "Ensure corpus PDFs contain relevant content and try again."
                ),
                supporting_claim_ids=[],
                source_count=0,
                visual_strength=0.0,
                weaknesses=["No claims extracted"],
                confidence=0.1,
            )
            return {"dominant": [placeholder], "alternative": [], "minority": []}

        # Detect contradictions in graph
        contradictory_pairs = [
            (u, v) for u, v, data in graph.graph.edges(data=True)
            if data.get("relationship") == "contradicts"
        ]
        
        # Detect communities (viewpoints) — cap at 6 to prevent fragmentation
        MAX_COMMUNITIES = 6
        communities = graph.get_claim_communities()
        
        if not communities:
            # Fallback: single viewpoint
            all_claims = list(graph.claim_map.keys())
            communities = [all_claims] if all_claims else []
        
        # Merge excess small communities into the largest one to prevent 20+ minority views
        if len(communities) > MAX_COMMUNITIES:
            communities.sort(key=len, reverse=True)
            merged = [c for sub in communities[MAX_COMMUNITIES - 1:] for c in sub]
            communities = communities[:MAX_COMMUNITIES - 1] + [merged]
        
        # If contradictions exist, ensure they form separate viewpoints
        # Only split if we have meaningful contradictions (≥2) to avoid over-splitting
        if contradictory_pairs and len(contradictory_pairs) >= 2 and len(communities) == 1:
            # Split single community into opposing viewpoints
            communities = self._split_by_contradictions(
                communities[0], contradictory_pairs, graph
            )
        
        # Generate viewpoint for each community
        viewpoints = []
        for i, community in enumerate(communities):
            viewpoint = self._generate_viewpoint(
                query, community, graph, f"Viewpoint {i+1}", 
                use_llm, has_contradictions=(len(contradictory_pairs) > 0)
            )
            viewpoints.append(viewpoint)
        
        # Rank viewpoints by support
        viewpoints.sort(key=lambda v: (v.source_count, v.confidence), reverse=True)
        
        # Categorize
        if len(viewpoints) == 0:
            return {"dominant": [], "alternative": [], "minority": []}
        elif len(viewpoints) == 1:
            return {
                "dominant": [viewpoints[0]],
                "alternative": [],
                "minority": []
            }
        else:
            return {
                "dominant": [viewpoints[0]],
                "alternative": viewpoints[1:3] if len(viewpoints) > 1 else [],
                "minority": viewpoints[3:] if len(viewpoints) > 3 else []
            }
    
    def _split_by_contradictions(
        self, 
        community: List[str], 
        contradictory_pairs: List[tuple],
        graph: DisagreementGraph
    ) -> List[List[str]]:
        """Split community at contradiction boundaries to preserve opposing views"""
        
        # Build contradiction graph
        contradict_set = set()
        for u, v in contradictory_pairs:
            contradict_set.add((u, v))
            contradict_set.add((v, u))
        
        # Partition claims into two groups by contradiction
        group1 = set()
        group2 = set()
        
        for claim_id in community:
            # Check which group this claim opposes
            opposes_group1 = any((claim_id, g1) in contradict_set for g1 in group1)
            opposes_group2 = any((claim_id, g2) in contradict_set for g2 in group2)
            
            if opposes_group1 and not opposes_group2:
                group2.add(claim_id)
            elif opposes_group2 and not opposes_group1:
                group1.add(claim_id)
            elif not group1:
                group1.add(claim_id)
            else:
                group2.add(claim_id)
        
        # Return non-empty groups
        result = []
        if group1:
            result.append(list(group1))
        if group2:
            result.append(list(group2))
        
        return result if len(result) > 1 else [community]
    
    
    def _generate_viewpoint(
        self,
        query: str,
        claim_ids: List[str],
        graph: DisagreementGraph,
        viewpoint_name: str,
        use_llm: bool,
        has_contradictions: bool = False
    ) -> Viewpoint:
        """Generate a single viewpoint from claims"""
        
        # Get claim texts
        claims_text = []
        doc_titles = set()
        for claim_id in claim_ids:
            claim = graph.claim_map.get(claim_id)
            if claim:
                claims_text.append(claim.text)
                doc_titles.add(claim.source_doc_title)
        
        if not use_llm:
            summary = self._build_heuristic_summary(
                query,
                claims_text,
                has_contradictions=has_contradictions,
            )
            return Viewpoint(
                name=viewpoint_name,
                summary=summary,
                supporting_claim_ids=claim_ids,
                source_count=len(doc_titles),
                visual_strength=0.5,
                weaknesses=[],
                confidence=0.6,
            )

        # Generate summary using LLM
        model_config = self.llm_config if use_llm else self.slm_config
        
        # Balanced prompt enhancement when contradictions exist
        contradiction_guidance = ""
        if has_contradictions:
            contradiction_guidance = """
Note: Multiple perspectives exist in the literature. Synthesize this group of claims naturally,
capturing their collective viewpoint without forcing artificial opposition."""
        
        # Plain-text prompt — more reliable on 0.6B than JSON output
        prompt = f"""You are summarizing a group of scientific claims about: "{query}"

Claims to summarize:
{chr(10).join(f"- {c}" for c in claims_text[:8])}

Write 2-3 sentences summarizing the main finding or perspective these claims represent.
Do NOT include JSON, bullet points, or headings. Plain sentences only."""

        response = self.client.generate(
            model=model_config.name,
            prompt=prompt,
            temperature=model_config.temperature,
            max_tokens=400,
        )

        # Strip any JSON the model accidentally includes
        import re as _re, json as _json
        summary = (response or "").strip()
        # If model returned JSON, extract summary field
        try:
            parsed = _json.loads(summary)
            if isinstance(parsed, dict):
                summary = parsed.get("summary") or parsed.get("text") or str(parsed)
        except Exception:
            pass
        # Strip residual braces/brackets
        if "{" in summary or "[" in summary:
            summary = _re.sub(r'[\{\}\[\]]', '', summary).strip()
        summary = self._truncate_summary(summary, 400) if summary else f"Viewpoint based on {len(claims_text)} claims"
        weaknesses = []
        
        return Viewpoint(
            name=viewpoint_name,
            summary=summary,
            supporting_claim_ids=claim_ids,
            source_count=len(doc_titles),
            visual_strength=0.5,  # Placeholder, filled by caller
            weaknesses=weaknesses,
            confidence=0.6  # Placeholder
        )

    def _build_heuristic_summary(
        self,
        query: str,
        claims_text: List[str],
        has_contradictions: bool = False,
    ) -> str:
        """Fast non-generative summary for low-latency mode."""
        if not claims_text:
            return f"Limited evidence was retrieved for '{query}'."

        lead_claim = claims_text[0].strip()
        if len(claims_text) == 1:
            return lead_claim[:400]

        secondary = claims_text[1].strip()
        if lead_claim.endswith("."):
            lead_claim = lead_claim[:-1]
        if secondary.endswith("."):
            secondary = secondary[:-1]

        summary = f"{lead_claim}. Additional retrieved evidence aligns around: {secondary}."
        if has_contradictions:
            summary += " The broader literature remains mixed, so this should be read as one viewpoint rather than a settled consensus."
        return self._truncate_summary(summary, 400)

    def _truncate_summary(self, text: str, limit: int) -> str:
        cleaned = " ".join(str(text or "").split())
        if len(cleaned) <= limit:
            return cleaned
        clipped = cleaned[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")
        sentence_end = max(clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"))
        if sentence_end > limit * 0.55:
            return clipped[:sentence_end + 1]
        return clipped + "..."


class ConfidenceCalibrator:
    """Calibrate confidence based on evidence structure"""
    
    def __init__(self, backend="local"):
        self.backend = backend
        self.weights = CONFIDENCE_CONFIG["factors"]
        self.thresholds = CONFIDENCE_CONFIG["thresholds"]
        self.client = OllamaClient()
    
    def calibrate(
        self,
        supporting_claims: List[AtomicClaim],
        contradicting_claims: List[AtomicClaim],
        metrics: DisagreementMetrics,
        hypothesis: ExplanationHypothesis
    ) -> Tuple[str, float, str]:
        """
        Calibrate confidence based on evidence structure
        Returns: (confidence_level, score, reasoning)
        """
        
        # Factor 1: Claim agreement ratio
        total_claims = len(supporting_claims) + len(contradicting_claims)
        if total_claims > 0:
            agreement_score = len(supporting_claims) / total_claims
        else:
            agreement_score = 0.5
        
        # Factor 2: Source diversity
        support_sources = set(c.source_doc_id for c in supporting_claims)
        contradict_sources = set(c.source_doc_id for c in contradicting_claims)
        all_sources = support_sources | contradict_sources
        
        if len(all_sources) > 1:
            diversity_score = min(len(all_sources) / 10, 1.0)  # Normalize to max 10 sources
        else:
            diversity_score = 0.3
        
        # Factor 3: Contradiction severity
        contradiction_score = 1.0 - metrics.conflict_ratio
        
        # Factor 4: Unverified assumptions
        unverified_ratio = len(hypothesis.assumptions) / max(len(hypothesis.supporting_claims), 1)
        assumption_score = max(0, 1.0 - unverified_ratio)
        
        # Factor 5: Visual alignment
        visual_score = 1.0 - metrics.visual_text_mismatch
        
        # Weighted combination
        final_score = (
            self.weights["claim_agreement"] * agreement_score +
            self.weights["source_diversity"] * diversity_score +
            self.weights["contradiction_severity"] * contradiction_score +
            self.weights["unverified_assumptions"] * assumption_score +
            self.weights["visual_alignment"] * visual_score
        )
        
        # Determine level
        if final_score >= self.thresholds["high"]:
            level = "high"
        elif final_score >= self.thresholds["medium"]:
            level = "medium"
        else:
            level = "low"
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            len(supporting_claims),
            len(contradicting_claims),
            len(support_sources),
            len(contradict_sources),
            metrics
        )
        
        return level, final_score, reasoning
    
    def _generate_reasoning(
        self,
        support_count: int,
        contradict_count: int,
        support_sources: int,
        contradict_sources: int,
        metrics: DisagreementMetrics
    ) -> str:
        """Generate human-readable confidence reasoning"""
        
        factors = []
        
        factors.append(f"{support_count} claims support, {contradict_count} oppose")
        factors.append(f"Evidence from {support_sources + contradict_sources} distinct documents")
        
        if metrics.conflict_ratio > 0.5:
            factors.append("High conflict ratio indicates significant disagreement")
        elif metrics.conflict_ratio < 0.2:
            factors.append("Low conflict ratio indicates general agreement")
        
        if metrics.visual_text_mismatch > 0.3:
            factors.append("Visual evidence partially ambiguous")
        
        return "\n- ".join([""] + factors)


class DisagreementReasoner:
    """Main interface for disagreement-aware reasoning"""
    
    def __init__(self, backend="local"):
        self.backend = backend
        self.graph = DisagreementGraph()
        self.metrics_calculator = None
        self.synthesizer = MultiViewSynthesizer(backend)
        self.calibrator = ConfidenceCalibrator(backend)
    
    def reason(
        self,
        query: str,
        hypothesis: ExplanationHypothesis,
        claims: List[AtomicClaim],
        relationships: List[ClaimRelationship],
        visual_mismatch: float = 0.0,
        escalate_to_llm: bool = False
    ) -> MultiViewAnswer:
        """Execute full disagreement reasoning pipeline"""
        
        # 1. Build graph
        self.graph.build(claims, relationships)
        
        # 2. Calculate metrics
        self.metrics_calculator = DisagreementMetricsCalculator(self.graph)
        metrics = self.metrics_calculator.calculate_all(visual_mismatch)
        
        # 3. Synthesize viewpoints
        viewpoints_dict = self.synthesizer.synthesize(
            query, self.graph, metrics, use_llm=escalate_to_llm
        )
        
        # 4. Identify supporting and truly contradicting claims via graph edges
        # Supporting = dominant view's claims
        # Contradicting = claims with explicit NLI 'contradicts' edge to any dominant claim
        if viewpoints_dict["dominant"]:
            dominant_claim_ids = viewpoints_dict["dominant"][0].supporting_claim_ids
            supporting_claims = [self.graph.claim_map[cid] for cid in dominant_claim_ids
                                if cid in self.graph.claim_map]
        else:
            dominant_claim_ids = []
            supporting_claims = []

        # Use explicit contradiction edges - only claims that NLI-confirmed oppose dominant view
        contradicting_claim_ids = set()
        for cid in dominant_claim_ids:
            if cid in self.graph.claim_map:
                for opp_id in self.graph.get_contradicting_claims(cid):
                    contradicting_claim_ids.add(opp_id)
        contradicting_claims = [self.graph.claim_map[cid] for cid in contradicting_claim_ids
                               if cid in self.graph.claim_map]
        
        # 5. Calibrate confidence
        confidence_level, confidence_score, reasoning = self.calibrator.calibrate(
            supporting_claims, contradicting_claims, metrics, hypothesis
        )

        direct_answer = self._build_direct_answer(
            query=query,
            dominant=viewpoints_dict["dominant"][0] if viewpoints_dict["dominant"] else None,
            alternatives=viewpoints_dict["alternative"] + viewpoints_dict["minority"],
            metrics=metrics,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
        )
        
        # 6. Create multi-view answer
        answer = MultiViewAnswer(
            query=query,
            dominant_view=viewpoints_dict["dominant"][0] if viewpoints_dict["dominant"] else None,
            alternative_views=viewpoints_dict["alternative"],
            minority_views=viewpoints_dict["minority"],
            overall_confidence=confidence_level,
            confidence_score=confidence_score,
            confidence_reasoning=reasoning,
            direct_answer=direct_answer,
        )
        
        return answer, metrics

    def _build_direct_answer(
        self,
        query: str,
        dominant: Optional[Viewpoint],
        alternatives: List[Viewpoint],
        metrics: DisagreementMetrics,
        confidence_level: str,
        confidence_score: float,
    ) -> str:
        """Create the user-facing answer before exposing the evidence views."""
        if not dominant or not dominant.summary:
            return (
                "The corpus did not return enough evidence to answer this query directly. "
                "Inspect the sources and claims to confirm whether the corpus contains the right material."
            )

        dominant_summary = self._clean_sentence(dominant.summary)
        strongest_alternative = self._clean_sentence(alternatives[0].summary) if alternatives else ""
        yes_no = self._looks_like_yes_no_query(query)
        leaning = self._infer_query_leaning(dominant_summary)

        if yes_no and leaning:
            answer = f"Short answer: the retrieved corpus leans {leaning}."
        elif yes_no:
            answer = "Short answer: the retrieved corpus is mixed rather than a clean yes or no."
        else:
            answer = "Best-supported answer from the retrieved corpus:"

        answer += f" {dominant_summary}"

        has_real_disagreement = metrics.conflict_ratio >= 0.15 or metrics.disagreement_density >= 0.15
        if (
            strongest_alternative
            and has_real_disagreement
            and not self._looks_like_source_artifact(strongest_alternative)
            and self._is_substantive_caveat(strongest_alternative)
        ):
            answer += f" Main caveat: {strongest_alternative}"
        elif has_real_disagreement:
            answer += " The claim graph still shows material disagreement, so treat this as a corpus-grounded position rather than a settled fact."

        answer += f" Confidence is {confidence_level} ({confidence_score:.2f})."
        return answer[:900]

    def _looks_like_yes_no_query(self, query: str) -> bool:
        first = (query or "").strip().split(" ", 1)[0].lower()
        return first in {
            "is", "are", "was", "were", "do", "does", "did", "can",
            "could", "should", "would", "will", "has", "have", "had"
        }

    def _infer_query_leaning(self, text: str) -> Optional[str]:
        lowered = text.lower()
        positive_terms = [
            "improve", "improves", "improved", "benefit", "beneficial",
            "better", "support", "supports", "encourage", "enhance",
            "increases", "effective", "important", "helps", "not burden",
            "does not burden", "responsible for engaging"
        ]
        negative_terms = [
            "not improve", "not better", "no evidence", "harm", "harms",
            "worse", "reduces", "reduce", "contradict", "fails", "failed",
            "denies", "ineffective"
        ]
        pos = sum(1 for term in positive_terms if term in lowered)
        neg = sum(1 for term in negative_terms if term in lowered)
        if pos > neg:
            return "yes"
        if neg > pos:
            return "no"
        return None

    def _clean_sentence(self, text: str) -> str:
        cleaned = " ".join(str(text or "").split())
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned

    def _looks_like_source_artifact(self, text: str) -> bool:
        lowered = text.lower()
        artifact_markers = [
            "doi:", ".docx", ".qxd", ".indd", "microsoft word",
            "access denied", "copyright", "downloaded from"
        ]
        if any(marker in lowered for marker in artifact_markers):
            return True
        words = [w for w in lowered.replace("/", " ").replace(".", " ").split() if w.isalpha()]
        return len(words) < 8

    def _is_substantive_caveat(self, text: str) -> bool:
        lowered = text.lower()
        caveat_terms = [
            "however", "but", "although", "negative", "burden", "harm",
            "worse", "reduce", "denies", "contradict", "opposes",
            "ineffective", "mixed", "limited", "no evidence"
        ]
        return any(term in lowered for term in caveat_terms)


if __name__ == "__main__":
    # Test disagreement reasoning
    print("Testing Disagreement Reasoning")
    print("=" * 60)
    
    # Mock data
    from epistemic_engine import AtomicClaim, ClaimRelationship, ExplanationHypothesis
    
    claims = [
        AtomicClaim("c1", "Overparameterization improves generalization", "chunk1", "doc1", "Paper A"),
        AtomicClaim("c2", "Double descent phenomenon observed", "chunk2", "doc1", "Paper A"),
        AtomicClaim("c3", "Overparameterization leads to overfitting", "chunk3", "doc2", "Paper B"),
    ]
    
    relationships = [
        ClaimRelationship("c1", "c2", "supports", 0.8, "Double descent supports overparameterization"),
        ClaimRelationship("c1", "c3", "contradicts", 0.9, "Direct contradiction"),
    ]
    
    hypothesis = ExplanationHypothesis(
        central_hypothesis="Overparameterization improves generalization",
        supporting_claims=["Double descent"],
        assumptions=["Sufficient data"],
        expected_counterclaims=["Overfitting"]
    )
    
    # Reason
    reasoner = DisagreementReasoner()
    answer, metrics = reasoner.reason(
        query="Does overparameterization help?",
        hypothesis=hypothesis,
        claims=claims,
        relationships=relationships
    )
    
    print("\nMetrics:")
    print(json.dumps(metrics.to_dict(), indent=2))
    
    print("\nAnswer:")
    print(json.dumps(answer.to_dict(), indent=2))
