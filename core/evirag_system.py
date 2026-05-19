"""
EVIRAG Main System
Orchestrates the complete evidence-centric, disagreement-aware RAG pipeline

Novel research contributions integrated (for tier-1 publication):
  - EpistemicDivergence (ED) metric: formal JSD-based viewpoint divergence
  - PolarizationIndex (PI): signed Laplacian spectral measure of bipartite conflict
  - CausalDisagreementAttribution: WHY papers disagree (CDA-7 taxonomy)
  - TemporalEpistemicDrift: how disagreements evolve over time
  - EvaluationFramework: formal benchmark with CD-F1, EC@K, VTC metrics
"""

import copy
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field

from config import (
    EXECUTION_MODES, GRAPH_CONFIG, SYSTEM_CONSTRAINTS,
    CLAIM_GRAPH_CONFIG, EMBEDDING_CONFIG, get_model_config
)
from data_layer import DataManager
from vlm_module import VisualEvidenceProcessor, VisualDisagreementAnalyzer
from epistemic_engine import EpistemicEngine, EpistemicIntent, ExplanationHypothesis
from multi_agent import MultiAgentOrchestrator
from disagreement import DisagreementReasoner, MultiViewAnswer

# Novel research contributions
from epistemic_divergence import EpistemicDivergenceCalculator
from causal_attribution import CausalDisagreementAttributor
from temporal_tracker import TemporalEpistemicDriftAnalyzer


@dataclass
class EVIRAGConfig:
    """Configuration for EVIRAG system"""
    mode: str = "evirag"  # "vanilla_rag" or "evirag"
    backend: str = "local"  # "local" (Ollama) or "cloud" (Groq)
    use_visual_grounding: bool = False
    enabled_agents: List[str] = None  # None = all agents
    depth_vs_speed: str = "balanced"  # "fast", "balanced", "deep"'thorough'
    rebuild_index: bool = False

    # Novel research flags (enable for full paper-quality output)
    use_epistemic_divergence: bool = True    # Compute ED, PI, CCS metrics
    use_causal_attribution: bool = True      # Attribute WHY papers disagree
    use_temporal_tracking: bool = True       # Track temporal epistemic drift
    auto_fallback_to_full: bool = True       # Fast mode can escalate to full pipeline

    def __post_init__(self):
        if self.mode not in EXECUTION_MODES:
            raise ValueError(f"Invalid mode: {self.mode}")

        if self.enabled_agents is None:
            self.enabled_agents = ["precision", "recall", "skeptic", "counterfactual"]


class EVIRAGSystem:
    """Main EVIRAG system"""
    
    def __init__(self, config: EVIRAGConfig):
        self.config = config
        self.backend = config.backend
        
        print(f"Initializing EVIRAG System (backend: {self.backend})...")
        
        # Initialize components
        self.data_manager = DataManager(backend=self.backend)
        
        # Initialize with backend parameter
        self.epistemic_engine = EpistemicEngine(backend=self.backend)
        self.multi_agent = MultiAgentOrchestrator(
            enabled_agents=config.enabled_agents,
            backend=self.backend
        )
        self.disagreement_reasoner = DisagreementReasoner(backend=self.backend) # Kept original DisagreementReasoner, assuming 'DisagreementAnalyzer' was a typo in instruction
        
        # VLM Module (always local)
        if config.use_visual_grounding:
            self.visual_processor = VisualEvidenceProcessor()
            self.visual_analyzer = VisualDisagreementAnalyzer(self.visual_processor)
        else:
            self.visual_processor = None
            self.visual_analyzer = None

        # Novel research modules
        self.ed_calculator = EpistemicDivergenceCalculator() if config.use_epistemic_divergence else None
        self.causal_attributor = CausalDisagreementAttributor(backend=config.backend) if config.use_causal_attribution else None
        self.temporal_analyzer = TemporalEpistemicDriftAnalyzer() if config.use_temporal_tracking else None

        print("EVIRAG System initialized")
    
    def initialize_corpus(self, rebuild: bool = None):
        """Initialize or rebuild corpus"""
        rebuild = rebuild if rebuild is not None else self.config.rebuild_index
        
        print("Processing corpus...")
        self.data_manager.process_corpus(rebuild=rebuild)

        # In fast mode, pay the embedding model warmup cost at startup rather than
        # on the user's first query.
        if self.config.depth_vs_speed == "fast":
            self.data_manager.embedding_engine.load_model()
        
        if self.config.use_visual_grounding:
            print("Processing visual evidence...")
            self.visual_processor.process_figures(
                self.data_manager.figures_db,
                rebuild=rebuild
            )
        
        print("Corpus ready")
    
    def query(self, query: str, config: Optional[EVIRAGConfig] = None) -> Dict[str, Any]:
        """
        Main query interface
        Returns complete EVIRAG response with multi-view answer and metadata
        """
        
        # Use provided config or default
        exec_config = config or self.config
        
        # Route to appropriate execution mode
        if exec_config.mode == "vanilla_rag":
            return self._vanilla_rag(query)
        elif exec_config.mode == "evirag":
            return self._evirag(query, exec_config)
        else:
            raise ValueError(f"Unknown mode: {exec_config.mode}")
    
    def _vanilla_rag(self, query: str) -> Dict[str, Any]:
        """Standard RAG: retrieve and synthesize"""
        
        print(f"\n{'='*60}")
        print(f"VANILLA RAG MODE")
        print(f"Query: {query}")
        print(f"{'='*60}\n")
        
        # Simple retrieval
        results = self.data_manager.retrieve(query, k=10)
        
        # Extract text
        context = "\n\n".join([
            f"Source: {chunk.metadata.get('title', 'Unknown')}\n{chunk.text}"
            for chunk, score in results
        ])
        
        # Synthesize answer from retrieved context using LLM
        from epistemic_engine import OllamaClient
        from config import get_model_config
        client = OllamaClient()
        model_cfg = get_model_config("synthesizer_escalation", "llm", self.backend)
        
        prompt = f"""Based on the following retrieved passages, answer the question concisely.

Question: {query}

Passages:
{context[:3000]}

Answer:"""
        
        answer = client.generate(
            model=model_cfg.name,
            prompt=prompt,
            temperature=0.3,
            max_tokens=512
        ) if model_cfg else f"Based on {len(results)} sources (LLM unavailable)"
        
        return {
            "mode": "vanilla_rag",
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "title": chunk.metadata.get("title", "Unknown"),
                    "text": chunk.text[:200] + "...",
                    "score": float(score)
                }
                for chunk, score in results[:5]
            ],
            "statistics": {
                "total_sources": len(results),
                "retrieval_mode": "vanilla_vector_rag",
            },
            "trace": {
                "requested_mode": "vanilla_rag",
                "pipeline_mode": "vanilla_rag",
                "query": query,
                "models": {
                    "backend": self.backend,
                    "embedding_model": EMBEDDING_CONFIG["text_model"],
                    "synthesizer": getattr(get_model_config("synthesizer_escalation", "llm", self.backend), "name", None),
                },
                "steps": [
                    {"step": "vector_retrieval", "retrieved_chunks": len(results)},
                    {"step": "single_view_synthesis", "used_model": model_cfg.name if model_cfg else None},
                ],
            },
        }
    

    def _evirag(self, query: str, config: EVIRAGConfig) -> Dict[str, Any]:
        """Route to fast or full EVIRAG execution."""
        if config.depth_vs_speed == "fast" and self.data_manager.claim_graph.is_ready():
            fast_result = self._evirag_fast_graph(query, config)
            if fast_result is None:
                return self._evirag_full_pipeline(query, config)
            if fast_result.get("_fallback_to_full"):
                full_result = self._evirag_full_pipeline(
                    query,
                    config,
                    prior_trace=fast_result.get("trace"),
                )
                full_result.setdefault("trace", {})
                full_result["trace"]["fallback"] = fast_result["trace"].get("fallback", {})
                full_result["statistics"]["fallback_to_full"] = True
                return full_result
            return fast_result

        return self._evirag_full_pipeline(query, config)

    def _evirag_full_pipeline(
        self,
        query: str,
        config: EVIRAGConfig,
        prior_trace: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Full EVIRAG: Multi-agent + visual grounding + full pipeline."""
        print(f"\n{'='*60}")
        print(f"EVIRAG MODE (Full Pipeline)")
        print(f"Query: {query}")
        print(f"{'='*60}\n")
        
        # 1. Analyze intent and generate hypothesis
        print("Step 1: Epistemic intent analysis...")
        intent, hypothesis = self.epistemic_engine.process_query(query)
        
        print(f"Intent: {intent.disagreement_level} disagreement expected")
        print(f"Hypothesis: {hypothesis.central_hypothesis}")
        
        # 2. Multi-agent deliberative retrieval
        print("\nStep 2: Multi-agent deliberative retrieval...")
        
        def retrieval_func(q, k):
            return self.data_manager.retrieve(q, k)
        
        def claim_extraction_func(chunk, chunk_id_counter=0):
            return self.epistemic_engine.claim_extractor.extract_from_chunk(chunk, chunk_id_counter)
        
        agent_outputs = self.multi_agent.deliberate(
            hypothesis=hypothesis,
            disagreement_level=intent.disagreement_level,
            retrieval_func=retrieval_func,
            claim_extraction_func=claim_extraction_func,
            enabled_agents=config.enabled_agents
        )
        
        agent_summary = self.multi_agent.get_agent_summary(agent_outputs)
        print(f"Agent outputs: {agent_summary['total_claims']} total claims")
        
        # 3. Consolidate claims from all agents
        print("\nStep 3: Consolidating claims...")
        all_claims = []
        for output in agent_outputs:
            all_claims.extend(output.extracted_claims)
        
        # Deduplicate by text (simple heuristic)
        seen_texts = set()
        unique_claims = []
        for claim in all_claims:
            if claim.text not in seen_texts:
                unique_claims.append(claim)
                seen_texts.add(claim.text)
        
        print(f"Unique claims: {len(unique_claims)}")
        
        # 4. Build claim relationships
        print("\nStep 4: Building claim relationships...")
        relationships = self.epistemic_engine.build_claim_relationships(
            unique_claims,
            max_pairs=GRAPH_CONFIG['max_pairs_cloud'] if config.backend == 'cloud' else GRAPH_CONFIG['max_pairs']
        )
        print(f"Relationships: {len(relationships)}")
        
        # 5. Visual evidence analysis (if enabled)
        visual_mismatch = 0.0
        visual_analysis = None
        
        if config.use_visual_grounding and self.visual_processor:
            print("\nStep 5: Visual evidence analysis...")
            
            # 5a. Per-claim figure alignment — find the best figure for each claim
            aligned_figures = []
            for claim in unique_claims[:20]:
                top_matches = self.visual_processor.align_claim_with_figures(
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

            # 5b. Mismatch analysis — detect claims with weak visual support
            claim_data = [
                {
                    "text": claim.text,
                    "doc_id": claim.source_doc_id,
                    "stance": "neutral",
                }
                for claim in unique_claims[:20]
            ]
            mismatch_result = self.visual_analyzer.analyze_visual_text_mismatch(claim_data)
            visual_mismatch = mismatch_result["mismatch_score"]

            # 5c. Cross-paper figure comparisons for contradicting claim pairs
            cross_comparisons = []
            contradiction_rels = [
                r for r in relationships
                if r.relationship in ("contradiction", "contradicts")
            ][:5]
            claim_map_local = {c.claim_id: c for c in unique_claims}
            for rel in contradiction_rels:
                c1 = claim_map_local.get(rel.claim1_id)
                c2 = claim_map_local.get(rel.claim2_id)
                if c1 and c2:
                    comparison = self.visual_processor.compare_figures_across_claims(
                        c1.text, c2.text, c1.source_doc_id, c2.source_doc_id
                    )
                    cross_comparisons.append({
                        "claim1_id": rel.claim1_id,
                        "claim2_id": rel.claim2_id,
                        "interpretation_conflict": comparison["interpretation_conflict"],
                        "similar_evidence_count": comparison["similar_evidence_count"],
                    })

            visual_analysis = {
                "mismatch_score": visual_mismatch,
                "claims_with_weak_support": mismatch_result.get("claims_with_weak_support", 0),
                "total_claims": mismatch_result.get("total_claims", 0),
                "mismatches": mismatch_result.get("mismatches", []),
                "aligned_figures": sorted(
                    aligned_figures, key=lambda x: x["alignment_score"], reverse=True
                )[:20],
                "cross_comparisons": cross_comparisons,
                "total_indexed_figures": len(self.visual_processor.visual_evidence_db),
            }
            print(f"Visual-text mismatch: {visual_mismatch:.3f}")
            print(f"Aligned {len(aligned_figures)} claim-figure pairs, "
                  f"{len(cross_comparisons)} cross-paper comparisons")
        
        # 6. Disagreement reasoning and synthesis
        print("\nStep 6: Disagreement reasoning and synthesis...")
        
        # Escalate to LLM if high disagreement
        escalate = intent.disagreement_level == "high" or len(relationships) > 50
        
        answer, metrics = self.disagreement_reasoner.reason(
            query=query,
            hypothesis=hypothesis,
            claims=unique_claims,
            relationships=relationships,
            visual_mismatch=visual_mismatch,
            escalate_to_llm=escalate
        )
        
        print(f"\nConfidence: {answer.overall_confidence} ({answer.confidence_score:.3f})")
        
        # Print comprehensive final answer
        print(f"\n{'='*80}")
        print(f"🎯 FINAL ANSWER TO YOUR QUERY")
        print(f"{'='*80}")
        print(f"\nQuery: {query}\n")
        
        # Dominant view
        if answer.dominant_view:
            print(f"📊 DOMINANT VIEW:")
            print(f"{answer.dominant_view.summary}\n")
            print(f"   Supporting claims: {len(answer.dominant_view.supporting_claim_ids)}")
        
        # Alternative views
        if answer.alternative_views:
            print(f"\n🔄 ALTERNATIVE VIEWS ({len(answer.alternative_views)}):")
            for i, alt in enumerate(answer.alternative_views, 1):
                print(f"\n{i}. {alt.summary}")
                print(f"   Supporting claims: {len(alt.supporting_claim_ids)}")
        
        # Disagreement summary
        if hasattr(answer, 'disagreement_summary') and answer.disagreement_summary:
            print(f"\n⚖️  DISAGREEMENT ANALYSIS:")
            print(f"{answer.disagreement_summary}")
        
        # Graph statistics
        print(f"\n📊 DISAGREEMENT GRAPH:")
        print(f"   Total claims: {len(unique_claims)}")
        print(f"   Relationships: {len(relationships)}")
        print(f"   Disagreement density: {metrics.disagreement_density:.3f}")
        print(f"   Conflict ratio: {metrics.conflict_ratio:.3f}")
        
        print(f"\n📈 Overall Confidence: {answer.overall_confidence} ({answer.confidence_score:.2f})")
        print(f"{'='*80}\n")
        
        # 7. Novel research metrics ─────────────────────────────────────────

        # 7a. EpistemicDivergence: formal measure of viewpoint spread
        epistemic_divergence_result = None
        if self.ed_calculator and answer.dominant_view:
            try:
                claim_map_local = {c.claim_id: c for c in unique_claims}
                all_views = []
                for vd in [answer.dominant_view.to_dict()] + \
                           [v.to_dict() for v in answer.alternative_views] + \
                           [v.to_dict() for v in answer.minority_views]:
                    claim_ids = vd.get("supporting_claim_ids", [])
                    vd["claim_texts"] = [
                        claim_map_local[cid].text
                        for cid in claim_ids if cid in claim_map_local
                    ] or [vd.get("summary", "no summary")]
                    all_views.append(vd)
                ed_res = self.ed_calculator.compute(viewpoints=all_views)
                epistemic_divergence_result = {
                    "ed_score": ed_res.ed_score,
                    "polarization_index": ed_res.polarization_index,
                    "consensus_collapse_score": ed_res.consensus_collapse_score,
                    "controversy_class": ed_res.controversy_class,
                    "reasoning": ed_res.reasoning,
                    "dominant_vs_minority_gap": ed_res.dominant_vs_minority_gap,
                    "inter_view_separation": ed_res.inter_view_separation,
                }
                print(f"\nEpistemicDivergence: {ed_res.ed_score:.3f} "
                      f"(class: {ed_res.controversy_class})")
            except Exception as e:
                print(f"  EpistemicDivergence error (non-fatal): {e}")

        # 7b. Causal Disagreement Attribution
        causal_report = None
        if self.causal_attributor and relationships:
            try:
                claim_map_for_attr = {c.claim_id: c for c in unique_claims}
                causal_report_obj = self.causal_attributor.attribute(
                    query=query,
                    claims=unique_claims,
                    relationships=relationships,
                    claim_map=claim_map_for_attr,
                    max_contradictions=10,
                )
                causal_report = causal_report_obj.to_dict()
                print(f"\nCausal Attribution: dominant cause = "
                      f"{causal_report_obj.dominant_cause} "
                      f"(depth: {causal_report_obj.controversy_depth})")
            except Exception as e:
                print(f"  CausalAttribution error (non-fatal): {e}")

        # 7c. Temporal Epistemic Drift
        temporal_report = None
        if self.temporal_analyzer:
            try:
                doc_metadata = {}
                for c in unique_claims:
                    if hasattr(c, 'source_doc_id') and c.source_doc_id not in doc_metadata:
                        doc_metadata[c.source_doc_id] = {}
                temporal_obj = self.temporal_analyzer.analyze(
                    query=query,
                    claims=unique_claims,
                    relationships=relationships,
                    doc_metadata=doc_metadata,
                )
                temporal_report = temporal_obj.to_dict()
                print(f"\nTemporal Drift: {temporal_obj.trajectory} "
                      f"(confidence: {temporal_obj.trajectory_confidence:.2f})")
            except Exception as e:
                print(f"  TemporalDrift error (non-fatal): {e}")

        # 8. Compile full response
        answer_dict = self._enrich_answer_with_citations(answer, unique_claims)
        claim_records = self._build_claim_records(unique_claims)
        graph_dict = self.disagreement_reasoner.graph.to_dict()
        agent_details = [output.to_dict() for output in agent_outputs]
        trace = self._build_full_trace(
            query=query,
            config=config,
            intent=intent,
            hypothesis=hypothesis,
            agent_summary=agent_summary,
            agent_details=agent_details,
            relationships=relationships,
            escalate=escalate,
            prior_trace=prior_trace,
        )

        return {
            "mode": "evirag",
            "query": query,
            "intent": intent.to_dict(),
            "hypothesis": hypothesis.to_dict(),
            "agent_outputs": agent_summary,
            "agent_details": agent_details,
            "answer": answer_dict,
            "metrics": metrics.to_dict(),
            "visual_analysis": visual_analysis,
            "claims": claim_records,
            "graph": graph_dict,
            "sources": self._build_source_records(unique_claims),
            # Novel research outputs
            "epistemic_divergence": epistemic_divergence_result,
            "causal_attribution": causal_report,
            "temporal_drift": temporal_report,
            "trace": trace,
            "statistics": {
                "total_claims": len(unique_claims),
                "total_relationships": len(relationships),
                "agents_used": len(agent_outputs),
                "visual_grounding": config.use_visual_grounding,
                "epistemic_divergence_enabled": config.use_epistemic_divergence,
                "causal_attribution_enabled": config.use_causal_attribution,
                "temporal_tracking_enabled": config.use_temporal_tracking,
                "fallback_to_full": False,
            }
        }

    def _evirag_fast_graph(self, query: str, config: EVIRAGConfig) -> Optional[Dict[str, Any]]:
        """Low-latency EVIRAG path using the offline claim graph."""

        print(f"\n{'='*60}")
        print("EVIRAG MODE (Fast Claim Graph)")
        print(f"Query: {query}")
        print(f"{'='*60}\n")

        max_claims = (
            SYSTEM_CONSTRAINTS["max_claims_cloud"]
            if config.backend == "cloud"
            else SYSTEM_CONSTRAINTS["max_claims_per_query"]
        )
        claims, relationships, retrieval_meta = self.data_manager.retrieve_claim_subgraph(
            query,
            max_claims=max_claims,
        )

        if not claims:
            print("Fast claim graph returned no evidence; falling back to full pipeline.")
            return None

        fallback_reason = None
        if config.auto_fallback_to_full:
            fallback_reason = self._should_fast_graph_fallback(claims, relationships, retrieval_meta)
            if fallback_reason:
                return {
                    "_fallback_to_full": True,
                    "trace": {
                        "requested_mode": "evirag",
                        "requested_speed": config.depth_vs_speed,
                        "pipeline_mode": "fast_claim_graph",
                        "fallback": {
                            "triggered": True,
                            "reason": fallback_reason,
                        },
                        "retrieval": retrieval_meta,
                        "models": self._collect_model_trace(config, pipeline_mode="fast"),
                    },
                }

        contradiction_edges = retrieval_meta.get("contradiction_edges", 0)
        disagreement_level = self._infer_disagreement_level(relationships)
        intent = EpistemicIntent(
            factual_vs_disputed="highly_disputed" if contradiction_edges else "somewhat_disputed",
            empirical_vs_theoretical="mixed",
            disagreement_level=disagreement_level,
            reasoning="Fast graph mode inferred from offline claim graph structure.",
        )
        hypothesis = ExplanationHypothesis(
            central_hypothesis=query,
            supporting_claims=[],
            assumptions=[],
            expected_counterclaims=[],
        )

        print(
            f"Retrieved {len(claims)} claim nodes and {len(relationships)} edges "
            f"from offline graph"
        )

        visual_analysis = None
        visual_mismatch = 0.0

        if config.use_visual_grounding and self.visual_processor:
            print("\n  Visual evidence analysis (fast mode)...")
            aligned_figures = []
            for claim in claims[:20]:
                top_matches = self.visual_processor.align_claim_with_figures(
                    claim.text, doc_id=claim.source_doc_id, top_k=1
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
                    
            claim_data = [{"text": c.text, "doc_id": c.source_doc_id, "stance": "neutral"} for c in claims[:20]]
            mismatch_result = self.visual_analyzer.analyze_visual_text_mismatch(claim_data)
            visual_mismatch = mismatch_result["mismatch_score"]

            cross_comparisons = []
            contradiction_rels = [r for r in relationships if r.relationship in ("contradiction", "contradicts")][:5]
            claim_map_local = {c.claim_id: c for c in claims}
            for rel in contradiction_rels:
                c1, c2 = claim_map_local.get(rel.claim1_id), claim_map_local.get(rel.claim2_id)
                if c1 and c2:
                    comparison = self.visual_processor.compare_figures_across_claims(
                        c1.text, c2.text, c1.source_doc_id, c2.source_doc_id
                    )
                    cross_comparisons.append({
                        "claim1_id": rel.claim1_id,
                        "claim2_id": rel.claim2_id,
                        "interpretation_conflict": comparison["interpretation_conflict"],
                        "similar_evidence_count": comparison["similar_evidence_count"],
                    })

            visual_analysis = {
                "mismatch_score": visual_mismatch,
                "claims_with_weak_support": mismatch_result.get("claims_with_weak_support", 0),
                "total_claims": mismatch_result.get("total_claims", 0),
                "mismatches": mismatch_result.get("mismatches", []),
                "aligned_figures": sorted(aligned_figures, key=lambda x: x["alignment_score"], reverse=True)[:20],
                "cross_comparisons": cross_comparisons,
                "total_indexed_figures": len(self.visual_processor.visual_evidence_db),
            }

        answer, metrics = self.disagreement_reasoner.reason(
            query=query,
            hypothesis=hypothesis,
            claims=claims,
            relationships=relationships,
            visual_mismatch=visual_mismatch,
            escalate_to_llm=False,
        )

        epistemic_divergence_result = None
        if self.ed_calculator and answer.dominant_view:
            try:
                claim_lookup = self.data_manager.claim_graph.get_embedding_lookup(
                    [claim.claim_id for claim in claims]
                )
                all_views = [
                    answer.dominant_view.to_dict(),
                    *[v.to_dict() for v in answer.alternative_views],
                    *[v.to_dict() for v in answer.minority_views],
                ]
                ed_res = self.ed_calculator.compute_from_claim_embeddings(
                    viewpoints=all_views,
                    claim_embeddings=claim_lookup,
                    graph=self.disagreement_reasoner.graph.graph,
                )
                epistemic_divergence_result = {
                    "ed_score": ed_res.ed_score,
                    "polarization_index": ed_res.polarization_index,
                    "consensus_collapse_score": ed_res.consensus_collapse_score,
                    "controversy_class": ed_res.controversy_class,
                    "reasoning": ed_res.reasoning,
                    "dominant_vs_minority_gap": ed_res.dominant_vs_minority_gap,
                    "inter_view_separation": ed_res.inter_view_separation,
                }
            except Exception as e:
                print(f"  EpistemicDivergence error (non-fatal): {e}")

        temporal_report = None
        if self.temporal_analyzer:
            try:
                doc_metadata = {c.source_doc_id: {} for c in claims}
                temporal_obj = self.temporal_analyzer.analyze(
                    query=query,
                    claims=claims,
                    relationships=relationships,
                    doc_metadata=doc_metadata,
                )
                temporal_report = temporal_obj.to_dict()
            except Exception as e:
                print(f"  TemporalDrift error (non-fatal): {e}")

        answer_dict = self._enrich_answer_with_citations(answer, claims)
        claim_records = self._build_claim_records(claims)
        graph_dict = self.disagreement_reasoner.graph.to_dict()
        trace = self._build_fast_trace(
            query=query,
            config=config,
            retrieval_meta=retrieval_meta,
            intent=intent,
            hypothesis=hypothesis,
            relationships=relationships,
        )

        return {
            "mode": "evirag",
            "query": query,
            "intent": intent.to_dict(),
            "hypothesis": hypothesis.to_dict(),
            "agent_outputs": {
                "total_agents": 0,
                "total_claims": len(claims),
                "stances": {"support": 0, "oppose": 0, "neutral": 0},
            },
            "agent_details": [],
            "answer": answer_dict,
            "metrics": metrics.to_dict(),
            "visual_analysis": visual_analysis,
            "claims": claim_records,
            "graph": graph_dict,
            "sources": self._build_source_records(claims),
            "epistemic_divergence": epistemic_divergence_result,
            "causal_attribution": None,
            "temporal_drift": temporal_report,
            "trace": trace,
            "statistics": {
                "total_claims": len(claims),
                "total_relationships": len(relationships),
                "agents_used": 0,
                "visual_grounding": config.use_visual_grounding,
                "epistemic_divergence_enabled": config.use_epistemic_divergence,
                "causal_attribution_enabled": False,
                "temporal_tracking_enabled": config.use_temporal_tracking,
                "retrieval_mode": retrieval_meta.get("mode", "fast_claim_graph"),
                "seed_claims": retrieval_meta.get("seed_claims", 0),
                "fallback_to_full": False,
            }
        }

    def _infer_disagreement_level(self, relationships):
        contradiction_edges = len([
            rel for rel in relationships if rel.relationship == "contradicts"
        ])
        if contradiction_edges >= 3:
            return "high"
        if contradiction_edges >= 1:
            return "medium"
        return "low"

    def _should_fast_graph_fallback(
        self,
        claims: List[Any],
        relationships: List[Any],
        retrieval_meta: Dict[str, Any],
    ) -> Optional[str]:
        """Escalate to full pipeline when the fast graph looks underpowered."""
        support_edges = retrieval_meta.get("support_edges", 0)
        contradiction_edges = retrieval_meta.get("contradiction_edges", 0)

        if len(claims) < 6:
            return f"Only {len(claims)} claims retrieved from the offline graph."
        if len(relationships) < 3:
            return f"Only {len(relationships)} claim relationships available in the subgraph."
        if contradiction_edges >= 6:
            return f"High contradiction density ({contradiction_edges} contradiction edges) requires full reasoning."
        if contradiction_edges > support_edges and len(claims) < 12:
            return "Graph evidence is mixed and too small to trust the fast path."
        return None

    def _claim_to_record(self, claim: Any) -> Dict[str, Any]:
        """Normalize claim metadata for API/UI consumption."""
        citation_parts = [claim.source_doc_title or "Unknown source"]
        if getattr(claim, "section", None):
            citation_parts.append(str(claim.section).title())
        if getattr(claim, "page_num", None) is not None:
            citation_parts.append(f"p.{int(claim.page_num) + 1}")
        citation_label = " · ".join(citation_parts)
        return {
            "claim_id": claim.claim_id,
            "text": claim.text,
            "context": claim.context,
            "section": getattr(claim, "section", None),
            "page_num": getattr(claim, "page_num", None),
            "source_doc_id": claim.source_doc_id,
            "source_doc_title": claim.source_doc_title,
            "source_chunk_id": claim.source_chunk_id,
            "source_path": getattr(claim, "source_path", None),
            "citation_label": citation_label,
        }

    def _build_claim_records(self, claims: List[Any]) -> List[Dict[str, Any]]:
        return [self._claim_to_record(claim) for claim in claims]

    def _build_source_records(self, claims: List[Any]) -> List[Dict[str, Any]]:
        seen = {}
        for claim in claims:
            key = claim.source_doc_id
            if key in seen:
                continue
            seen[key] = {
                "doc_id": claim.source_doc_id,
                "title": claim.source_doc_title,
                "path": getattr(claim, "source_path", None),
            }
        return list(seen.values())

    def _build_viewpoint_citations(
        self,
        viewpoint_dict: Optional[Dict[str, Any]],
        claim_map: Dict[str, Any],
        limit: int = 5,
    ) -> Dict[str, Any]:
        if not viewpoint_dict:
            return viewpoint_dict

        supporting_ids = viewpoint_dict.get("supporting_claim_ids", [])
        citations = []
        for claim_id in supporting_ids[:limit]:
            claim = claim_map.get(claim_id)
            if not claim:
                continue
            citations.append(self._claim_to_record(claim))
        viewpoint_dict["citations"] = citations
        return viewpoint_dict

    def _enrich_answer_with_citations(self, answer: MultiViewAnswer, claims: List[Any]) -> Dict[str, Any]:
        answer_dict = answer.to_dict()
        claim_map = {claim.claim_id: claim for claim in claims}
        answer_dict["dominant_view"] = self._build_viewpoint_citations(
            answer_dict.get("dominant_view"),
            claim_map,
        )
        answer_dict["alternative_views"] = [
            self._build_viewpoint_citations(view, claim_map)
            for view in answer_dict.get("alternative_views", [])
        ]
        answer_dict["minority_views"] = [
            self._build_viewpoint_citations(view, claim_map)
            for view in answer_dict.get("minority_views", [])
        ]
        answer_dict["source_documents"] = self._build_source_records(claims)
        return answer_dict

    def _collect_model_trace(self, config: EVIRAGConfig, pipeline_mode: str) -> Dict[str, Any]:
        """Expose which models/components were used for the run."""
        trace = {
            "backend": config.backend,
            "embedding_model": EMBEDDING_CONFIG["text_model"],
            "claim_graph_verification": {
                "mode": CLAIM_GRAPH_CONFIG.get("verify_edges", "auto"),
                "local_budget": CLAIM_GRAPH_CONFIG.get("verification_budget_local", 0),
                "cloud_budget": CLAIM_GRAPH_CONFIG.get("verification_budget_cloud", 0),
            },
        }
        if pipeline_mode == "fast":
            trace["query_time_models"] = ["embedding_lookup", "heuristic_disagreement_synthesis"]
            return trace

        trace["query_time_models"] = {
            "intent_analyzer": getattr(get_model_config("intent_analyzer", "slm", config.backend), "name", None),
            "hypothesis_generator_slm": getattr(get_model_config("hypothesis_generator", "slm", config.backend), "name", None),
            "hypothesis_generator_llm": getattr(get_model_config("hypothesis_generator", "llm", config.backend), "name", None),
            "claim_extractor": getattr(get_model_config("claim_extractor", "llm", config.backend), "name", None),
            "nli_verifier": getattr(get_model_config("nli_verifier", "llm", config.backend), "name", None),
            "synthesizer": getattr(get_model_config("synthesizer_escalation", "llm", config.backend), "name", None),
            "agents": {
                agent_name: getattr(get_model_config(agent_name, "agent", config.backend), "name", None)
                for agent_name in config.enabled_agents
            },
        }
        return trace

    def _build_fast_trace(
        self,
        query: str,
        config: EVIRAGConfig,
        retrieval_meta: Dict[str, Any],
        intent: EpistemicIntent,
        hypothesis: ExplanationHypothesis,
        relationships: List[Any],
    ) -> Dict[str, Any]:
        return {
            "requested_mode": config.mode,
            "requested_speed": config.depth_vs_speed,
            "pipeline_mode": "fast_claim_graph",
            "query": query,
            "models": self._collect_model_trace(config, pipeline_mode="fast"),
            "fallback": {"triggered": False, "reason": None},
            "retrieval": retrieval_meta,
            "steps": [
                {"step": "offline_claim_graph_lookup", "selected_claims": retrieval_meta.get("selected_claims", 0)},
                {"step": "fast_intent_inference", "result": intent.to_dict()},
                {"step": "heuristic_hypothesis", "result": hypothesis.to_dict()},
                {
                    "step": "disagreement_reasoning",
                    "relationship_count": len(relationships),
                    "contradiction_edges": retrieval_meta.get("contradiction_edges", 0),
                },
            ],
        }

    def _build_full_trace(
        self,
        query: str,
        config: EVIRAGConfig,
        intent: EpistemicIntent,
        hypothesis: ExplanationHypothesis,
        agent_summary: Dict[str, Any],
        agent_details: List[Dict[str, Any]],
        relationships: List[Any],
        escalate: bool,
        prior_trace: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        trace = {
            "requested_mode": config.mode,
            "requested_speed": config.depth_vs_speed,
            "pipeline_mode": "full_evirag",
            "query": query,
            "models": self._collect_model_trace(config, pipeline_mode="full"),
            "fallback": {"triggered": False, "reason": None},
            "steps": [
                {"step": "intent_analysis", "result": intent.to_dict()},
                {"step": "hypothesis_generation", "result": hypothesis.to_dict()},
                {
                    "step": "multi_agent_retrieval",
                    "summary": agent_summary,
                    "agents": agent_details,
                },
                {"step": "claim_relationship_building", "relationship_count": len(relationships)},
                {"step": "disagreement_synthesis", "llm_escalated": escalate},
            ],
        }
        if prior_trace:
            trace["fast_graph_precheck"] = prior_trace
        return trace
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and configuration"""
        return {
            "config": {
                "mode": self.config.mode,
                "backend": self.config.backend,
                "enabled_agents": self.config.enabled_agents,
                "visual_grounding": self.config.use_visual_grounding,
                "depth_vs_speed": self.config.depth_vs_speed,
                "auto_fallback_to_full": self.config.auto_fallback_to_full,
            },
            "corpus": {
                "indexed": self.data_manager.vector_store.index is not None,
                "num_chunks": len(self.data_manager.vector_store.chunks) if self.data_manager.vector_store.chunks else 0,
                "claim_graph_ready": self.data_manager.claim_graph.is_ready(),
                "num_claims": len(self.data_manager.claim_graph.claims),
                "num_claim_edges": len(self.data_manager.claim_graph.relationships),
            },
            "visual": {
                "enabled": self.config.use_visual_grounding,
                "num_figures": len(self.visual_processor.visual_evidence_db) if self.visual_processor else 0
            }
        }


def demo():
    """Demo EVIRAG system"""
    
    print("="*60)
    print("EVIRAG DEMO")
    print("="*60)
    
    # Initialize system
    config = EVIRAGConfig(
        mode="evirag",
        use_visual_grounding=True,
        depth_vs_speed="balanced",
        rebuild_index=False
    )
    
    system = EVIRAGSystem(config)
    
    # Initialize corpus
    system.initialize_corpus()
    
    # Test query
    query = "Is overparameterization necessary for generalization in deep neural networks?"
    
    print(f"\n\nQuery: {query}\n")
    
    # Run EVIRAG
    result = system.query(query)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\nMode: {result['mode']}")
    print(f"Confidence: {result['answer']['overall_confidence']}")
    print(f"\nDominant View:")
    if result['answer']['dominant_view']:
        print(f"  {result['answer']['dominant_view']['summary']}")
    
    print(f"\nAlternative Views: {len(result['answer']['alternative_views'])}")
    
    print(f"\nStatistics:")
    for key, value in result['statistics'].items():
        print(f"  {key}: {value}")
    
    print(f"\nMetrics:")
    print(f"  Disagreement density: {result['metrics']['disagreement_density']:.3f}")
    print(f"  Conflict ratio: {result['metrics']['conflict_ratio']:.3f}")
    
    # Save full output
    import os
    output_path = os.path.join(os.path.dirname(__file__), "demo_output.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nFull output saved to: {output_path}")


if __name__ == "__main__":
    demo()
