"""
EVIRAG FastAPI Service
Backend API for the frontend and external integrations.
"""

from contextlib import asynccontextmanager
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import asyncio
import re
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (
    AGENT_CONFIG,
    CONFIDENCE_CONFIG,
    EMBEDDING_CONFIG,
    METRICS_CONFIG,
    SYSTEM_CONSTRAINTS,
    get_model_config,
)
from evirag_system import EVIRAGSystem, EVIRAGConfig
from mode_comparison import compare_modes


def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy scalars to native Python types.

    FastAPI's jsonable_encoder handles plain Python types fine but
    crashes on numpy.bool_, numpy.float64, numpy.int64 etc.  This
    function is a defensive final-pass sanitiser applied to every
    response before it is returned.
    """
    try:
        import numpy as np
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize(v) for v in obj)
    return obj


class QueryConfig(BaseModel):
    mode: str = "evirag"
    backend: str = "local"
    enabled_agents: Optional[List[str]] = None
    use_visual_grounding: bool = False
    depth_vs_speed: str = "balanced"
    use_epistemic_divergence: bool = True
    use_causal_attribution: bool = False
    use_temporal_tracking: bool = True
    auto_fallback_to_full: bool = True
    rebuild_index: bool = False


class QueryRequest(BaseModel):
    query: str
    config: Optional[QueryConfig] = None


class EvaluationRequest(BaseModel):
    queries: Optional[List[str]] = None
    backend: str = "local"
    enabled_agents: Optional[List[str]] = None


class BatchRequest(BaseModel):
    queries: List[str]
    config: Optional[QueryConfig] = None


system_registry: Dict[Tuple[Any, ...], EVIRAGSystem] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm default system in a thread so the event loop is not blocked.
    default_config = EVIRAGConfig(
        mode="evirag",
        backend="local",
        use_visual_grounding=False,
        depth_vs_speed="fast",
        use_causal_attribution=False,
        auto_fallback_to_full=True,
    )
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_or_create_system, default_config, False)
    yield


app = FastAPI(
    title="EVIRAG API",
    description="Evidence-Centric, Disagreement-Aware RAG for Scientific Literature",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _config_to_evirag(config: Optional[QueryConfig]) -> EVIRAGConfig:
    if config is None:
        return EVIRAGConfig(
            mode="evirag",
            backend="local",
            use_visual_grounding=False,
            depth_vs_speed="balanced",
            use_causal_attribution=False,
            auto_fallback_to_full=True,
        )
    return EVIRAGConfig(
        mode=config.mode,
        backend=config.backend,
        enabled_agents=config.enabled_agents,
        use_visual_grounding=config.use_visual_grounding,
        depth_vs_speed=config.depth_vs_speed,
        use_epistemic_divergence=config.use_epistemic_divergence,
        use_causal_attribution=config.use_causal_attribution,
        use_temporal_tracking=config.use_temporal_tracking,
        auto_fallback_to_full=config.auto_fallback_to_full,
        rebuild_index=config.rebuild_index,
    )


def _system_key(config: EVIRAGConfig) -> Tuple[Any, ...]:
    return (
        config.mode,
        config.backend,
        config.depth_vs_speed,
        config.use_visual_grounding,
        config.use_epistemic_divergence,
        config.use_causal_attribution,
        config.use_temporal_tracking,
    )


def get_or_create_system(config: EVIRAGConfig, rebuild: bool = False) -> EVIRAGSystem:
    key = _system_key(config)
    system = system_registry.get(key)
    if system is None:
        system = EVIRAGSystem(config)
        system.initialize_corpus(rebuild=rebuild or config.rebuild_index)
        system_registry[key] = system
    elif rebuild:
        system.initialize_corpus(rebuild=True)
    return system


def _frontend_config(backend: str = "local") -> EVIRAGConfig:
    return EVIRAGConfig(
        mode="evirag",
        backend=backend,
        use_visual_grounding=False,
        depth_vs_speed="fast",
        use_causal_attribution=False,
        auto_fallback_to_full=True,
    )


def _extract_year(text: Any) -> Optional[int]:
    match = re.search(r"\b(19|20)\d{2}\b", str(text or ""))
    return int(match.group(0)) if match else None


def _display_title(raw_title: Any, path: Any, doc_id: str) -> str:
    title = re.sub(r"\s+", " ", str(raw_title or "")).strip()
    looks_like_artifact = (
        not title
        or title.lower() in {"unknown", "untitled"}
        or title.lower().endswith(".qxd")
        or title.lower().endswith(".indd")
        or title.lower().startswith("microsoft word")
        or bool(re.fullmatch(r"[\w-]{4,}", title)) and not re.search(r"\s", title)
    )
    if not looks_like_artifact:
        return title

    if path:
        stem = Path(str(path)).stem
        cleaned = re.sub(r"[_-]+", " ", stem).strip()
        if cleaned:
            return cleaned.title()

    return f"Document {doc_id}"


def _short_text(text: Any, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def _corpus_stats(system: EVIRAGSystem) -> Dict[str, Any]:
    chunks = system.data_manager.vector_store.chunks
    figures_db = system.data_manager.figures_db or {}
    return {
        "total_chunks": len(chunks),
        "total_documents": len(set(chunk.doc_id for chunk in chunks)),
        "total_claims": len(system.data_manager.claim_graph.claims),
        "total_claim_edges": len(system.data_manager.claim_graph.relationships),
        "total_figures": sum(len(figures) for figures in figures_db.values()),
    }


def _document_records(system: EVIRAGSystem) -> List[Dict[str, Any]]:
    chunks_by_doc: Dict[str, List[Any]] = defaultdict(list)
    for chunk in system.data_manager.vector_store.chunks:
        chunks_by_doc[chunk.doc_id].append(chunk)

    claims_by_doc: Dict[str, List[Any]] = defaultdict(list)
    for claim in system.data_manager.claim_graph.claims:
        claims_by_doc[claim.source_doc_id].append(claim)

    claim_map = system.data_manager.claim_graph.claim_map
    rel_counts: Dict[str, Counter] = defaultdict(Counter)
    for rel in system.data_manager.claim_graph.relationships:
        claim1 = claim_map.get(rel.claim1_id)
        claim2 = claim_map.get(rel.claim2_id)
        doc_ids = {
            claim.source_doc_id
            for claim in (claim1, claim2)
            if claim is not None and claim.source_doc_id
        }
        for doc_id in doc_ids:
            rel_counts[doc_id][rel.relationship] += 1

    docs = []
    for doc_id, chunks in chunks_by_doc.items():
        first = chunks[0]
        meta = first.metadata or {}
        path = meta.get("path")
        title = _display_title(meta.get("title"), path, doc_id)
        sections = Counter(chunk.section or "unknown" for chunk in chunks)
        claims = claims_by_doc.get(doc_id, [])
        page_candidates = [
            chunk.page_num for chunk in chunks
            if chunk.page_num is not None
        ]
        metadata_pages = meta.get("pages")
        pages = int(metadata_pages) if metadata_pages else (
            max(page_candidates) + 1 if page_candidates else 0
        )
        top_claims = sorted(
            claims,
            key=lambda claim: len(claim.text or ""),
            reverse=True,
        )[:3]
        figures = system.data_manager.get_figures_for_doc(doc_id)
        counts = rel_counts.get(doc_id, Counter())
        edge_total = sum(counts.values())
        stance = "neutral"
        if counts.get("contradicts", 0) > counts.get("supports", 0):
            stance = "contra"
        elif counts.get("supports", 0) > 0:
            stance = "support"

        docs.append({
            "id": doc_id,
            "title": title,
            "raw_title": meta.get("title") or "",
            "filename": Path(str(path)).name if path else "",
            "path": path or "",
            "author": meta.get("author") or "",
            "year": meta.get("year") or _extract_year(path) or _extract_year(title),
            "pages": pages,
            "chunks": len(chunks),
            "claims": len(claims),
            "figures": len(figures),
            "sections": [
                {"name": name, "chunks": count}
                for name, count in sections.most_common()
            ],
            "support_edges": counts.get("supports", 0),
            "contradict_edges": counts.get("contradicts", 0),
            "neutral_edges": counts.get("neutral", 0),
            "edge_count": edge_total,
            "stance": stance,
            "is_indexable_content": "access denied" not in (first.text or "").lower(),
            "snippet": _short_text(first.text),
            "top_claims": [
                {
                    "claim_id": claim.claim_id,
                    "text": claim.text,
                    "section": claim.section,
                    "page_num": claim.page_num,
                }
                for claim in top_claims
            ],
            "query": f"What are the main findings in {title}?",
        })

    docs.sort(key=lambda doc: (doc["claims"], doc["chunks"]), reverse=True)
    return docs


def _graph_overview(system: EVIRAGSystem, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    doc_lookup = {doc["id"]: doc for doc in documents}
    claim_map = system.data_manager.claim_graph.claim_map
    clusters = []
    for doc in documents:
        density = (
            doc["contradict_edges"] / doc["edge_count"]
            if doc["edge_count"]
            else 0.0
        )
        if doc["edge_count"] or doc["claims"]:
            clusters.append({
                "doc_id": doc["id"],
                "title": doc["title"],
                "claims": doc["claims"],
                "edges": doc["edge_count"],
                "support_edges": doc["support_edges"],
                "contradict_edges": doc["contradict_edges"],
                "density": density,
                "class": "conflicted" if doc["contradict_edges"] else "aligned" if doc["support_edges"] else "unlinked",
                "query": f"Where does the corpus disagree about {doc['title']}?",
            })

    edge_samples = []
    for rel in sorted(
        system.data_manager.claim_graph.relationships,
        key=lambda item: item.confidence,
        reverse=True,
    )[:20]:
        claim1 = claim_map.get(rel.claim1_id)
        claim2 = claim_map.get(rel.claim2_id)
        if not claim1 or not claim2:
            continue
        doc = doc_lookup.get(claim1.source_doc_id) or doc_lookup.get(claim2.source_doc_id)
        edge_samples.append({
            "source": rel.claim1_id,
            "target": rel.claim2_id,
            "relationship": rel.relationship,
            "confidence": rel.confidence,
            "reasoning": rel.reasoning,
            "source_text": claim1.text,
            "target_text": claim2.text,
            "doc_title": doc["title"] if doc else claim1.source_doc_title,
        })

    return {
        "nodes": len(system.data_manager.claim_graph.claims),
        "edges": len(system.data_manager.claim_graph.relationships),
        "clusters": clusters[:12],
        "edge_samples": edge_samples,
    }


def _agent_status(backend: str = "local") -> List[Dict[str, Any]]:
    labels = {
        "precision": ("P", "Precision", "Strict semantic retrieval for high-confidence supporting evidence."),
        "recall": ("R", "Recall", "Broad retrieval for coverage across the indexed corpus."),
        "skeptic": ("S", "Skeptic", "Contradiction-oriented retrieval and stance pressure testing."),
        "counterfactual": ("C", "Counterfactual", "Alternative explanation retrieval against the current hypothesis."),
    }
    agents = []
    for key, config in AGENT_CONFIG.items():
        model = get_model_config(key, "agent", backend)
        glyph, name, role = labels.get(key, (key[:1].upper(), key.title(), config.get("objective", "")))
        agents.append({
            "key": key,
            "glyph": glyph,
            "name": name,
            "role": role,
            "model": model.name if model else None,
            "backend": backend,
            "retrieval_k": config.get("retrieval_k"),
            "confidence_threshold": config.get("confidence_threshold"),
            "objective": config.get("objective"),
            "search_strategy": config.get("search_strategy"),
            "temperature": model.temperature if model else None,
            "max_tokens": model.max_tokens if model else None,
        })
    return agents


def _topics_from_corpus(documents: List[Dict[str, Any]], graph: Dict[str, Any]) -> Dict[str, Any]:
    topic_docs = [
        doc for doc in documents
        if doc.get("is_indexable_content") and doc.get("claims", 0) >= 5
    ]
    top_docs = topic_docs[:8]
    document_items = [
        {
            "q": doc["query"],
            "meta": f"{doc['claims']} claims · {doc['chunks']} chunks · {doc['figures']} figures",
            "title": doc["title"],
            "doc_id": doc["id"],
            "level": "med" if doc["edge_count"] else "low",
        }
        for doc in top_docs
    ]
    disagreement_items = [
        {
            "q": cluster["query"],
            "meta": f"{cluster['claims']} claims · {cluster['edges']} edges · density {cluster['density'] * 100:.1f}%",
            "title": cluster["title"],
            "doc_id": cluster["doc_id"],
            "level": "high" if cluster["contradict_edges"] else "med" if cluster["support_edges"] else "low",
        }
        for cluster in graph["clusters"][:8]
        if any(doc["id"] == cluster["doc_id"] for doc in topic_docs)
    ]
    claim_items = []
    for doc in top_docs:
        for claim in doc["top_claims"][:1]:
            claim_items.append({
                "q": f"What evidence supports this claim: {claim['text']}",
                "meta": f"{doc['title']} · {claim.get('section') or 'unknown section'}",
                "title": doc["title"],
                "doc_id": doc["id"],
                "claim_id": claim["claim_id"],
                "level": doc["stance"] == "contra" and "high" or "med",
            })
    return {
        "tabs": [
            {"id": "documents", "label": "Documents", "items": document_items},
            {"id": "disagreement", "label": "Disagreement", "items": disagreement_items},
            {"id": "claims", "label": "Claims", "items": claim_items[:8]},
        ]
    }


def _calibration_config() -> List[Dict[str, Any]]:
    labels = {
        "claim_agreement": "Claim agreement",
        "source_diversity": "Source diversity",
        "contradiction_severity": "Contradiction severity",
        "unverified_assumptions": "Unverified assumptions",
        "visual_alignment": "Visual alignment",
    }
    return [
        {"key": key, "label": labels.get(key, key.replace("_", " ").title()), "weight": weight}
        for key, weight in CONFIDENCE_CONFIG["factors"].items()
    ]


def _query_trace_plan(query: str, config: EVIRAGConfig, system: EVIRAGSystem) -> Dict[str, Any]:
    stats = _corpus_stats(system)
    graph_ready = system.data_manager.claim_graph.is_ready()
    visual_available = bool(getattr(system, "visual_analyzer", None))

    def step(key: str, label: str, detail: str) -> Dict[str, str]:
        return {"step": key, "label": label, "detail": detail}

    if config.mode == "vanilla_rag":
        steps = [
            step(
                "vector_retrieval",
                "Searching corpus chunks",
                f"Embedding the query and ranking passages across {stats['total_chunks']} indexed chunks.",
            ),
            step(
                "single_view_synthesis",
                "Synthesizing single-view answer",
                "Sending the retrieved source context to the configured synthesizer model.",
            ),
        ]
        pipeline = "vanilla_rag"
        path_label = "corpus retrieval"
        mode_label = "Vanilla RAG"
    elif config.depth_vs_speed == "fast" and graph_ready:
        steps = [
            step(
                "offline_claim_graph_lookup",
                "Retrieving claim subgraph",
                f"Searching {stats['total_claims']} indexed claims and {stats['total_claim_edges']} graph edges.",
            ),
            step(
                "fast_intent_inference",
                "Classifying epistemic intent",
                "Inferring expected disagreement from the retrieved graph neighborhood.",
            ),
            step(
                "heuristic_hypothesis",
                "Building X-IR hypothesis",
                "Creating the initial hypothesis frame from the user query.",
            ),
            step(
                "disagreement_reasoning",
                "Reasoning over claim relationships",
                "Separating support, contradiction, and neutral edges for the answer views.",
            ),
        ]
        if config.use_epistemic_divergence:
            steps.append(step(
                "epistemic_divergence",
                "Calibrating viewpoint spread",
                "Computing divergence over the returned claim viewpoints when embeddings are available.",
            ))
        if config.use_temporal_tracking:
            steps.append(step(
                "temporal_drift",
                "Checking temporal drift",
                "Estimating whether the retrieved disagreement is stable, narrowing, or open.",
            ))
        steps.append(step(
            "multi_view_answer",
            "Preparing multi-view answer",
            "Formatting answer, graph, claims, visual, and source panes from the backend result.",
        ))
        pipeline = "fast_claim_graph"
        path_label = "fast claim graph"
        mode_label = "EVIRAG"
    else:
        enabled_agents = config.enabled_agents or list(AGENT_CONFIG.keys())
        steps = [
            step(
                "intent_analysis",
                "Classifying epistemic intent",
                "Running the epistemic engine to estimate domain and disagreement level.",
            ),
            step(
                "hypothesis_generation",
                "Generating X-IR hypothesis",
                "Producing the central hypothesis, assumptions, and expected counterclaims.",
            ),
            step(
                "multi_agent_retrieval",
                "Running deliberative retrieval agents",
                f"Launching {len(enabled_agents)} configured agents in parallel over {stats['total_documents']} corpus documents (max concurrency {SYSTEM_CONSTRAINTS['max_concurrent_agents']}).",
            ),
            step(
                "claim_consolidation",
                "Consolidating atomic claims",
                "Deduplicating claims extracted by the retrieval agents.",
            ),
            step(
                "claim_relationship_building",
                "Building claim relationships",
                "Classifying support, contradiction, and neutral relationships between claims.",
            ),
        ]
        if config.use_visual_grounding:
            steps.append(step(
                "visual_evidence_analysis",
                "Aligning visual evidence",
                "Running visual-text mismatch analysis over returned claims and indexed figures."
                if visual_available else
                "Visual grounding was requested, but no visual analyzer is available for this backend configuration.",
            ))
        steps.append(step(
            "disagreement_synthesis",
            "Synthesizing disagreement-aware answer",
            "Building dominant, alternative, and minority views from the verified claim graph.",
        ))
        if config.use_epistemic_divergence:
            steps.append(step(
                "epistemic_divergence",
                "Calibrating viewpoint spread",
                "Computing ED, polarization, and consensus-collapse metrics for the returned views.",
            ))
        if config.use_temporal_tracking:
            steps.append(step(
                "temporal_drift",
                "Checking temporal drift",
                "Estimating how the disagreement evolves across the retrieved corpus metadata.",
            ))
        steps.append(step(
            "multi_view_answer",
            "Preparing multi-view answer",
            "Formatting answer, graph, claims, visual, and source panes from the backend result.",
        ))
        pipeline = "full_evirag"
        path_label = "full graph + visual" if config.use_visual_grounding else "full graph + agents"
        mode_label = "EVIRAG"

    return {
        "query": query,
        "mode": config.mode,
        "mode_label": mode_label,
        "requested_speed": config.depth_vs_speed,
        "pipeline_mode": pipeline,
        "path_label": path_label,
        "visual_grounding_requested": config.use_visual_grounding,
        "visual_grounding_available": visual_available,
        "corpus": stats,
        "steps": steps,
    }


@app.get("/")
async def root():
    return {
        "service": "EVIRAG API",
        "version": "2.0.0",
        "systems_loaded": len(system_registry),
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if system_registry else "initializing",
        "systems_loaded": len(system_registry),
    }


@app.post("/api/initialize")
async def initialize_system(config: Optional[QueryConfig] = None):
    exec_config = _config_to_evirag(config)
    system = get_or_create_system(exec_config, rebuild=exec_config.rebuild_index)
    return {
        "status": "ready",
        "config": exec_config.__dict__,
        "system": system.get_system_status(),
    }


@app.get("/api/system/status")
async def system_status(
    backend: str = "local",
    use_visual_grounding: bool = False,
    use_epistemic_divergence: bool = True,
    use_causal_attribution: bool = False,
    use_temporal_tracking: bool = True,
):
    config = EVIRAGConfig(
        backend=backend,
        use_visual_grounding=use_visual_grounding,
        use_epistemic_divergence=use_epistemic_divergence,
        use_causal_attribution=use_causal_attribution,
        use_temporal_tracking=use_temporal_tracking,
    )
    system = get_or_create_system(config)
    return system.get_system_status()


@app.post("/api/process_query")
async def process_query(request: QueryRequest):
    exec_config = _config_to_evirag(request.config)
    try:
        system = get_or_create_system(exec_config, rebuild=exec_config.rebuild_index)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, system.query, request.query, exec_config
        )
        return _sanitize(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/query_trace_plan")
async def query_trace_plan(request: QueryRequest):
    exec_config = _config_to_evirag(request.config)
    try:
        system = get_or_create_system(exec_config, rebuild=exec_config.rebuild_index)
        return _sanitize(_query_trace_plan(request.query, exec_config, system))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/rebuild_index")
async def rebuild_index(config: Optional[QueryConfig] = None):
    exec_config = _config_to_evirag(config)
    try:
        system = get_or_create_system(exec_config, rebuild=True)
        return {
            "status": "success",
            "message": "Index rebuilt",
            "system": system.get_system_status(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Visual grounding endpoints ─────────────────────────────────────────────

@app.get("/api/visual/figure/{figure_id}")
async def visual_figure_image(figure_id: str, backend: str = "local"):
    """Serve a figure image extracted from the PDF corpus.
    
    Resolves the image by figure_id against the LOCAL FIGURES_DIR so stale
    paths cached in visual_evidence.pkl don't break the endpoint.
    """
    from config import FIGURES_DIR
    # 1. Try to find the image directly by figure_id in FIGURES_DIR
    local_path = FIGURES_DIR / f"{figure_id}.png"
    if local_path.exists():
        return FileResponse(str(local_path), media_type="image/png", filename=local_path.name)

    # 2. Fall back to whatever the processor recorded (may be stale on another machine)
    config = EVIRAGConfig(
        backend=backend,
        use_visual_grounding=True,
        depth_vs_speed="fast",
    )
    system = get_or_create_system(config)
    if not system.visual_processor:
        raise HTTPException(status_code=404, detail="Visual grounding not enabled")
    ve = system.visual_processor.visual_evidence_db.get(figure_id)
    if not ve:
        raise HTTPException(status_code=404, detail=f"Figure {figure_id} not found")
    image_path = FIGURES_DIR / Path(str(ve.image_path)).name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file missing: {image_path.name}")
    return FileResponse(str(image_path), media_type="image/png", filename=image_path.name)


@app.get("/api/visual/evidence")
async def visual_evidence_index(backend: str = "local"):
    """List all indexed visual evidence entries with corrected local paths."""
    from config import FIGURES_DIR
    config = EVIRAGConfig(
        backend=backend,
        use_visual_grounding=True,
        depth_vs_speed="fast",
    )
    system = get_or_create_system(config)
    if not system.visual_processor:
        return _sanitize({"figures": [], "total": 0})
    entries = []
    for ve in system.visual_processor.visual_evidence_db.values():
        # Resolve image_path to local machine regardless of cached value
        fname = Path(str(ve.image_path)).name
        local_path = FIGURES_DIR / fname
        # Resolve dimensions from actual file if not cached
        w = getattr(ve, 'width', 0) or 0
        h = getattr(ve, 'height', 0) or 0
        if (w == 0 or h == 0) and local_path.exists():
            try:
                from PIL import Image as _Img
                img = _Img.open(local_path)
                w, h = img.size
            except Exception:
                pass
        entries.append({
            "figure_id": ve.figure_id,
            "doc_id": ve.doc_id,
            "caption_text": ve.caption_text,
            "image_path": str(local_path),
            "image_url": f"/api/visual/figure/{ve.figure_id}",
            "page_num": ve.page_num,
            "width": w,
            "height": h,
            "exists": local_path.exists(),
        })
    return _sanitize({"figures": entries, "total": len(entries)})


@app.get("/api/corpus/stats")
async def corpus_stats(backend: str = "local"):
    config = _frontend_config(backend=backend)
    system = get_or_create_system(config)
    return _sanitize(_corpus_stats(system))


@app.get("/api/corpus/documents")
async def corpus_documents(backend: str = "local"):
    config = _frontend_config(backend=backend)
    system = get_or_create_system(config)
    return _sanitize({
        "stats": _corpus_stats(system),
        "documents": _document_records(system),
    })


@app.get("/api/corpus/documents/{doc_id}/pdf")
async def corpus_document_pdf(doc_id: str, backend: str = "local"):
    config = _frontend_config(backend=backend)
    system = get_or_create_system(config)
    for doc in _document_records(system):
        if doc["id"] == doc_id and doc.get("path"):
            path = Path(doc["path"])
            if path.exists():
                return FileResponse(str(path), media_type="application/pdf", filename=path.name)
    raise HTTPException(status_code=404, detail="Document PDF not found")


@app.get("/api/graph/overview")
async def graph_overview(backend: str = "local"):
    config = _frontend_config(backend=backend)
    system = get_or_create_system(config)
    documents = _document_records(system)
    return _sanitize(_graph_overview(system, documents))


@app.get("/api/agents/status")
async def agents_status(backend: str = "local"):
    return _sanitize({
        "backend": backend,
        "agents": _agent_status(backend),
        "embedding_model": EMBEDDING_CONFIG["text_model"],
    })


@app.get("/api/ui/bootstrap")
async def ui_bootstrap(backend: str = "local"):
    config = _frontend_config(backend=backend)
    system = get_or_create_system(config)
    documents = _document_records(system)
    graph = _graph_overview(system, documents)
    return _sanitize({
        "stats": _corpus_stats(system),
        "documents": documents,
        "graph": graph,
        "topics": _topics_from_corpus(documents, graph),
        "agents": _agent_status(backend),
        "calibration": _calibration_config(),
        "models": {
            "backend": backend,
            "embedding": EMBEDDING_CONFIG["text_model"],
            "clip": EMBEDDING_CONFIG["clip_model"],
        },
        "metrics": {
            key: enabled
            for key, enabled in METRICS_CONFIG.items()
        },
    })


@app.post("/api/batch_process")
async def batch_process(request: BatchRequest):
    exec_config = _config_to_evirag(request.config)
    queries = request.queries
    system = get_or_create_system(exec_config, rebuild=exec_config.rebuild_index)
    results = []
    for query in queries:
        try:
            results.append({
                "query": query,
                "status": "success",
                "result": system.query(query, exec_config),
            })
        except Exception as exc:
            results.append({
                "query": query,
                "status": "error",
                "error": str(exc),
            })
    return {
        "total": len(queries),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "error"]),
        "results": results,
    }


@app.post("/api/evaluate_modes")
async def evaluate_modes(request: EvaluationRequest):
    try:
        return compare_modes(
            queries=request.queries,
            backend=request.backend,
            enabled_agents=request.enabled_agents,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def run_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
