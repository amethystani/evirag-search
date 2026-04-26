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
    if isinstance(obj, str):
        # Strip control characters (ASCII 0-8, 11-31 except \t \n \r) that break JSON
        return "".join(c for c in obj if ord(c) >= 32 or c in "\t\n\r")
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


# ══════════════════════════════════════════════════════════════════════════════
# PERPLEXITY-STYLE CHAT ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

from uuid import uuid4

class ChatMessage(BaseModel):
    role: str          # "user" | "assistant"
    content: str
    sources: Optional[List[Dict]] = None  # attached citations for assistant turns
    claim: Optional[str] = None           # evolving argument/claim for assistant turns
    turn: int = 0

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None     # None = create new session
    backend: str = "local"               # "local" = Ollama, "cloud" = Groq
    agents: bool = False                 # UI toggle for 4 agents
    vlm: bool = False                    # UI toggle for visual grounding

class ChatResponse(BaseModel):
    session_id: str
    answer: str                          # Perplexity-style markdown with [N] citations
    claim: str                           # evolving synthesized argument
    sources: List[Dict]                  # [{n, title, year, snippet, doi, source}]
    turn: int
    latency_ms: float
    history: List[Dict]                  # full session history

# In-memory session store {session_id: {history, accumulated_context, claim, acc_claims, acc_graph}}
_chat_sessions: Dict[str, Dict] = {}


def _synthesize_view_reason(query: str, snippets: List[str], stance: str) -> str:
    """
    Ask the cloud model to write ONE sentence explaining why the retrieved papers
    support or challenge the query. Returns a reasoned string, never raw text.
    """
    import re as _re
    from ollama_cloud_client import get_cloud_client
    snip_text = " | ".join(s[:180] for s in snippets[:3] if s.strip())
    if not snip_text:
        return f"No papers directly address the {stance} position on this topic."

    if stance == "dominant":
        instr = "In ONE sentence, explain what these papers suggest supports or confirms the main position on this topic."
    else:
        instr = "In ONE sentence, explain what alternative or contrasting perspective these papers raise about this topic."

    system_msg = "You are a scientific analyst. Be concise. Do NOT copy paper titles verbatim. Your response MUST be based ONLY on the provided papers. Do not use outside knowledge."
    user_msg   = f"Topic: {query}\nPapers: {snip_text}\n\n{instr}"

    try:
        client = get_cloud_client()
        text = client.generate(
            prompt=user_msg,
            system=system_msg,
            temperature=0.25,
            max_tokens=100,
        )
        text = text.strip()
        # Strip Evolving Claim leakage and control characters / newlines
        text = text.split("**Evolving Claim")[0]
        text = _re.sub(r'[\x00-\x1f\x7f]+', ' ', text)
        text = _re.sub(r' {2,}', ' ', text).strip()
        # Trim to the last complete sentence
        for end in ['. ', '! ', '? ']:
            last = text.rfind(end)
            if last > len(text) // 2:
                text = text[:last + 1]
                break
        text = text.rstrip('.,; ')
        if not text:
            return f"These papers suggest a {stance} perspective on the topic."
        if not text[-1] in '.!?':
            text += '.'
        return text[:240]
    except Exception as exc:
        print(f"[chat] Ollama cloud error (view reason): {exc}")
        return f"These papers suggest a {stance} perspective on the topic."


def _local_chat_synthesize(
    sources_block: str,
    dom_view: str,
    alt_view: str,
    prev_claim: str,
    message: str,
    prev_turns: str,
    chat_history: Optional[List[Dict]] = None,
) -> str:
    """
    Synthesize a Perplexity-style answer using the Ollama cloud model (gpt-oss:120b).
    Passes the full conversation history as proper multi-turn messages so the model
    maintains context across turns (not just a text summary of previous questions).
    """
    from ollama_cloud_client import get_cloud_client
    has_ctx = bool(sources_block.strip())

    if has_ctx:
        system_msg = (
            "You are EVIRAG, an expert scientific research assistant like Perplexity AI. "
            "You are in a multi-turn conversation — always maintain full context from previous turns and answer follow-up questions in that context. "
            "Write a thorough, well-structured answer of at least 5-6 sentences. Cover key evidence, mechanisms, and nuances. Never give a one-sentence or short answer. "
            "Retrieved passages are provided below. Use [N] to cite a passage ONLY when it directly supports a specific claim. "
            "IMPORTANT: If the retrieved passages are not relevant to the question, IGNORE them entirely and answer from your scientific knowledge. "
            "Do NOT copy paper titles verbatim. "
            "End your response with a new line: '**Evolving Claim:**' followed by a one-sentence synthesis of the key finding."
        )
    else:
        system_msg = (
            "You are EVIRAG, an expert scientific research assistant like Perplexity AI. "
            "You are in a multi-turn conversation — always maintain full context from previous turns and answer follow-up questions in that context. "
            "Write a thorough, well-structured answer of at least 5-6 sentences using established scientific knowledge. "
            "Cover mechanisms, current scientific consensus, evidence, and nuances. Never give a short answer. "
            "Do NOT use [N] citation markers or fabricate references. "
            "End your response with a new line: '**Evolving Claim:**' followed by a one-sentence synthesis of the key finding."
        )

    # Build proper multi-turn message list so the model has real conversation context
    messages = [{"role": "system", "content": system_msg}]

    # Inject previous turns as actual assistant/user messages (not just a text stub)
    if chat_history:
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Build this turn's user message
    user_parts = []
    if has_ctx:
        user_parts.append(
            f"Retrieved passages (cite with [N] when directly applicable):\n{sources_block[:3000]}"
        )
    user_parts.append(message)
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})

    try:
        client = get_cloud_client()
        data = client.chat(
            messages=messages,
            temperature=0.35,
            max_tokens=2048,
        )
        content = data.get("message", {}).get("content", "")
        if isinstance(content, str):
            import re as _re
            content = _re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
    except Exception as e:
        print(f"[chat] Ollama cloud error: {e}")
        return ""

# Backward-compat alias
_cloud_chat_synthesize = _local_chat_synthesize


def _fetch_faiss_sources(query: str, k: int = 8) -> tuple[List[Dict], float]:
    """
    Query the FAISS search backend (port 7860) for relevant papers.
    Returns (sources_list, relevance_score 0-1).
    Falls back to ([], 0.5) if backend unreachable.
    """
    import requests as _req
    try:
        resp = _req.post(
            "http://localhost:7860/search",
            json={"query": query, "k": k},
            timeout=25,
        )
        resp.raise_for_status()
        data = resp.json()
        papers = data.get("results", [])
        sources: List[Dict] = []
        scores: List[float] = []
        for p in papers:
            excerpt = (p.get("text_excerpt") or "").strip()
            sources.append({
                "n":       p["rank"],
                "title":   p.get("title") or "Unknown",
                "year":    p.get("year"),
                "snippet": excerpt[:300],
                "doi":     p.get("doi") or "",
                "source":  p.get("source") or "",
            })
            scores.append(float(p.get("score", 768)))
        # Hamming distance: 0=identical, 768=random. Normalize to [0,1]
        # Use top-1 distance (not mean) — mean is inflated by borderline papers.
        # Threshold: dist < ~200 (relevance > 0.74) = genuinely on-corpus query.
        top_dist = min(scores) if scores else 768
        relevance = round(max(0.0, 1.0 - top_dist / 768), 3)
        return sources, relevance
    except Exception as e:
        print(f"[chat] FAISS backend unavailable: {e}")
        return [], 0.5


def _build_perplexity_answer(
    message: str,
    evirag_result: Dict,
    session: Dict,
    turn: int,
    faiss_sources: Optional[List[Dict]] = None,
    faiss_relevance: float = 0.0,
) -> tuple[str, str, List[Dict]]:
    """
    Synthesize a Perplexity-style answer.
    Prefers faiss_sources (peS2o 100k corpus) over EVIRAG's local PDF corpus.
    Returns (answer_markdown, evolving_claim, sources_list)
    """
    import re

    # ── Prefer FAISS sources; fall back to EVIRAG claims/sources ─────────────
    if faiss_sources:
        sources = faiss_sources
    else:
        raw_claims  = evirag_result.get("claims", [])
        raw_sources = evirag_result.get("sources", [])
        seen: set = set()
        sources = []
        for src in raw_sources:
            title = src.get("title") or src.get("source_doc_title") or "Unknown"
            if title in seen: continue
            seen.add(title)
            sources.append({
                "n":       len(sources) + 1,
                "title":   title,
                "year":    src.get("year"),
                "snippet": (src.get("text") or src.get("snippet") or "")[:300],
                "doi":     src.get("doi") or "",
                "source":  src.get("source") or src.get("venue") or "",
            })
            if len(sources) >= 12: break
        if not sources:
            seen = set()
            for claim in raw_claims[:12]:
                title = claim.get("source_doc_title") or "Unknown"
                if title in seen: continue
                seen.add(title)
                sources.append({
                    "n": len(sources) + 1, "title": title, "year": None,
                    "snippet": claim.get("text", "")[:300], "doi": "", "source": "",
                })

    # ── Build citation index for the LLM prompt ───────────────────────────────
    # Only pass snippets (not titles) to the LLM — small models copy titles verbatim.
    # Only include sources when FAISS relevance is high enough to be useful.
    # Gate sources for LLM: distance threshold (rel > 0.70) AND keyword overlap.
    # Expand hyphenated terms and use 3+ char words to avoid over-filtering.
    RELEVANCE_THRESHOLD = 0.70
    _raw = set()
    for w in message.lower().split():
        wc = w.strip("?!.,;:")
        _raw.add(wc)
        for part in wc.split("-"):
            if len(part) >= 3:
                _raw.add(part)
    query_terms = {t for t in _raw if len(t) >= 3}
    all_snippet_text = " ".join(
        (s.get("snippet") or "").lower() for s in sources[:6]
    )
    keyword_match = not query_terms or any(t in all_snippet_text for t in query_terms)
    has_relevant_sources = (
        faiss_relevance >= RELEVANCE_THRESHOLD
        and keyword_match
        and bool([s for s in sources[:8] if s.get("snippet")])
    )

    if has_relevant_sources:
        src_block = "\n".join(
            f"[{s['n']}] {s['snippet'][:250]}"
            for s in sources[:6] if s.get("snippet")
        )
    else:
        src_block = ""  # corpus has no directly relevant papers — LLM answers from knowledge

    # Derive dom/alt view from FAISS snippets if relevant, else from EVIRAG result
    if faiss_sources and has_relevant_sources:
        dom_view = " ".join(
            s["snippet"][:150] for s in sources[:3] if s.get("snippet")
        )
        alt_view = sources[3]["snippet"][:150] if len(sources) > 3 and sources[3].get("snippet") else ""
    else:
        answer_dict = evirag_result.get("answer", {})
        dom_view = (answer_dict.get("dominant_view") or {}).get("summary", "")
        alt_views = [(v.get("summary") or "") for v in (answer_dict.get("alternative_views") or [])]
        min_views = [(v.get("summary") or "") for v in (answer_dict.get("minority_views") or [])]
        alt_view  = alt_views[0] if alt_views else (min_views[0] if min_views else "")

    prev_claim = session.get("claim", "")
    history    = session.get("history", [])

    # Build actual chat message history (user + assistant pairs) for multi-turn context.
    # The 120B model can properly follow conversation when given real message history
    # rather than a truncated text summary of previous questions.
    chat_history_messages: List[Dict] = []
    for h in history[-4:]:   # last 4 turns = last 8 messages (4 user + 4 assistant)
        chat_history_messages.append({"role": "user",      "content": h["user"]})
        chat_history_messages.append({"role": "assistant", "content": h["answer"]})

    # prev_turns kept for legacy fallback only
    prev_turns = "First turn." if not history else f"Q: {history[-1]['user'][:80]}"

    answer_md = _local_chat_synthesize(
        sources_block=src_block,
        dom_view=dom_view,
        alt_view=alt_view,
        prev_claim=prev_claim,
        message=message,
        prev_turns=prev_turns,
        chat_history=chat_history_messages,
    )

    if not answer_md:
        answer_md = (
            f"{dom_view}\n\n"
            f"**Conflicting Evidence:** {alt_view or 'None found.'}\n\n"
            f"**Evolving Claim:** {dom_view[:200]}"
        )

    # Match "**Evolving Claim:**" or "Evolving Claim:" (Qwen sometimes omits bold markers)
    claim_match = re.search(r'\*{0,2}Evolving Claim:\*{0,2}\s*(.+?)(?:\n|$)', answer_md, re.DOTALL)
    if claim_match:
        claim = claim_match.group(1).strip()
    else:
        # Fall back to the last sentence of the answer body as the claim
        sentences = [s.strip() for s in re.split(r'[.!?]+', answer_md) if s.strip()]
        claim = sentences[-1] if sentences else message

    # Strip the trailing Evolving Claim block from the answer body —
    # it's rendered separately as the styled .evolving-claim element in the UI.
    answer_body = re.sub(r'(?i)(?:The best(?:-| )supported position is:)?\s*\*{0,2}Evolving Claim:\*{0,2}.*$', '', answer_md, flags=re.DOTALL).strip()

    return answer_body, claim, sources


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Perplexity-style multi-turn chat backed by EVIRAG's 4-agent system.

    Each turn:
      1. Merges conversation history → enriched query
      2. Runs EVIRAG fast pipeline (claim graph + disagreement)
      3. Synthesizes Perplexity-style answer with inline [N] citations via Groq
      4. Returns evolving claim that refines across turns
    """
    import time
    t0 = time.perf_counter()

    # ── Session management ─────────────────────────────────────────────────────
    session_id = request.session_id or str(uuid4())
    if session_id not in _chat_sessions:
        _chat_sessions[session_id] = {
            "history":              [],
            "accumulated_context":  "",
            "claim":                "",
            "turn":                 0,
            "acc_claims":           [],     # union of all retrieved claims across turns
            "acc_sources":          {},     # {doc_id: source_dict} deduped across turns
        }
    session = _chat_sessions[session_id]
    turn    = session["turn"] + 1

    # ── Build enriched query (accumulate context across turns) ─────────────────
    acc = session["accumulated_context"]
    if acc:
        enriched_query = f"{acc}. Now specifically: {request.message}"
    else:
        enriched_query = request.message

    # Keep accumulated context short (last 3 topics)
    new_acc = f"{acc}; {request.message}" if acc else request.message
    words   = new_acc.split()
    session["accumulated_context"] = " ".join(words[-120:])  # trim to ~120 words

    # ── Run EVIRAG pipeline ───────────────────────────────────────────────────
    try:
        config = EVIRAGConfig(
            mode="evirag",
            backend=request.backend,
            use_visual_grounding=request.vlm,
            depth_vs_speed="fast",
            enabled_agents=["precision", "recall", "skeptic", "counterfactual"] if request.agents else [],
            use_epistemic_divergence=True,
            use_causal_attribution=False,
            use_temporal_tracking=True,
            auto_fallback_to_full=request.agents,    # Allow fallback if agents are enabled
        )
        system  = get_or_create_system(config)
        loop    = asyncio.get_event_loop()
        result  = await loop.run_in_executor(None, system.query, enriched_query, config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EVIRAG pipeline error: {e}")

    # ── Fetch relevant papers from peS2o FAISS index (port 7860) ────────────
    # For short follow-up questions (< 5 words like "what about mars"), use the
    # enriched query so FAISS gets planetary context rather than "mars" the verb.
    # For longer self-contained questions, use request.message to avoid drift.
    _faiss_query = (
        enriched_query
        if len(request.message.split()) < 5
        else request.message
    )
    faiss_sources, faiss_relevance = _fetch_faiss_sources(_faiss_query)

    # ── Compute relevance gate ────────────────────────────────────────────────
    # Two thresholds:
    #   0.70 + keyword → feed sources to LLM (permissive, allow broad retrieval)
    #   0.75           → show views in UI (strict, only genuinely on-corpus queries)
    _raw_terms = set()
    for w in request.message.lower().split():
        w_clean = w.strip("?!.,;:")
        _raw_terms.add(w_clean)
        for part in w_clean.split("-"):  # "fine-tuning" → "fine", "tuning"
            if len(part) >= 3:
                _raw_terms.add(part)
    _query_terms = {t for t in _raw_terms if len(t) >= 3}
    _all_snips   = " ".join((s.get("snippet") or "").lower() for s in faiss_sources[:6])
    _kw_match    = not _query_terms or any(t in _all_snips for t in _query_terms)
    has_relevant_sources = (
        faiss_relevance >= 0.70
        and _kw_match
        and bool([s for s in faiss_sources[:8] if s.get("snippet")])
    )
    # Stricter gate for views — avoids showing views from tangentially matching papers
    # (e.g. "flat" appears in "flat norm" papers for "is earth flat" query)
    views_relevant = faiss_relevance >= 0.75 and has_relevant_sources

    # ── Patch EVIRAG result with FAISS corpus data ────────────────────────────
    # Always clear EVIRAG homework-PDF views — they must never reach the UI.
    # Views are re-added below only when FAISS finds genuinely relevant papers.
    answer_obj = result.setdefault("answer", {})
    answer_obj.pop("dominant_view", None)
    answer_obj.pop("alternative_views", None)
    answer_obj["minority_views"] = []

    if faiss_sources:
        # Replace flat sources list with peS2o FAISS results
        result["sources"] = [
            {
                "doc_id":   s.get("doi") or s["title"][:12],
                "title":    s["title"],
                "year":     s.get("year"),
                "doi":      s.get("doi") or "",
                "source":   s.get("source") or "",
                "snippet":  s.get("snippet") or "",
                "text":     s.get("snippet") or "",
                "cited_by_count": None,
            }
            for s in faiss_sources
        ]

        def _view_from_sources(srcs, name, stance="dominant"):
            snips = [s["snippet"] for s in srcs if s.get("snippet")]
            if not snips:
                return None
            reason = _synthesize_view_reason(enriched_query, snips, stance)
            return {
                "name":         name,
                "summary":      reason,
                "source_count": len(snips),
                "confidence":   0.72,
                "supporting_claim_ids": [],
                "citations": [
                    {"citation_label": s["title"][:60], "text": s.get("snippet", "")[:200]}
                    for s in srcs[:3]
                ],
            }

        # Only show views when FAISS finds genuinely topic-relevant papers (rel ≥ 0.75)
        if views_relevant:
            dom = _view_from_sources(faiss_sources[:3],  "Most relevant (peS2o corpus)", "dominant")
            alt = _view_from_sources(faiss_sources[3:6], "Related perspective",          "alternative")
            if dom:
                answer_obj["dominant_view"] = dom
            if alt:
                answer_obj["alternative_views"] = [alt]

        result.setdefault("statistics", {})["total_sources"] = len(faiss_sources)

    # ── Synthesize Perplexity-style answer ────────────────────────────────────
    answer_md, claim, sources = _build_perplexity_answer(
        message=request.message,
        evirag_result=result,
        session=session,
        turn=turn,
        faiss_sources=faiss_sources if faiss_sources else None,
        faiss_relevance=faiss_relevance,
    )

    # ── Accumulate claims and sources across turns ────────────────────────────
    # Claims from this turn (from EVIRAG claim graph)
    new_claims = result.get("claims", [])
    seen_claim_ids = {c.get("id") or c.get("text", "")[:40] for c in session["acc_claims"]}
    for c in new_claims:
        cid = c.get("id") or c.get("text", "")[:40]
        if cid and cid not in seen_claim_ids:
            session["acc_claims"].append(c)
            seen_claim_ids.add(cid)
    # Keep last 60 unique claims
    session["acc_claims"] = session["acc_claims"][-60:]

    # Sources from this turn (FAISS) — deduplicate by title
    for s in sources:
        key = (s.get("title") or "")[:50]
        if key and key not in session["acc_sources"]:
            session["acc_sources"][key] = s
    # Keep last 40 unique sources
    if len(session["acc_sources"]) > 40:
        keys = list(session["acc_sources"].keys())
        session["acc_sources"] = {k: session["acc_sources"][k] for k in keys[-40:]}

    # Merge accumulated claims into the result claims list for the UI
    # (so claim count grows turn by turn)
    result["claims"] = session["acc_claims"]

    # ── Update session history ────────────────────────────────────────────────
    session["history"].append({
        "turn":    turn,
        "user":    request.message,
        "answer":  answer_md,
        "claim":   claim,
        "sources": [s["title"] for s in sources[:5]],
    })
    session["claim"] = claim
    session["turn"]  = turn
    # Limit session history to last 20 turns
    if len(session["history"]) > 20:
        session["history"] = session["history"][-20:]

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Accumulated source list for the UI (union of all turns)
    acc_sources_list = list(session["acc_sources"].values())

    # Inject the chat context into the full EVIRAG result
    result["chat"] = {
        "session_id":  session_id,
        "answer":      answer_md,
        "claim":       claim,
        "sources":     acc_sources_list,  # full accumulated source list
        "turn":        turn,
        "latency_ms":  latency_ms,
        "total_claims": len(session["acc_claims"]),
        "total_sources": len(acc_sources_list),
        "history": [
            msg
            for h in session["history"]
            for msg in (
                {"role": "user",      "content": h["user"],   "turn": h["turn"]},
                {"role": "assistant", "content": h["answer"], "turn": h["turn"],
                 "claim": h["claim"], "sources": h["sources"]},
            )
        ]
    }

    return _sanitize(result)


@app.delete("/api/chat/{session_id}")
async def delete_chat_session(session_id: str):
    """Clear a chat session."""
    _chat_sessions.pop(session_id, None)
    return {"status": "deleted", "session_id": session_id}


@app.get("/api/chat/sessions")
async def list_chat_sessions():
    """List all active chat sessions."""
    return {
        "sessions": [
            {"session_id": sid, "turns": s["turn"], "claim": s["claim"][:100]}
            for sid, s in _chat_sessions.items()
        ]
    }


if __name__ == "__main__":
    run_server()

