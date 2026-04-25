#!/usr/bin/env python3
"""
Compare EVIRAG fast mode against the full pipeline on a fixed query set.
"""

import time
from typing import Any, Dict, List, Optional

from evirag_system import EVIRAGConfig, EVIRAGSystem


DEFAULT_QUERIES = [
    "Does homework improve academic achievement?",
    "Is homework effective for science learning outcomes?",
    "What are the trade-offs of homework for student achievement and attitudes?",
]


def _extract_view_summaries(result: Dict[str, Any]) -> List[str]:
    answer = result.get("answer", {})
    views = []
    dominant = answer.get("dominant_view") or {}
    if dominant.get("summary"):
        views.append(dominant["summary"])
    for view in answer.get("alternative_views", []):
        if view.get("summary"):
            views.append(view["summary"])
    for view in answer.get("minority_views", []):
        if view.get("summary"):
            views.append(view["summary"])
    return views


def _extract_source_titles(result: Dict[str, Any]) -> set:
    sources = result.get("sources") or result.get("answer", {}).get("source_documents", [])
    titles = set()
    for source in sources:
        title = source.get("title") or source.get("source_doc_title")
        if title:
            titles.add(title)
    return titles


def _token_jaccard(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def compare_modes(
    queries: Optional[List[str]] = None,
    backend: str = "local",
    enabled_agents: Optional[List[str]] = None,
) -> Dict[str, Any]:
    queries = queries or DEFAULT_QUERIES
    enabled_agents = enabled_agents or ["precision", "recall", "skeptic", "counterfactual"]

    base_config = EVIRAGConfig(
        mode="evirag",
        backend=backend,
        use_visual_grounding=False,
        enabled_agents=enabled_agents,
        depth_vs_speed="fast",
        use_causal_attribution=False,
        use_temporal_tracking=True,
        use_epistemic_divergence=True,
        auto_fallback_to_full=False,
    )

    system = EVIRAGSystem(base_config)
    init_start = time.time()
    system.initialize_corpus()
    init_elapsed = time.time() - init_start

    rows = []
    for query in queries:
        fast_config = EVIRAGConfig(
            **{**base_config.__dict__, "depth_vs_speed": "fast", "auto_fallback_to_full": False}
        )
        full_config = EVIRAGConfig(
            **{**base_config.__dict__, "depth_vs_speed": "balanced", "auto_fallback_to_full": False}
        )

        t0 = time.time()
        fast_result = system.query(query, fast_config)
        fast_elapsed = time.time() - t0

        t1 = time.time()
        full_result = system.query(query, full_config)
        full_elapsed = time.time() - t1

        fast_dom = (fast_result.get("answer", {}).get("dominant_view") or {}).get("summary", "")
        full_dom = (full_result.get("answer", {}).get("dominant_view") or {}).get("summary", "")
        fast_sources = _extract_source_titles(fast_result)
        full_sources = _extract_source_titles(full_result)
        source_overlap = len(fast_sources & full_sources) / max(1, len(fast_sources | full_sources))

        rows.append({
            "query": query,
            "fast_seconds": round(fast_elapsed, 3),
            "full_seconds": round(full_elapsed, 3),
            "speedup": round(full_elapsed / max(fast_elapsed, 1e-6), 2),
            "fast_confidence": fast_result.get("answer", {}).get("confidence_score", 0.0),
            "full_confidence": full_result.get("answer", {}).get("confidence_score", 0.0),
            "fast_relationships": fast_result.get("statistics", {}).get("total_relationships", 0),
            "full_relationships": full_result.get("statistics", {}).get("total_relationships", 0),
            "dominant_view_similarity": round(_token_jaccard(fast_dom, full_dom), 3),
            "source_overlap": round(source_overlap, 3),
            "fast_pipeline": fast_result.get("trace", {}).get("pipeline_mode", "unknown"),
            "fallback_to_full": fast_result.get("statistics", {}).get("fallback_to_full", False),
        })

    return {
        "initialize_seconds": round(init_elapsed, 3),
        "backend": backend,
        "queries": rows,
        "aggregate": {
            "mean_fast_seconds": round(sum(row["fast_seconds"] for row in rows) / max(1, len(rows)), 3),
            "mean_full_seconds": round(sum(row["full_seconds"] for row in rows) / max(1, len(rows)), 3),
            "mean_speedup": round(sum(row["speedup"] for row in rows) / max(1, len(rows)), 2),
            "mean_source_overlap": round(sum(row["source_overlap"] for row in rows) / max(1, len(rows)), 3),
            "mean_view_similarity": round(sum(row["dominant_view_similarity"] for row in rows) / max(1, len(rows)), 3),
        }
    }


if __name__ == "__main__":
    import json
    print(json.dumps(compare_modes(), indent=2))
