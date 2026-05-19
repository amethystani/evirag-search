"""
EVIRAG Streamlit Frontend
Talks to the FastAPI backend via HTTP — no direct model imports.
"""

import html
import time
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EVIRAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; padding-top: 1.8rem; padding-bottom: 3rem;}
    .mode-chip {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: rgba(100, 149, 237, 0.12);
        border: 1px solid rgba(100, 149, 237, 0.35);
        font-size: 0.8rem;
        margin-right: 0.4rem;
    }
    .source-card {
        padding: 0.8rem 0.9rem;
        border-radius: 0.8rem;
        border: 1px solid rgba(120, 120, 120, 0.2);
        margin-bottom: 0.75rem;
        background: rgba(255,255,255,0.02);
    }
    .api-ok  { color: #22c55e; font-weight: 600; }
    .api-err { color: #ef4444; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Backend URL — override with EVIRAG_API_URL env var
# ---------------------------------------------------------------------------

import os
BACKEND_URL = os.environ.get("EVIRAG_API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Mode presets (identical to before)
# ---------------------------------------------------------------------------

MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    "Fast": {
        "mode": "evirag",
        "depth_vs_speed": "fast",
        "auto_fallback_to_full": False,   # stay fast — no fallback to full pipeline
        "show_reasoning": False,
    },
    "Balanced": {
        "mode": "evirag",
        "depth_vs_speed": "balanced",
        "auto_fallback_to_full": True,    # balanced may fall back when evidence is weak
        "show_reasoning": True,
    },
    "Deep Reasoning": {
        "mode": "evirag",
        "depth_vs_speed": "deep",
        "auto_fallback_to_full": True,
        "show_reasoning": True,
    },
    "Vanilla RAG": {
        "mode": "vanilla_rag",
        "depth_vs_speed": "balanced",
        "auto_fallback_to_full": False,
        "show_reasoning": False,
    },
}


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

class EVIRAGAPIClient:
    """Thin HTTP wrapper around the FastAPI backend."""

    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ── health ──────────────────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"status": "unreachable"}

    # ── initialize / warm ───────────────────────────────────────────────────

    def initialize(self, config: Dict[str, Any], rebuild: bool = False) -> Dict[str, Any]:
        payload = dict(config)
        payload["rebuild_index"] = rebuild
        r = requests.post(
            f"{self.base_url}/api/initialize",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    # ── query ────────────────────────────────────────────────────────────────

    def query(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/api/process_query",
            json={"query": query, "config": config},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    # ── compare modes ────────────────────────────────────────────────────────

    def evaluate_modes(
        self,
        queries: Optional[List[str]] = None,
        backend: str = "local",
        enabled_agents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/api/evaluate_modes",
            json={
                "queries": queries,
                "backend": backend,
                "enabled_agents": enabled_agents,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    # ── corpus stats ─────────────────────────────────────────────────────────

    def corpus_stats(self, backend: str = "local") -> Dict[str, Any]:
        r = requests.get(
            f"{self.base_url}/api/corpus/stats",
            params={"backend": backend},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def build_api_config(selected_mode: str) -> Tuple[Dict[str, Any], bool]:
    """Return (config_dict_for_api, show_reasoning)."""
    preset = MODE_PRESETS[selected_mode]
    cfg = {
        "mode": preset["mode"],
        "backend": "local",
        "enabled_agents": ["precision", "recall", "skeptic", "counterfactual"],
        "use_visual_grounding": False,
        "depth_vs_speed": preset["depth_vs_speed"],
        "use_epistemic_divergence": True,
        "use_causal_attribution": False,
        "use_temporal_tracking": True,
        "auto_fallback_to_full": preset["auto_fallback_to_full"],
        "rebuild_index": False,
    }
    return cfg, preset["show_reasoning"]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def initialize_session_state():
    defaults = {
        "messages": [],
        "latest_result": None,
        "evaluation_result": None,
        "active_mode": "Fast",
        "initialized_configs": set(),  # set of mode keys already warmed
        "api_client": EVIRAGAPIClient(BACKEND_URL),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _config_key(cfg: Dict[str, Any]) -> str:
    return f"{cfg['mode']}|{cfg['depth_vs_speed']}|{cfg['auto_fallback_to_full']}"


def ensure_warm(cfg: Dict[str, Any], rebuild: bool = False) -> None:
    """Call /api/initialize once per config (or always on rebuild)."""
    client: EVIRAGAPIClient = st.session_state.api_client
    key = _config_key(cfg)
    if rebuild or key not in st.session_state.initialized_configs:
        with st.spinner("Preparing corpus on the backend…"):
            try:
                client.initialize(cfg, rebuild=rebuild)
                st.session_state.initialized_configs.add(key)
            except Exception as exc:
                st.error(f"Backend initialization failed: {exc}")
                st.stop()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    client: EVIRAGAPIClient = st.session_state.api_client

    # --- Connection status ---
    st.sidebar.title("EVIRAG")
    health = client.health()
    status_txt = health.get("status", "unreachable")
    if status_txt == "healthy":
        st.sidebar.markdown(
            f"<span class='api-ok'>● Backend connected</span> "
            f"<small>({BACKEND_URL})</small>",
            unsafe_allow_html=True,
        )
    elif status_txt == "initializing":
        st.sidebar.markdown(
            f"<span style='color:#f59e0b;font-weight:600'>● Backend initializing…</span>",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            f"<span class='api-err'>● Backend unreachable</span> "
            f"<small>({BACKEND_URL})</small>",
            unsafe_allow_html=True,
        )
        st.sidebar.warning(
            "Start the FastAPI backend:\n```\npython run.py --mode api\n```"
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Chat Settings")

    selected_mode = st.sidebar.radio(
        "Conversation Mode",
        list(MODE_PRESETS.keys()),
        index=list(MODE_PRESETS.keys()).index(st.session_state.active_mode),
        help="Fast uses the offline claim graph. Balanced/Deep run the full multi-agent pipeline.",
    )
    st.session_state.active_mode = selected_mode

    st.sidebar.markdown("**Mode descriptions**")
    st.sidebar.caption("- `Fast`: offline claim graph only — always <15s")
    st.sidebar.caption("- `Balanced`: full multi-agent pipeline, ~2–4 min local")
    st.sidebar.caption("- `Deep Reasoning`: same pipeline, thoroughness priority")
    st.sidebar.caption("- `Vanilla RAG`: simple retrieval + synthesis baseline")

    st.sidebar.markdown("---")
    cfg, _ = build_api_config(selected_mode)

    if st.sidebar.button("Warm corpus", width="stretch"):
        ensure_warm(cfg, rebuild=False)
        st.sidebar.success("Corpus ready")

    if st.sidebar.button("Rebuild corpus graph", width="stretch"):
        ensure_warm(cfg, rebuild=True)
        st.sidebar.success("Rebuild complete")

    with st.sidebar.expander("Developer", expanded=False):
        if st.sidebar.button("Run fast vs full comparison", width="stretch", key="run_eval"):
            with st.spinner("Running comparison (this may take a few minutes)…"):
                try:
                    st.session_state.evaluation_result = client.evaluate_modes(backend="local")
                except Exception as exc:
                    st.error(f"Comparison failed: {exc}")
        if st.session_state.evaluation_result:
            st.json(st.session_state.evaluation_result.get("aggregate", {}))

        # Corpus stats
        if st.sidebar.button("Show corpus stats", width="stretch", key="corpus_stats"):
            try:
                stats = client.corpus_stats()
                st.json(stats)
            except Exception as exc:
                st.warning(f"Could not fetch corpus stats: {exc}")

    return selected_mode


# ---------------------------------------------------------------------------
# Rendering (unchanged from before — works on the same dict schema)
# ---------------------------------------------------------------------------

def _truncate_label(text: str, limit: int = 80) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _dot_escape(text: str) -> str:
    return html.escape((text or "").replace("\\", "\\\\").replace('"', '\\"'))


def render_graph_visual(graph: Dict[str, Any], max_nodes: int = 18, max_edges: int = 28):
    nodes = graph.get("nodes", []) or []
    edges = graph.get("edges", []) or []
    if not nodes:
        st.info("No graph available for this answer.")
        return

    selected_nodes = nodes[:max_nodes]
    selected_ids = {node.get("id") for node in selected_nodes}
    selected_edges = [
        edge for edge in edges
        if edge.get("source") in selected_ids and edge.get("target") in selected_ids
    ][:max_edges]

    palette = ["#E0F2FE", "#DCFCE7", "#FEF3C7", "#FCE7F3", "#EDE9FE", "#FEE2E2"]
    doc_colors: Dict[str, str] = {}
    for node in selected_nodes:
        doc_title = node.get("doc_title") or "Unknown source"
        doc_colors.setdefault(doc_title, palette[len(doc_colors) % len(palette)])

    dot_lines = [
        "graph EVIRAG {",
        'graph [layout=neato, overlap=false, splines=true, pad="0.2"];',
        'node [shape=box, style="rounded,filled", color="#64748B", fontname="Helvetica", fontsize=10];',
        'edge [fontname="Helvetica", fontsize=9];',
    ]
    for node in selected_nodes:
        node_id = _dot_escape(str(node.get("id")))
        text_label = _truncate_label(node.get("text", ""), 72)
        source_label = _truncate_label(node.get("doc_title", "Unknown source"), 28)
        label = _dot_escape(f"{text_label}\\n[{source_label}]")
        fill = doc_colors.get(node.get("doc_title") or "Unknown source", "#F8FAFC")
        dot_lines.append(f'"{node_id}" [label="{label}", fillcolor="{fill}"];')

    edge_colors = {"supports": "#0F766E", "contradicts": "#B91C1C", "neutral": "#64748B"}
    for edge in selected_edges:
        source = _dot_escape(str(edge.get("source")))
        target = _dot_escape(str(edge.get("target")))
        relationship = str(edge.get("relationship", "neutral"))
        color = edge_colors.get(relationship, "#64748B")
        confidence = float(edge.get("confidence", 0.5) or 0.5)
        penwidth = 1.0 + (confidence * 2.5)
        dot_lines.append(
            f'"{source}" -- "{target}" [label="{relationship}", color="{color}", '
            f'fontcolor="{color}", penwidth={penwidth:.2f}];'
        )
    dot_lines.append("}")

    st.caption(
        f"Showing {len(selected_nodes)} of {len(nodes)} claims and "
        f"{len(selected_edges)} of {len(edges)} relationships."
    )
    st.caption("Green = support, red = contradiction, gray = neutral.")
    try:
        st.graphviz_chart("\n".join(dot_lines), width="stretch")
    except Exception:
        st.info("Graph rendering unavailable; showing edge list.")
        if selected_edges:
            st.dataframe(pd.DataFrame(selected_edges), width="stretch")


def render_answer(result: Dict[str, Any]):
    if result.get("mode") == "vanilla_rag":
        st.markdown(result.get("answer", ""))
        for source in result.get("sources", []):
            st.caption(f"{source.get('title')} · score {source.get('score', 0):.3f}")
        return

    answer = result.get("answer", {})
    dominant = answer.get("dominant_view") or {}
    st.markdown(dominant.get("summary", "_No answer generated._"))

    chips = []
    chips.append(
        f"Confidence {answer.get('overall_confidence', 'unknown')} "
        f"({answer.get('confidence_score', 0.0):.3f})"
    )
    pipeline = result.get("trace", {}).get("pipeline_mode")
    if pipeline:
        chips.append(pipeline)
    if result.get("statistics", {}).get("fallback_to_full"):
        chips.append("fell back to full reasoning")
    st.markdown(
        "".join(f"<span class='mode-chip'>{chip}</span>" for chip in chips),
        unsafe_allow_html=True,
    )

    citations = dominant.get("citations", [])
    if citations:
        st.markdown("**Sources behind this answer**")
        for citation in citations:
            label = citation.get("citation_label", citation.get("source_doc_title", "Unknown"))
            st.markdown(
                f"<div class='source-card'><strong>{label}</strong><br>"
                f"{citation.get('text', '')}</div>",
                unsafe_allow_html=True,
            )

    alt_views = answer.get("alternative_views", [])
    if alt_views:
        with st.expander("Other viewpoints in the corpus", expanded=False):
            for idx, view in enumerate(alt_views, start=1):
                st.markdown(f"**{idx}. {view.get('name', 'View')}**")
                st.write(view.get("summary", ""))

    graph = result.get("graph", {})
    if graph.get("nodes"):
        st.markdown("**Claim graph**")
        render_graph_visual(graph)


def render_reasoning(result: Dict[str, Any]):
    trace = result.get("trace") or {}
    if not trace:
        st.info("No reasoning trace available.")
        return

    cols = st.columns(3)
    cols[0].metric("Pipeline", trace.get("pipeline_mode", "unknown"))
    cols[1].metric("Speed setting", trace.get("requested_speed", "unknown"))
    fallback = trace.get("fallback", {}) or {}
    cols[2].metric(
        "Fallback",
        "Yes" if fallback.get("triggered") or result.get("statistics", {}).get("fallback_to_full") else "No",
    )
    if fallback.get("reason"):
        st.warning(fallback["reason"])

    with st.expander("Reasoning steps", expanded=True):
        for step in trace.get("steps", []):
            st.markdown(f"**{step.get('step', 'step')}**")
            st.json(step)

    with st.expander("Models and components", expanded=False):
        st.json(trace.get("models", {}))


def render_claims_graph_metrics(result: Dict[str, Any]):
    claims = result.get("claims", [])
    graph = result.get("graph", {})
    metrics = result.get("metrics", {})
    stats = result.get("statistics", {})

    tab_claims, tab_graph, tab_metrics = st.tabs(["Claims", "Graph", "Metrics"])

    with tab_claims:
        if claims:
            rows = [
                {
                    "claim_id": claim.get("claim_id"),
                    "text": claim.get("text"),
                    "source": claim.get("citation_label"),
                }
                for claim in claims
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch")
        else:
            st.info("No structured claims available.")

    with tab_graph:
        st.write(f"Nodes: {len(graph.get('nodes', []))}")
        st.write(f"Edges: {len(graph.get('edges', []))}")
        render_graph_visual(graph)
        if graph.get("edges"):
            st.dataframe(pd.DataFrame(graph["edges"]), width="stretch")
        else:
            st.info("No graph edges available.")

    with tab_metrics:
        st.json({
            "metrics": metrics,
            "statistics": stats,
            "epistemic_divergence": result.get("epistemic_divergence"),
            "temporal_drift": result.get("temporal_drift"),
        })


def render_message(message: Dict[str, Any], show_reasoning: bool):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
            return
        result = message["result"]
        render_answer(result)
        if show_reasoning:
            with st.expander("Reasoning", expanded=False):
                render_reasoning(result)
        with st.expander("Evidence and metrics", expanded=False):
            render_claims_graph_metrics(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    initialize_session_state()
    selected_mode = render_sidebar()
    cfg, show_reasoning = build_api_config(selected_mode)
    client: EVIRAGAPIClient = st.session_state.api_client

    st.title("EVIRAG")
    st.caption(
        "Talk to your local paper corpus. "
        "Choose a reasoning mode in the sidebar and ask questions normally."
    )

    example_cols = st.columns(3)
    example_cols[0].caption("Try: `Does homework improve academic achievement?`")
    example_cols[1].caption("Try: `What disagreements exist about RAG effectiveness?`")
    example_cols[2].caption("Try: `Summarize the main viewpoints on this topic.`")

    st.markdown("---")

    for message in st.session_state.messages:
        render_message(message, show_reasoning=show_reasoning)

    prompt = st.chat_input("Ask a question about the corpus")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            # Ensure backend is warm for this config
            ensure_warm(cfg)
            try:
                with st.spinner("Searching the corpus…"):
                    t0 = time.time()
                    result = client.query(prompt, cfg)
                    elapsed = time.time() - t0
                    result.setdefault("_query_elapsed_s", round(elapsed, 2))
                st.session_state.latest_result = result
                render_answer(result)
                if show_reasoning:
                    with st.expander("Reasoning", expanded=False):
                        render_reasoning(result)
                with st.expander("Evidence and metrics", expanded=False):
                    render_claims_graph_metrics(result)
                st.caption(f"Response time: {elapsed:.1f}s")
            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot reach the backend. "
                    "Start it with: `python run.py --mode api`"
                )
                return
            except requests.exceptions.HTTPError as exc:
                st.error(f"Backend error {exc.response.status_code}: {exc.response.text[:300]}")
                return
            except Exception as exc:
                st.error(f"Query failed: {exc}")
                st.exception(exc)
                return

        st.session_state.messages.append({"role": "assistant", "result": result})

    if st.session_state.latest_result:
        st.markdown("---")
        with st.expander("Latest result details", expanded=False):
            render_claims_graph_metrics(st.session_state.latest_result)


if __name__ == "__main__":
    main()
