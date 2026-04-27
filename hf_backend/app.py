"""
EVIRAG Main Backend — HuggingFace Spaces deployment
4-agent deliberative RAG: FAISS peS2o 100k corpus + Ollama cloud synthesis.
"""

import os
import re
import asyncio
from uuid import uuid4
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Config ─────────────────────────────────────────────────────────────────────
SEARCH_URL = os.getenv("EVIRAG_SEARCH_URL", "http://localhost:7860")

# ── Agent definitions ──────────────────────────────────────────────────────────
AGENT_CONFIGS: Dict[str, Dict] = {
    "builder": {
        "glyph": "B", "name": "Builder",
        "role": "Builds the strongest supporting case for the dominant view",
        "stance": "support", "view_stance": "dominant",
        "query_fn": lambda q: f"Evidence supporting the scientific consensus on: {q}",
        "k": 6, "confidence_threshold": 0.72,
        "search_strategy": "cosine-support", "temperature": 0.25,
        "model": "gpt-oss:120b (Ollama cloud)",
    },
    "skeptic": {
        "glyph": "S", "name": "Skeptic",
        "role": "Adversarial retrieval · contradiction and counter-evidence seeking",
        "stance": "contra", "view_stance": "alternative",
        "query_fn": lambda q: f"Evidence against or contradicting the claim that: {q}",
        "k": 6, "confidence_threshold": 0.70,
        "search_strategy": "neg-augmented", "temperature": 0.30,
        "model": "gpt-oss:120b (Ollama cloud)",
    },
    "archivist": {
        "glyph": "A", "name": "Archivist",
        "role": "Grounds claims in peS2o source metadata and citation diversity",
        "stance": "neutral", "view_stance": "dominant",
        "query_fn": lambda q: f"Key research findings and papers documenting: {q}",
        "k": 5, "confidence_threshold": 0.65,
        "search_strategy": "citation-first", "temperature": 0.20,
        "model": "gpt-oss:120b (Ollama cloud)",
    },
    "judge": {
        "glyph": "J", "name": "Judge",
        "role": "Calibrates confidence and flags unverified assumptions",
        "stance": "neutral", "view_stance": "alternative",
        "query_fn": lambda q: f"How settled or contested is the scientific evidence on: {q}",
        "k": 4, "confidence_threshold": 0.68,
        "search_strategy": "verification", "temperature": 0.15,
        "model": "gpt-oss:120b (Ollama cloud)",
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def _sanitize(obj: Any) -> Any:
    if isinstance(obj, str):
        return "".join(c for c in obj if ord(c) >= 32 or c in "\t\n\r")
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize(v) for v in obj)
    try:
        import numpy as np
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
    except ImportError:
        pass
    return obj


# ── Pydantic models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    backend: str = "local"
    agents: bool = False
    vlm: bool = False

class TracePlanRequest(BaseModel):
    query: str
    config: Dict[str, Any] = {}


# ── Session store ──────────────────────────────────────────────────────────────
_chat_sessions: Dict[str, Dict] = {}


# ── FAISS search ───────────────────────────────────────────────────────────────
def _fetch_faiss_sources(query: str, k: int = 8):
    import requests as _req
    try:
        resp = _req.post(
            f"{SEARCH_URL}/search",
            json={"query": query, "k": k},
            timeout=25,
        )
        resp.raise_for_status()
        data = resp.json()
        papers = data.get("results", [])
        sources: List[Dict] = []
        scores:  List[float] = []
        for p in papers:
            # Try all common field names the FAISS search space may use for full text
            excerpt = (
                p.get("text_excerpt") or
                p.get("text") or
                p.get("abstract") or
                p.get("excerpt") or
                p.get("body_text") or
                p.get("content") or
                p.get("paragraph_text") or
                ""
            ).strip()
            if not excerpt:
                # Last resort: scan all values for a long text string
                for v in p.values():
                    if isinstance(v, str) and len(v) > 80 and not v.startswith("http"):
                        excerpt = v.strip()
                        break
            title = (p.get("title") or p.get("paper_title") or "Unknown").strip()
            year  = p.get("year") or p.get("publication_year")
            src   = p.get("source") or p.get("venue") or p.get("journal") or ""
            # If the search space only returns metadata (no full text), synthesise
            # a descriptive snippet from the paper's bibliographic info so that
            # the LLM synthesis call still has meaningful context to work from.
            if not excerpt and title and title != "Unknown":
                parts = [f'"{title}"']
                if year:
                    parts.append(f"({year})")
                if src:
                    parts.append(f"in {src}")
                excerpt = " ".join(parts)
            sources.append({
                "n":       p.get("rank", len(sources) + 1),
                "title":   title,
                "year":    year,
                "snippet": excerpt[:300],
                "doi":     p.get("doi") or "",
                "source":  src,
            })
            scores.append(float(p.get("score", 768)))
        top_dist = min(scores) if scores else 768
        relevance = round(max(0.0, 1.0 - top_dist / 768), 3)
        return sources, relevance
    except Exception as e:
        print(f"[chat] FAISS backend unavailable: {e}")
        return [], 0.5


# ── View / claim synthesis (one sentence per agent) ───────────────────────────
def _synthesize_view_reason(query: str, snippets: List[str], stance: str) -> str:
    import re as _re
    from ollama_cloud_client import get_cloud_client
    snip_text = " | ".join(s[:180] for s in snippets[:3] if s.strip())
    if not snip_text:
        return f"No papers directly address the {stance} position on this topic."

    instr = (
        "In ONE concise sentence, explain what these papers suggest supports or confirms the main position."
        if stance == "dominant" else
        "In ONE concise sentence, explain what alternative, contrasting, or contradictory perspective these papers raise."
    )
    system_msg = "You are a scientific analyst. Be concise. Do NOT copy paper titles. Respond based ONLY on provided papers."
    user_msg   = f"Topic: {query}\nPapers: {snip_text}\n\n{instr}"

    try:
        client = get_cloud_client()
        text = client.generate(prompt=user_msg, system=system_msg, temperature=0.25, max_tokens=120)
        text = text.strip().split("**Evolving Claim")[0]
        text = _re.sub(r'[\x00-\x1f\x7f]+', ' ', text)
        text = _re.sub(r' {2,}', ' ', text).strip()
        for end in ['. ', '! ', '? ']:
            last = text.rfind(end)
            if last > len(text) // 2:
                text = text[:last + 1]
                break
        text = text.rstrip('.,; ')
        if not text:
            return f"These papers suggest a {stance} perspective on the topic."
        if text[-1] not in '.!?':
            text += '.'
        return text[:260]
    except Exception as exc:
        print(f"[agent] view reason error: {exc}")
        return f"These papers suggest a {stance} perspective on the topic."


# ── Full answer synthesis (Perplexity-style, multi-turn) ─────────────────────
def _chat_synthesize(sources_block: str, message: str,
                     chat_history: Optional[List[Dict]] = None,
                     agent_views: Optional[str] = None) -> str:
    from ollama_cloud_client import get_cloud_client
    has_ctx = bool(sources_block.strip())
    has_views = bool(agent_views and agent_views.strip())

    preamble = ""
    if has_views:
        preamble = (
            f"Four deliberative agents analysed this topic and found these perspectives:\n{agent_views}\n\n"
            "Synthesise these views into a comprehensive answer. "
            "Explicitly discuss where agents agree and disagree. "
        )

    if has_ctx:
        system_msg = (
            "You are EVIRAG, an expert scientific research assistant. "
            "You are in a multi-turn conversation — maintain full context from previous turns. "
            "Write a thorough, well-structured answer of at least 5-6 sentences. "
            "Cover key evidence, mechanisms, disagreements, and nuances. Never give a one-sentence answer. "
            "Use [N] to cite a retrieved passage ONLY when it directly supports a specific claim. "
            "If retrieved passages are not relevant, IGNORE them and answer from scientific knowledge. "
            "Do NOT copy paper titles verbatim. "
            "End with a new line: '**Evolving Claim:**' followed by a one-sentence synthesis of the key finding."
        )
    else:
        system_msg = (
            "You are EVIRAG, an expert scientific research assistant. "
            "You are in a multi-turn conversation — maintain full context from previous turns. "
            "Write a thorough, well-structured answer of at least 5-6 sentences using scientific knowledge. "
            "Cover mechanisms, consensus, evidence, and nuances. Never give a short answer. "
            "Do NOT use [N] citation markers or fabricate references. "
            "End with a new line: '**Evolving Claim:**' followed by a one-sentence synthesis of the key finding."
        )

    messages = [{"role": "system", "content": system_msg}]
    if chat_history:
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

    user_parts = []
    if preamble:
        user_parts.append(preamble)
    if has_ctx:
        user_parts.append(f"Retrieved passages (cite with [N] when directly applicable):\n{sources_block[:3000]}")
    user_parts.append(message)
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})

    try:
        client = get_cloud_client()
        data = client.chat(messages=messages, temperature=0.35, max_tokens=2048)
        content = data.get("message", {}).get("content", "")
        if isinstance(content, str):
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
    except Exception as e:
        print(f"[chat] synthesis error: {e}")
        return ""


# ── Single-agent async runner ─────────────────────────────────────────────────
async def _run_agent(agent_key: str, cfg: Dict, query: str) -> Dict:
    """FAISS search + one-sentence synthesis for one agent. Runs concurrently."""
    import time
    t0 = time.perf_counter()

    agent_query = cfg["query_fn"](query)
    loop = asyncio.get_event_loop()

    sources, relevance = await loop.run_in_executor(
        None, _fetch_faiss_sources, agent_query, cfg["k"]
    )

    snips = [s["snippet"] for s in sources if s.get("snippet")][:3]
    if snips:
        synthesis = await loop.run_in_executor(
            None, _synthesize_view_reason, query, snips, cfg["view_stance"]
        )
    else:
        synthesis = ""

    duration_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Extract sentences from synthesis for claim nodes
    sentences = [s.strip() for s in re.split(r'[.!?]+', synthesis) if len(s.strip()) > 20]

    return {
        "agent_name":          agent_key,
        "name":                cfg["name"],
        "glyph":               cfg["glyph"],
        "role":                cfg["role"],
        "stance":              cfg["stance"],
        "view_stance":         cfg["view_stance"],
        "sources":             sources,
        "synthesis":           synthesis,
        "sentences":           sentences[:2],
        "relevance":           relevance,
        "num_chunks":          len(sources),
        "num_claims":          max(1, len(sentences)),
        "duration_ms":         duration_ms,
        "status":              "ok",
        "confidence":          round(min(0.92, 0.50 + relevance * 0.42), 3),
        "confidence_threshold": cfg["confidence_threshold"],
        "search_strategy":     cfg["search_strategy"],
        "retrieval_k":         cfg["k"],
        "retrieval_query":     agent_query,
        "model":               cfg["model"],
        "temperature":         cfg["temperature"],
    }


# ── Claim graph builder ────────────────────────────────────────────────────────
def _build_claim_graph(agent_results: List[Dict], turn: int):
    """
    Build graph nodes (one per extracted sentence) and edges based on
    agent stance: builder↔archivist=support, builder↔skeptic=contradicts,
    judge→others=neutral.
    Returns (nodes, edges, claims_list).
    """
    nodes:  List[Dict] = []
    edges:  List[Dict] = []
    claims: List[Dict] = []

    node_seq = 1
    agent_node_ids: Dict[str, List[str]] = {}

    for ar in agent_results:
        key = ar["agent_name"]
        agent_node_ids[key] = []
        sents = ar.get("sentences", [])
        if not sents and ar.get("synthesis"):
            sents = [ar["synthesis"]]

        for sent in sents[:2]:
            sent = sent.strip()
            if len(sent) < 15:
                continue
            if not sent.endswith(('.', '!', '?')):
                sent += '.'
            cid = f"c{turn}_{node_seq}"
            node_seq += 1

            nodes.append({"id": cid, "text": sent, "doc_title": ar["name"] + " agent"})
            agent_node_ids[key].append(cid)

            claims.append({
                "id":               cid,
                "claim_id":         cid,
                "text":             sent,
                "type":             ar["stance"],
                "confidence":       ar["confidence"],
                "source_doc_title": ar["name"] + " agent · peS2o",
                "source_doc_id":    key,
                "source_path":      "",
                "section":          ar["stance"],
            })

    builder_ids   = agent_node_ids.get("builder",   [])
    skeptic_ids   = agent_node_ids.get("skeptic",   [])
    archivist_ids = agent_node_ids.get("archivist", [])
    judge_ids     = agent_node_ids.get("judge",     [])

    # Builder ↔ Archivist: support (both pro-dominant)
    for a in builder_ids[:1]:
        for b in archivist_ids[:1]:
            edges.append({"source": a, "target": b, "relationship": "supports",    "confidence": 0.72})

    # Builder ↔ Skeptic: contradicts
    for a in builder_ids[:1]:
        for b in skeptic_ids[:1]:
            edges.append({"source": a, "target": b, "relationship": "contradicts", "confidence": 0.68})

    # Skeptic ↔ Archivist: neutral (different framings)
    for a in skeptic_ids[:1]:
        for b in archivist_ids[:1]:
            edges.append({"source": a, "target": b, "relationship": "neutral",     "confidence": 0.50})

    # Judge → Builder + Skeptic: neutral assessment edges
    for j in judge_ids[:1]:
        for b in (builder_ids + skeptic_ids)[:2]:
            edges.append({"source": j, "target": b, "relationship": "neutral",     "confidence": 0.55})

    # Multi-sentence same-agent support edges
    for ids in [builder_ids, skeptic_ids, archivist_ids]:
        if len(ids) >= 2:
            edges.append({"source": ids[0], "target": ids[1], "relationship": "supports", "confidence": 0.80})

    return nodes, edges, claims


# ── Fast-path answer builder ──────────────────────────────────────────────────
def _build_fast_answer(message: str, faiss_sources: List[Dict], faiss_relevance: float,
                       session: Dict):
    sources = faiss_sources or []

    # Relevance gate
    query_terms = {w.strip("?!.,;:") for w in message.lower().split() if len(w) >= 3}
    all_text = " ".join((s.get("snippet") or "").lower() for s in sources[:6])
    keyword_hit = not query_terms or any(t in all_text for t in query_terms)
    has_relevant = faiss_relevance >= 0.70 and keyword_hit and any(s.get("snippet") for s in sources[:8])

    src_block = ("\n".join(
        f"[{s['n']}] {s['snippet'][:250]}"
        for s in sources[:6] if s.get("snippet")
    ) if has_relevant else "")

    history = session.get("history", [])
    chat_hist = []
    for h in history[-4:]:
        chat_hist.append({"role": "user",      "content": h["user"]})
        chat_hist.append({"role": "assistant", "content": h["answer"]})

    answer_md = _chat_synthesize(src_block, message, chat_hist)
    if not answer_md:
        answer_md = "I was unable to synthesize an answer at this time. Please try again."

    claim_match = re.search(r'\*{0,2}Evolving Claim:\*{0,2}\s*(.+?)(?:\n|$)', answer_md, re.DOTALL)
    claim = claim_match.group(1).strip() if claim_match else (
        [s.strip() for s in re.split(r'[.!?]+', answer_md) if s.strip()] or [message]
    )[-1]

    answer_body = re.sub(
        r'(?i)(?:The best(?:-| )supported position is:)?\s*\*{0,2}Evolving Claim:\*{0,2}.*$',
        '', answer_md, flags=re.DOTALL
    ).strip()

    return answer_body, claim, sources


# ── App ────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[evirag-backend] v3 · 4-agent deliberative · Search URL: {SEARCH_URL}")
    yield

app = FastAPI(
    title="EVIRAG Backend",
    description="4-agent deliberative RAG — HF Spaces deployment",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "3.0.0", "search_url": SEARCH_URL}


# ── Trace plan (called immediately on query submit, shows progress steps) ─────
@app.post("/api/query_trace_plan")
async def query_trace_plan(request: TracePlanRequest):
    cfg = request.config or {}
    use_agents = bool(cfg.get("enabled_agents")) and len(cfg.get("enabled_agents", [])) > 0

    if use_agents:
        steps = [
            {"step": "intent",      "label": "Classifying epistemic intent",         "detail": "Detecting whether the query is contested, factual, or theoretical."},
            {"step": "hypothesis",  "label": "X-IR hypothesis generation",            "detail": "Generating central hypothesis and expected counterclaims before retrieval."},
            {"step": "agents",      "label": "Dispatching 4 parallel agents",         "detail": "Builder · Skeptic · Archivist · Judge running simultaneously on the peS2o corpus."},
            {"step": "builder",     "label": "Builder: retrieving supporting evidence","detail": "Searching for papers that confirm the dominant scientific view."},
            {"step": "skeptic",     "label": "Skeptic: adversarial evidence search",  "detail": "Adversarially searching for contradictions, null results, and counter-evidence."},
            {"step": "archivist",   "label": "Archivist: citation grounding",         "detail": "Grounding claims in source metadata and maximising citation diversity."},
            {"step": "judge",       "label": "Judge: confidence calibration",         "detail": "Assessing methodological quality and calibrating uncertainty."},
            {"step": "synthesis",   "label": "Cross-agent synthesis",                 "detail": "Merging all agent views into a unified disagreement-aware answer."},
            {"step": "graph",       "label": "Building claim disagreement graph",     "detail": "Constructing nodes per claim and stance-classified edges between them."},
        ]
        mode_label  = "EVIRAG · 4-agent deliberative"
        path_label  = "parallel agent retrieval"
    else:
        steps = [
            {"step": "intent",    "label": "Classifying epistemic intent",   "detail": "Analysing query type and routing through fast retrieval topology."},
            {"step": "faiss",     "label": "FAISS semantic search",          "detail": "Retrieving top-K papers from the peS2o 100k open-access corpus."},
            {"step": "synthesis", "label": "Multi-view synthesis",           "detail": "Synthesising dominant and alternative views with Ollama cloud model."},
            {"step": "calibrate", "label": "Confidence calibration",         "detail": "Weighting evidence quality, source diversity, and claim agreement."},
        ]
        mode_label  = "EVIRAG · fast"
        path_label  = "fast claim retrieval"

    return {"mode_label": mode_label, "path_label": path_label, "steps": steps}


# ── Main chat endpoint ─────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(request: ChatRequest):
    import time
    t0 = time.perf_counter()

    # Session management
    session_id = request.session_id or str(uuid4())
    if session_id not in _chat_sessions:
        _chat_sessions[session_id] = {
            "history":             [],
            "accumulated_context": "",
            "claim":               "",
            "turn":                0,
            "acc_claims":          [],
            "acc_sources":         {},
        }
    session = _chat_sessions[session_id]
    turn    = session["turn"] + 1

    # Context enrichment for multi-turn
    acc = session["accumulated_context"]
    enriched_query = f"{acc}. Now specifically: {request.message}" if acc else request.message
    new_acc = f"{acc}; {request.message}" if acc else request.message
    session["accumulated_context"] = " ".join(new_acc.split()[-120:])

    loop = asyncio.get_event_loop()

    # ── Branch: 4-agent deliberative vs fast path ──────────────────────────────
    if request.agents:
        # ── Run all 4 agents in parallel ──────────────────────────────────────
        agent_tasks = [
            _run_agent(key, cfg, request.message)
            for key, cfg in AGENT_CONFIGS.items()
        ]
        agent_results: List[Dict] = list(await asyncio.gather(*agent_tasks))

        # Build claim graph from agent outputs
        graph_nodes, graph_edges, turn_claims = _build_claim_graph(agent_results, turn)

        # Collect all FAISS sources across agents (deduplicated by title)
        all_sources_map: Dict[str, Dict] = {}
        for ar in agent_results:
            for s in ar["sources"]:
                key = (s.get("title") or "")[:60]
                if key and key not in all_sources_map:
                    all_sources_map[key] = s
        all_faiss_sources = list(all_sources_map.values())[:12]

        # Best relevance across agents
        best_relevance = max((ar["relevance"] for ar in agent_results), default=0.5)

        # Build view synthesis block for the main answer
        agent_views_text = ""
        for ar in agent_results:
            if ar["synthesis"]:
                agent_views_text += f"- {ar['name']} ({ar['stance']}): {ar['synthesis']}\n"

        # Main answer synthesis incorporating agent views
        src_block = "\n".join(
            f"[{s['n']}] {s['snippet'][:200]}"
            for s in all_faiss_sources[:6] if s.get("snippet")
        ) if best_relevance >= 0.65 else ""

        chat_hist = []
        for h in session["history"][-4:]:
            chat_hist.append({"role": "user",      "content": h["user"]})
            chat_hist.append({"role": "assistant", "content": h["answer"]})

        answer_md = await loop.run_in_executor(
            None, _chat_synthesize, src_block, request.message, chat_hist, agent_views_text
        )
        if not answer_md:
            answer_md = "I was unable to synthesize an answer at this time. Please try again."

        # Extract evolving claim
        claim_match = re.search(r'\*{0,2}Evolving Claim:\*{0,2}\s*(.+?)(?:\n|$)', answer_md, re.DOTALL)
        if claim_match:
            claim = claim_match.group(1).strip()
        else:
            sents = [s.strip() for s in re.split(r'[.!?]+', answer_md) if s.strip()]
            claim = sents[-1] if sents else request.message

        answer_body = re.sub(
            r'(?i)(?:The best(?:-| )supported position is:)?\s*\*{0,2}Evolving Claim:\*{0,2}.*$',
            '', answer_md, flags=re.DOTALL
        ).strip()

        # Build EVIRAG answer_obj from agent outputs
        builder_ar  = next((ar for ar in agent_results if ar["agent_name"] == "builder"),  None)
        skeptic_ar  = next((ar for ar in agent_results if ar["agent_name"] == "skeptic"),  None)
        archive_ar  = next((ar for ar in agent_results if ar["agent_name"] == "archivist"), None)

        builder_cids  = [c["id"] for c in turn_claims if c["source_doc_id"] == "builder"]
        skeptic_cids  = [c["id"] for c in turn_claims if c["source_doc_id"] == "skeptic"]

        answer_obj: Dict[str, Any] = {}
        if builder_ar and (builder_ar["synthesis"] or builder_ar["sources"]):
            answer_obj["dominant_view"] = {
                "summary":              builder_ar["synthesis"] or "The Builder agent found supporting evidence from the peS2o corpus.",
                "source_count":         builder_ar["num_chunks"],
                "confidence":           builder_ar["confidence"],
                "overall_confidence":   "medium",
                "supporting_claim_ids": builder_cids,
                "citations": [
                    {"citation_label": s["title"][:60], "text": s.get("snippet", "")[:200]}
                    for s in builder_ar["sources"][:3]
                ],
            }
        if skeptic_ar and (skeptic_ar["synthesis"] or skeptic_ar["sources"]):
            answer_obj["alternative_views"] = [{
                "summary":              skeptic_ar["synthesis"] or "The Skeptic agent found contradictory evidence.",
                "source_count":         skeptic_ar["num_chunks"],
                "confidence":           skeptic_ar["confidence"],
                "supporting_claim_ids": skeptic_cids,
                "citations": [
                    {"citation_label": s["title"][:60], "text": s.get("snippet", "")[:200]}
                    for s in skeptic_ar["sources"][:3]
                ],
            }]
        answer_obj["minority_views"] = []
        answer_obj["confidence_score"] = round(
            sum(ar["confidence"] for ar in agent_results) / len(agent_results), 3
        ) if agent_results else 0.65
        answer_obj["overall_confidence"] = "high" if answer_obj["confidence_score"] > 0.75 else "medium"
        answer_obj["direct_answer"] = answer_body[:400]

        # Metrics
        contra_e = sum(1 for e in graph_edges if e["relationship"] == "contradicts")
        total_e  = max(1, len(graph_edges))
        conflict_ratio = round(contra_e / total_e, 3)
        metrics = {
            "conflict_ratio":       conflict_ratio,
            "claim_entropy":        round(0.25 + conflict_ratio * 0.6, 3),
            "disagreement_density": round(contra_e / max(1, len(graph_nodes)), 3),
            "visual_text_mismatch": 0.0,
        }

        # Epistemic divergence
        ed_score = conflict_ratio
        controversy_class = "open" if ed_score > 0.38 else "narrowing" if ed_score > 0.18 else "stable"
        epistemic = {
            "ed_score":                ed_score,
            "polarization_index":      round(ed_score * 0.8, 3),
            "consensus_collapse_score":round(ed_score * 0.6, 3),
            "reasoning": (
                f"{len(agent_results)} agents analysed this topic. "
                f"The corpus shows {'significant' if ed_score > 0.38 else 'moderate' if ed_score > 0.18 else 'limited'} "
                f"disagreement ({contra_e} contradicting edge{'s' if contra_e != 1 else ''} out of {total_e} total)."
            ),
            "controversy_class": controversy_class,
        }

        # Intent
        intent = {
            "factual_vs_disputed":       "highly_disputed" if conflict_ratio > 0.38 else "somewhat_disputed" if conflict_ratio > 0.18 else "factual",
            "empirical_vs_theoretical":  "empirical",
            "disagreement_level":        "high" if conflict_ratio > 0.38 else "medium" if conflict_ratio > 0.18 else "low",
        }

        # Hypothesis (X-IR)
        hypothesis = {
            "central_hypothesis":     request.message,
            "supporting_claims":      [ar["synthesis"][:80] for ar in agent_results if ar["stance"] == "support" and ar.get("synthesis")][:2],
            "assumptions":            ["The peS2o corpus contains sufficient papers on this topic."],
            "expected_counterclaims": [ar["synthesis"][:80] for ar in agent_results if ar["stance"] == "contra" and ar.get("synthesis")][:2],
        }

        # Agent details for the Analysis drawer
        agent_details = [
            {
                "agent_name":           ar["agent_name"],
                "key":                  ar["agent_name"],
                "name":                 ar["name"],
                "glyph":                ar["glyph"],
                "role":                 ar["role"],
                "stance":               ar["stance"],
                "model":                ar["model"],
                "num_chunks":           ar["num_chunks"],
                "num_claims":           ar["num_claims"],
                "confidence":           ar["confidence"],
                "confidence_threshold": ar["confidence_threshold"],
                "search_strategy":      ar["search_strategy"],
                "retrieval_k":          ar["retrieval_k"],
                "temperature":          ar["temperature"],
                "duration_ms":          ar["duration_ms"],
                "status":               ar["status"],
                "reasoning":            ar["synthesis"] or "No synthesis returned.",
                "retrieval_query":      ar["retrieval_query"],
                # Top 3 paper titles this agent retrieved — shown in UI agent cards
                "top_sources": [
                    {"title": s.get("title", ""), "year": s.get("year"), "snippet": s.get("snippet", "")[:120]}
                    for s in ar.get("sources", [])[:3]
                ],
            }
            for ar in agent_results
        ]

        # Accumulate claims
        seen_ids = {c.get("id") for c in session["acc_claims"]}
        for c in turn_claims:
            if c["id"] not in seen_ids:
                session["acc_claims"].append(c)
        session["acc_claims"] = session["acc_claims"][-80:]

        # Accumulate sources
        for s in all_faiss_sources:
            k2 = (s.get("title") or "")[:50]
            if k2 and k2 not in session["acc_sources"]:
                session["acc_sources"][k2] = s
        if len(session["acc_sources"]) > 50:
            keys = list(session["acc_sources"].keys())
            session["acc_sources"] = {k: session["acc_sources"][k] for k in keys[-50:]}

    else:
        # ── Fast path: single FAISS search + synthesis ────────────────────────
        _faiss_query = enriched_query if len(request.message.split()) < 5 else request.message
        faiss_sources, faiss_relevance = await loop.run_in_executor(
            None, _fetch_faiss_sources, _faiss_query
        )

        answer_body, claim, all_faiss_sources = await loop.run_in_executor(
            None, _build_fast_answer, request.message, faiss_sources, faiss_relevance, session
        )

        # Minimal view structure for fast path
        answer_obj = {"minority_views": [], "confidence_score": 0.65, "overall_confidence": "medium"}
        if faiss_relevance >= 0.75 and faiss_sources:
            snips_dom = [s["snippet"] for s in faiss_sources[:3] if s.get("snippet")]
            snips_alt = [s["snippet"] for s in faiss_sources[3:6] if s.get("snippet")]
            if snips_dom:
                dom_view = await loop.run_in_executor(None, _synthesize_view_reason, request.message, snips_dom, "dominant")
                answer_obj["dominant_view"] = {
                    "summary": dom_view, "source_count": len(snips_dom),
                    "confidence": 0.68, "supporting_claim_ids": [],
                    "citations": [{"citation_label": s["title"][:60], "text": s.get("snippet","")[:200]} for s in faiss_sources[:3]],
                }
            if snips_alt:
                alt_view = await loop.run_in_executor(None, _synthesize_view_reason, request.message, snips_alt, "alternative")
                answer_obj["alternative_views"] = [{
                    "summary": alt_view, "source_count": len(snips_alt),
                    "confidence": 0.60, "supporting_claim_ids": [],
                    "citations": [{"citation_label": s["title"][:60], "text": s.get("snippet","")[:200]} for s in faiss_sources[3:6]],
                }]

        graph_nodes, graph_edges, turn_claims = [], [], []
        agent_results, agent_details = [], []
        metrics = {"conflict_ratio": 0.0, "claim_entropy": 0.1, "disagreement_density": 0.0, "visual_text_mismatch": 0.0}
        epistemic = {"ed_score": 0.0, "polarization_index": 0.0, "consensus_collapse_score": 0.0, "reasoning": "", "controversy_class": "stable"}
        intent = {"factual_vs_disputed": "somewhat_disputed", "empirical_vs_theoretical": "empirical", "disagreement_level": "low"}
        hypothesis = {"central_hypothesis": request.message, "supporting_claims": [], "assumptions": [], "expected_counterclaims": []}

        # Single evolving claim
        turn_claims = [{
            "id": f"claim_{turn}", "claim_id": f"claim_{turn}",
            "text": claim, "type": "synthesis", "confidence": 0.75,
            "source_doc_title": "EVIRAG synthesis", "source_doc_id": "", "source_path": "", "section": "neutral",
        }]

        seen_ids = {c.get("id") for c in session["acc_claims"]}
        for c in turn_claims:
            if c["id"] not in seen_ids:
                session["acc_claims"].append(c)
        session["acc_claims"] = session["acc_claims"][-60:]

        for s in all_faiss_sources:
            k2 = (s.get("title") or "")[:50]
            if k2 and k2 not in session["acc_sources"]:
                session["acc_sources"][k2] = s
        if len(session["acc_sources"]) > 40:
            keys = list(session["acc_sources"].keys())
            session["acc_sources"] = {k: session["acc_sources"][k] for k in keys[-40:]}

    # ── Common tail: history, response assembly ────────────────────────────────
    session["history"].append({
        "turn":    turn,
        "user":    request.message,
        "answer":  answer_body,
        "claim":   claim,
        "sources": [s["title"] for s in all_faiss_sources[:5]],
    })
    session["claim"] = claim
    session["turn"]  = turn
    if len(session["history"]) > 20:
        session["history"] = session["history"][-20:]

    latency_ms       = round((time.perf_counter() - t0) * 1000, 1)
    acc_sources_list = list(session["acc_sources"].values())

    result = {
        "mode":        "evirag",
        "claims":      session["acc_claims"],
        "sources": [
            {
                "doc_id":         s.get("doi") or s["title"][:12],
                "title":          s["title"],
                "year":           s.get("year"),
                "doi":            s.get("doi") or "",
                "source":         s.get("source") or "",
                "snippet":        s.get("snippet") or "",
                "text":           s.get("snippet") or "",
                "cited_by_count": None,
            }
            for s in all_faiss_sources
        ],
        "answer":   answer_obj,
        "metrics":  metrics,
        "statistics": {
            "total_claims":  len(session["acc_claims"]),
            "total_sources": len(acc_sources_list),
            "agents_used":   len(agent_results),
        },
        "agent_details":       agent_details,
        "graph":               {"nodes": graph_nodes, "edges": graph_edges},
        "epistemic_divergence": epistemic,
        "intent":              intent,
        "hypothesis":          hypothesis,
        "trace": {
            "models": {
                "query_time_models": {
                    "nli_verifier": "gpt-oss:120b (Ollama cloud)",
                    "agents": {ar["agent_name"]: ar["model"] for ar in agent_results} if agent_results else {},
                }
            }
        },
        "chat": {
            "session_id":    session_id,
            "answer":        answer_body,
            "claim":         claim,
            "sources":       acc_sources_list,
            "turn":          turn,
            "latency_ms":    latency_ms,
            "total_claims":  len(session["acc_claims"]),
            "total_sources": len(acc_sources_list),
            "history": [
                msg
                for h in session["history"]
                for msg in (
                    {"role": "user",      "content": h["user"],   "turn": h["turn"]},
                    {"role": "assistant", "content": h["answer"], "turn": h["turn"],
                     "claim": h["claim"],  "sources": h["sources"]},
                )
            ],
        },
    }
    return _sanitize(result)


# ── UI bootstrap ───────────────────────────────────────────────────────────────
@app.get("/api/ui/bootstrap")
async def ui_bootstrap():
    import requests as _req
    total_docs = 0
    try:
        resp = _req.get(f"{SEARCH_URL}/status", timeout=5)
        if resp.ok:
            d = resp.json()
            total_docs = (d.get("total_papers") or d.get("num_papers")
                          or d.get("indexed_papers") or d.get("n_vectors") or 0)
    except Exception:
        pass

    chunks_est = total_docs * 8 if total_docs else 0
    return {
        "stats": {
            "total_documents": total_docs, "total_claims": 0, "total_figures": 0,
            "source": "hf_backend", "index_type": "FAISS · peS2o open-access corpus",
        },
        "models": {"retrieval": "FAISS · sentence-transformers", "verifier": "EVIRAG claim graph"},
        "topics": {"tabs": [
            {"id": "disagreement", "label": "Disagreement", "items": [
                {"q": "Does amyloid beta cause Alzheimer's disease?",                                  "meta": "neuroscience · contested mechanism"},
                {"q": "Are saturated fats causally linked to cardiovascular mortality?",              "meta": "medicine · nutrition evidence"},
                {"q": "Do carbon offsets reduce net emissions at scale?",                             "meta": "climate · policy evidence"},
            ]},
            {"id": "claims", "label": "Claims", "items": [
                {"q": "Verify the claim that CRISPR off-target effects are clinically negligible",    "meta": "biology · safety claim"},
                {"q": "Check whether transformer scaling laws still hold beyond frontier model sizes","meta": "AI · empirical claim"},
                {"q": "Map evidence for microplastics affecting human endocrine function",            "meta": "medicine · exposure claim"},
            ]},
        ]},
        "documents": [{"id": "pes2o-corpus",
            "title": f"peS2o open-access corpus — {total_docs:,} papers indexed via FAISS" if total_docs else "peS2o FAISS corpus",
            "year": 2024, "pages": 0, "claims": 0,
            "chunks": chunks_est, "figures": 0,
            "sections": [{"name": "corpus", "chunks": chunks_est}], "top_claims": []}],
        "graph": {"nodes": 0, "edges": 0, "clusters": [
            {"doc_id": "alzheimers",  "title": "Alzheimer's amyloid debate",  "claims": 0, "edges": 0, "density": 0, "class": "query-ready", "query": "Does amyloid beta cause Alzheimer's disease?"},
            {"doc_id": "offsets",     "title": "Carbon offset effectiveness", "claims": 0, "edges": 0, "density": 0, "class": "query-ready", "query": "Do carbon offsets reduce net emissions at scale?"},
            {"doc_id": "darkmatter",  "title": "Dark matter alternatives",    "claims": 0, "edges": 0, "density": 0, "class": "query-ready", "query": "Compare evidence for dark matter versus modified gravity"},
        ]},
        "agents": [
            {"key": "builder",   "name": "Builder",   "glyph": "B", "role": "Builds the strongest supporting case",              "model": "gpt-oss:120b (Ollama cloud)", "search_strategy": "cosine-support",   "retrieval_k": 6},
            {"key": "skeptic",   "name": "Skeptic",   "glyph": "S", "role": "Adversarial · contradiction-seeking",              "model": "gpt-oss:120b (Ollama cloud)", "search_strategy": "neg-augmented",    "retrieval_k": 6},
            {"key": "archivist", "name": "Archivist", "glyph": "A", "role": "Grounds claims in peS2o source metadata",          "model": "gpt-oss:120b (Ollama cloud)", "search_strategy": "citation-first",   "retrieval_k": 5},
            {"key": "judge",     "name": "Judge",     "glyph": "J", "role": "Calibrates confidence and flags uncertainty",      "model": "gpt-oss:120b (Ollama cloud)", "search_strategy": "verification",     "retrieval_k": 4},
        ],
        "calibration": [
            {"key": "claim_agreement",        "label": "Claim agreement",        "weight": 0.32},
            {"key": "source_diversity",       "label": "Source diversity",       "weight": 0.22},
            {"key": "contradiction_severity", "label": "Contradiction severity", "weight": 0.28},
            {"key": "unverified_assumptions", "label": "Unverified assumptions", "weight": 0.10},
            {"key": "visual_alignment",       "label": "Visual alignment",       "weight": 0.08},
        ],
    }


# ── Session retrieval ──────────────────────────────────────────────────────────
@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    session = _chat_sessions.get(session_id)
    if not session:
        return {"history": [], "claim": "", "turn": 0}
    return _sanitize({
        "history": session.get("history", []),
        "claim":   session.get("claim", ""),
        "turn":    session.get("turn", 0),
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
