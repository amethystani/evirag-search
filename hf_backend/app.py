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
        "role": "Builds the strongest evidence-based case for the dominant scientific view",
        "stance": "support", "view_stance": "dominant",
        "query_fn": lambda q: (
            f"systematic review meta-analysis randomized controlled trial "
            f"supporting consensus evidence mechanisms: {q}"
        ),
        "k": 6, "confidence_threshold": 0.72,
        "search_strategy": "cosine-support", "temperature": 0.25,
        "model": "gpt-oss:120b (Ollama cloud)",
    },
    "skeptic": {
        "glyph": "S", "name": "Skeptic",
        "role": "Seeks contradictions, replication failures, and alternative interpretations",
        "stance": "contra", "view_stance": "alternative",
        "query_fn": lambda q: (
            f"contradictory evidence replication failure methodological critique "
            f"alternative explanation null result against: {q}"
        ),
        "k": 6, "confidence_threshold": 0.70,
        "search_strategy": "neg-augmented", "temperature": 0.30,
        "model": "gpt-oss:120b (Ollama cloud)",
    },
    "archivist": {
        "glyph": "A", "name": "Archivist",
        "role": "Establishes the breadth of literature, key authors, methodologies, and historical development",
        "stance": "neutral", "view_stance": "dominant",
        "query_fn": lambda q: (
            f"landmark study historical development key researchers methodology "
            f"review literature seminal paper: {q}"
        ),
        "k": 5, "confidence_threshold": 0.65,
        "search_strategy": "citation-first", "temperature": 0.20,
        "model": "gpt-oss:120b (Ollama cloud)",
    },
    "judge": {
        "glyph": "J", "name": "Judge",
        "role": "Identifies where evidence is genuinely contested, calibrates confidence, and exposes assumptions",
        "stance": "neutral", "view_stance": "alternative",
        "query_fn": lambda q: (
            f"scientific controversy open question contested evidence "
            f"unresolved debate conflicting studies uncertainty: {q}"
        ),
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

# ── Full-text paper cache (paper_id → full text string) ───────────────────────
_paper_text_cache: Dict[str, str] = {}

# ── S2 API helpers ─────────────────────────────────────────────────────────────
_S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
_S2_FIELDS    = "externalIds,openAccessPdf,abstract"

async def _s2_batch_meta(corpus_ids: List[str]) -> Dict[str, Dict]:
    """One S2 API call → ArXiv IDs + open-access PDF URLs for all papers."""
    if not corpus_ids:
        return {}
    try:
        import httpx
        ids = [f"CorpusID:{cid}" for cid in corpus_ids]
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.post(
                f"{_S2_BATCH_URL}?fields={_S2_FIELDS}",
                json={"ids": ids},
                headers={"Content-Type": "application/json"},
            )
        data = resp.json() if isinstance(resp.json(), list) else []
        return {cid: (entry or {}) for cid, entry in zip(corpus_ids, data)}
    except Exception as e:
        print(f"[fulltext] S2 batch error: {e}")
        return {}


async def _fetch_arxiv_text(arxiv_id: str) -> str:
    """Fetch the full paper text from ar5iv (arXiv → clean HTML renderer).
    Returns up to 15 000 chars covering abstract, intro, methods, results,
    discussion, and conclusion — the whole paper in reading order.
    """
    from html.parser import HTMLParser as _HP
    import httpx, re as _re

    class _Strip(_HP):
        """Minimal HTML → text extractor using only stdlib."""
        _SKIP = {"script", "style", "nav", "header", "footer", "figure",
                 "table", "cite", "references", "bibliography"}
        def __init__(self):
            super().__init__()
            self.parts: List[str] = []
            self._depth = 0
        def handle_starttag(self, tag, attrs):
            if tag.lower() in self._SKIP:
                self._depth += 1
        def handle_endtag(self, tag):
            if tag.lower() in self._SKIP and self._depth:
                self._depth -= 1
        def handle_data(self, data):
            if not self._depth:
                chunk = data.strip()
                if chunk:
                    self.parts.append(chunk)

    try:
        url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
        async with httpx.AsyncClient(timeout=22, follow_redirects=True) as cl:
            resp = await cl.get(url, headers={"User-Agent": "EVIRAG/1.0 (research)"})
        if resp.status_code != 200:
            return ""
        parser = _Strip()
        parser.feed(resp.text)
        text = " ".join(parser.parts)
        text = _re.sub(r"\s{3,}", " ", text).strip()
        print(f"[fulltext] arXiv {arxiv_id} → {len(text):,} chars")
        return text[:15000]
    except Exception as e:
        print(f"[fulltext] arXiv fetch error {arxiv_id}: {e}")
        return ""


async def _fetch_pdf_text(url: str) -> str:
    """Download an open-access PDF and extract its text (first 20 pages)."""
    try:
        import httpx, io
        async with httpx.AsyncClient(timeout=28, follow_redirects=True) as cl:
            resp = await cl.get(url, headers={"User-Agent": "EVIRAG/1.0 (research)"})
        if resp.status_code != 200 or len(resp.content) < 2000:
            return ""
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(resp.content))
            pages  = [reader.pages[i].extract_text() or "" for i in range(min(20, len(reader.pages)))]
            text   = "\n".join(pages)
        except Exception:
            return ""
        import re as _re
        text = _re.sub(r"\s{3,}", " ", text).strip()
        print(f"[fulltext] PDF {url[:60]} → {len(text):,} chars")
        return text[:15000]
    except Exception as e:
        print(f"[fulltext] PDF error {url[:60]}: {e}")
        return ""


async def _enrich_paper_text(paper_id: str, s2_entry: Dict, fallback: str) -> str:
    """Best available text for a single paper, with in-memory caching.

    Priority:
      1. arXiv HTML  — full paper (up to 15 000 chars)
      2. Open-access PDF — full paper (up to 15 000 chars)
      3. S2 abstract  — often longer / cleaner than our stored excerpt
      4. FAISS text_excerpt — the 2 000-char fallback we always have
    """
    if paper_id in _paper_text_cache:
        cached = _paper_text_cache[paper_id]
        return cached

    text = ""
    ext   = s2_entry.get("externalIds") or {}
    arxiv = ext.get("ArXiv") or ext.get("arxiv")
    pdf_url = (s2_entry.get("openAccessPdf") or {}).get("url")

    if arxiv:
        text = await _fetch_arxiv_text(arxiv)

    if not text and pdf_url:
        text = await _fetch_pdf_text(pdf_url)

    if not text:
        text = (s2_entry.get("abstract") or "").strip()

    if not text:
        text = fallback

    _paper_text_cache[paper_id] = text
    return text


# ── FAISS search + full-text enrichment ───────────────────────────────────────
# Whether to ALSO try arXiv/PDF runtime enrichment when full_text is short.
# Toggled off in the peS2o-only path because the parquet already gives us
# the entire paper body within 4000 chars (test corpus) or 32 000 chars (prod).
_RUNTIME_ENRICH = os.getenv("EVIRAG_RUNTIME_ENRICH", "0") == "1"

async def _fetch_faiss_sources_enriched(query: str, k: int = 8):
    """
    Fast peS2o-first path:
      1. POST /search to the FAISS space (~5 ms locally / ~50 ms HF)
      2. Read full_text directly from the response (joined from corpus.parquet)
      3. Only fall back to slow arXiv/PDF runtime fetch if EVIRAG_RUNTIME_ENRICH=1

    This makes a typical agent retrieval go from ~2-5 s (with arXiv/PDF round trips)
    down to ~50-200 ms — the AI gets the FULL peS2o body text directly from
    our own indexed parquet, no third-party hops, no rate limits.
    """
    import httpx
    try:
        async with httpx.AsyncClient(timeout=15) as cl:
            resp = await cl.post(f"{SEARCH_URL}/search", json={"query": query, "k": k})
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        print(f"[chat] FAISS backend unavailable: {e}")
        return [], 0.5

    papers = data.get("results", [])
    if not papers:
        return [], 0.5

    # ── Path A: peS2o full_text already available — use it directly ──────────
    # Each paper carries up to 4000 chars of body text from corpus.parquet.
    # No external API calls needed.
    sources: List[Dict] = []
    scores:  List[float] = []
    needs_runtime: List[int] = []   # indices that still want arXiv/PDF backup

    for idx, p in enumerate(papers):
        ft   = (p.get("full_text") or "").strip()
        ex   = (p.get("text_excerpt") or "").strip()
        # Prefer full_text > text_excerpt > title fallback
        body = ft if len(ft) >= len(ex) else ex

        title = (p.get("title") or p.get("paper_title") or "Unknown").strip()
        year  = p.get("year") or p.get("publication_year")
        src   = p.get("source") or p.get("venue") or p.get("journal") or ""
        doi   = p.get("doi") or ""

        if not body and title and title != "Unknown":
            parts = [f'"{title}"']
            if year: parts.append(f"({year})")
            if src:  parts.append(f"in {src}")
            body = " ".join(parts)

        # Mark for opt-in runtime enrichment if body is too short for solid synthesis
        if _RUNTIME_ENRICH and len(body) < 1200:
            needs_runtime.append(idx)

        sources.append({
            "n":         p.get("rank", idx + 1),
            "title":     title,
            "year":      year,
            "doi":       doi,
            "source":    src,
            "snippet":   body[:500],     # short preview for UI sidebar
            "full_text": body,            # full body fed to LLM (≤4000 chars test / ≤15000 enriched)
        })
        scores.append(float(p.get("score", 768)))

    # ── Path B: optional runtime enrichment for short stubs ───────────────────
    # Only runs if EVIRAG_RUNTIME_ENRICH=1 and at least one paper is short.
    if _RUNTIME_ENRICH and needs_runtime:
        corpus_ids: List[str] = []
        for idx in needs_runtime:
            raw = papers[idx].get("openalex_id", "")
            corpus_ids.append(raw.replace("pes2o:", "") if raw else "")
        s2_map = await _s2_batch_meta([c for c in corpus_ids if c])

        async def _maybe_enrich(idx: int, cid: str):
            paper_id = papers[idx].get("openalex_id") or cid
            return idx, await _enrich_paper_text(paper_id, s2_map.get(cid, {}), sources[idx]["full_text"])

        enriched = await asyncio.gather(*[
            _maybe_enrich(i, cid) for i, cid in zip(needs_runtime, corpus_ids)
        ], return_exceptions=True)
        for r in enriched:
            if isinstance(r, Exception):
                continue
            idx, body = r
            if body and len(body) > len(sources[idx]["full_text"]):
                sources[idx]["full_text"] = body
                sources[idx]["snippet"]   = body[:500]

    top_dist  = min(scores) if scores else 768
    relevance = round(max(0.0, 1.0 - top_dist / 768), 3)
    return sources, relevance


# ── View / claim synthesis (one sentence per agent) ───────────────────────────
def _synthesize_view_reason(query: str, snippets: List[str], stance: str) -> str:
    import re as _re
    from ollama_cloud_client import get_cloud_client
    # Feed the FULL peS2o body text (up to 4000 chars per paper) for 4 papers —
    # 16k chars of evidence so the agent reasons from the whole paper body,
    # not just an opening sentence. The LLM's 128k context can take much more.
    paper_blocks = []
    for i, s in enumerate(snippets[:4], 1):
        s = s.strip()
        if s:
            paper_blocks.append(f"[Paper {i}]\n{s[:4000]}")
    snip_text = "\n\n".join(paper_blocks)
    if not snip_text:
        return f"No papers directly address the {stance} position on this topic."

    if stance == "dominant":
        instr = (
            "In 1-2 sentences: What specific finding or mechanism do these papers provide "
            "as the strongest evidence supporting the mainstream position? "
            "Name a concrete result, effect size, or biological mechanism if present. "
            "Be precise — never generic."
        )
    else:
        instr = (
            "In 1-2 sentences: What specific counter-evidence, replication failure, "
            "methodological limitation, or alternative explanation do these papers reveal? "
            "Cite a concrete result or critique — never just say 'there is disagreement'."
        )

    system_msg = (
        "You are a rigorous scientific analyst producing one key finding per agent turn. "
        "RULES: Never copy paper titles verbatim. Never say 'these papers suggest' or use "
        "hedged meta-commentary — state the finding directly. Respond ONLY from the provided texts. "
        "If the texts are irrelevant, say 'No directly relevant evidence found.' in one sentence."
    )
    user_msg   = f"Research question: {query}\n\nFull paper texts:\n{snip_text}\n\n{instr}"

    try:
        client = get_cloud_client()
        text = client.generate(prompt=user_msg, system=system_msg, temperature=0.22, max_tokens=160)
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
        # Pass agent views as anonymous "research perspectives" — NOT as agent outputs.
        # The LLM must NOT mention agents, Builder, Skeptic, Judge, or deliberation
        # in its final answer; those belong only in the separate deliberation panel.
        preamble = (
            f"Background research perspectives on this topic:\n{agent_views}\n\n"
            "Use these perspectives as evidence to ground your answer. "
            "Do NOT mention 'agents', 'Builder', 'Skeptic', 'Judge', 'Archivist', "
            "'deliberation', 'rounds', or any pipeline terminology in your answer. "
            "Write as a knowledgeable scientific expert giving a clear, balanced response."
        )

    agent_note = (
        " CRITICAL: Never mention 'agents', 'Builder agent', 'Skeptic', 'Judge', "
        "'deliberation', 'rounds', or any internal pipeline terminology in your answer. "
        "Write clean, readable prose as if you are an expert directly answering the question."
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
            + agent_note
        )
    else:
        system_msg = (
            "You are EVIRAG, an expert scientific research assistant. "
            "You are in a multi-turn conversation — maintain full context from previous turns. "
            "Write a thorough, well-structured answer of at least 5-6 sentences using scientific knowledge. "
            "Cover mechanisms, consensus, evidence, and nuances. Never give a short answer. "
            "Do NOT use [N] citation markers or fabricate references. "
            "End with a new line: '**Evolving Claim:**' followed by a one-sentence synthesis of the key finding."
            + agent_note
        )

    messages = [{"role": "system", "content": system_msg}]
    if chat_history:
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

    user_parts = []
    if preamble:
        user_parts.append(preamble)
    if has_ctx:
        # Pass the FULL retrieved corpus context (up to ~24k chars).
        # Modern LLMs handle 128k+ tokens — there is no reason to truncate here.
        user_parts.append(f"Retrieved passages (cite with [N] when directly applicable):\n{sources_block[:24000]}")
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
    """FAISS search + full-text enrichment + one-sentence synthesis for one agent."""
    import time
    t0 = time.perf_counter()

    agent_query = cfg["query_fn"](query)
    loop = asyncio.get_event_loop()

    # Full-text enriched search: FAISS → S2 batch → arXiv/PDF fetch (all async)
    sources, relevance = await _fetch_faiss_sources_enriched(agent_query, cfg["k"])

    # Use full_text for synthesis so the agent reasons from the whole paper,
    # not just the first 180 chars of an abstract. 4 papers × 4000 chars = 16k context.
    full_texts = [s.get("full_text") or s.get("snippet", "") for s in sources if s.get("full_text") or s.get("snippet")][:4]
    if full_texts:
        synthesis = await loop.run_in_executor(
            None, _synthesize_view_reason, query, full_texts, cfg["view_stance"]
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


# ── Deliberation helpers ───────────────────────────────────────────────────────
def _delib_llm(prompt: str, system: str, max_tokens: int = 140) -> str:
    """Blocking single-call LLM for deliberation exchange. Short, punchy output."""
    from ollama_cloud_client import get_cloud_client
    try:
        client = get_cloud_client()
        text = client.generate(prompt=prompt, system=system,
                               temperature=0.35, max_tokens=max_tokens)
        if not text:
            return ""
        text = text.strip()
        # Keep only the first sentence
        for end in ['. ', '! ', '? ']:
            idx = text.find(end)
            if 40 < idx < len(text) - 1:
                return text[:idx + 1].strip()
        return text[:260].strip()
    except Exception as e:
        print(f"[deliberation] LLM error: {e}")
        return ""


async def _run_deliberation(query: str, round1: List[Dict]) -> Dict:
    """
    Multi-round deliberation engine.

    After Round 1 (parallel independent retrieval + stance formation):
      • Judge identifies the exact factual conflict between Builder and Skeptic.
      • Builder and Skeptic each write a rebuttal (seeing the other's Round 1 view).
      • Judge renders a final verdict weighing the full exchange.

    All four calls run with maximum parallelism:
      - Judge conflict + Builder rebuttal + Skeptic rebuttal: 3 parallel
      - Judge verdict: 1 sequential (needs the rebuttals first)
    """
    import time
    t0 = time.perf_counter()

    builder  = next((r for r in round1 if r["agent_name"] == "builder"),   None)
    skeptic  = next((r for r in round1 if r["agent_name"] == "skeptic"),   None)
    archivist= next((r for r in round1 if r["agent_name"] == "archivist"), None)

    b_view = (builder   or {}).get("synthesis", "")
    s_view = (skeptic   or {}).get("synthesis", "")
    a_view = (archivist or {}).get("synthesis", "")

    if not b_view or not s_view:
        return {"conflict": "", "round2": [], "verdict": "", "duration_ms": 0}

    loop = asyncio.get_event_loop()

    # ── 3 parallel calls ───────────────────────────────────────────────────────
    conflict_prompt = (
        f"Research topic: {query}\n\n"
        f"Builder's finding: {b_view}\n"
        f"Skeptic's finding: {s_view}\n\n"
        "In ONE precise sentence: Identify the single most specific factual claim "
        "where Builder and Skeptic directly contradict each other. "
        "Name the exact mechanism, measurement, or study design in dispute — "
        "not a vague 'they disagree about X'. Be surgical."
    )
    builder_r_prompt = (
        f"Research topic: {query}\n\n"
        f"Your Round 1 finding: {b_view}\n"
        f"Skeptic challenges: {s_view}\n\n"
        "In ONE sentence: Rebut the Skeptic's specific challenge. "
        "Either cite a concrete piece of evidence that directly counters their critique, "
        "or acknowledge a narrow limitation while defending the core finding. "
        "No rhetorical hedging — give a factual, direct response."
    )
    skeptic_r_prompt = (
        f"Research topic: {query}\n\n"
        f"Your Round 1 counter-finding: {s_view}\n"
        f"Builder defends: {b_view}\n\n"
        "In ONE sentence: Sharpen your critique of Builder's evidence. "
        "Identify a specific methodological weakness, confound, or replication issue "
        "in their position — or acknowledge where their evidence is genuinely strong "
        "and redirect to a different vulnerability. Be specific and evidence-driven."
    )

    sys_judge   = (
        "You are a calibrated scientific arbiter — precise, evidence-based, and fair. "
        "Your job is to identify genuine factual disagreements, not rhetorical differences. "
        "Focus on empirical claims, not opinions. Never be vague."
    )
    sys_builder = (
        "You are the Builder agent in a structured scientific deliberation. "
        "You defend the dominant scientific view using specific empirical evidence. "
        "Be direct and cite concrete findings — never be vague or rhetorical."
    )
    sys_skeptic = (
        "You are the Skeptic agent in a structured scientific deliberation. "
        "Your role is to surface genuine weaknesses in the dominant view: "
        "replication failures, confounds, effect-size inflation, or alternative mechanisms. "
        "Be precise — name actual studies, methods, or results where possible."
    )

    conflict, builder_reb, skeptic_reb = await asyncio.gather(
        loop.run_in_executor(None, _delib_llm, conflict_prompt,  sys_judge,   100),
        loop.run_in_executor(None, _delib_llm, builder_r_prompt, sys_builder, 140),
        loop.run_in_executor(None, _delib_llm, skeptic_r_prompt, sys_skeptic, 140),
    )

    # ── Judge verdict (needs rebuttals first) ──────────────────────────────────
    verdict_prompt = (
        f"Research question: {query}\n\n"
        f"[Round 1] Builder: {b_view}\n"
        f"[Round 1] Skeptic: {s_view}\n"
        f"[Round 1] Archivist: {a_view}\n\n"
        f"[Round 2] Builder rebuttal: {builder_reb}\n"
        f"[Round 2] Skeptic rebuttal: {skeptic_reb}\n\n"
        "Issue your verdict in exactly 2 sentences:\n"
        "Sentence 1 — What does the preponderance of evidence support? "
        "Name specific findings, effect sizes, or mechanisms where they exist.\n"
        "Sentence 2 — What genuine scientific uncertainty or ongoing disagreement remains? "
        "Be specific about what is still unresolved and why. "
        "Do NOT be vague — 'more research is needed' is not acceptable."
    )
    sys_verdict = (
        "You are a calibrated scientific judge rendering a final verdict. "
        "Your ruling must be specific, evidence-grounded, and intellectually honest. "
        "Acknowledge what is well-established AND what remains genuinely contested. "
        "Never give a fence-sitting non-answer — take a defensible position."
    )
    verdict = await loop.run_in_executor(None, _delib_llm, verdict_prompt, sys_verdict, 200)

    duration_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "conflict": conflict,
        "round2": [
            {
                "agent":        "builder",
                "glyph":        "B",
                "name":         "Builder",
                "stance":       "support",
                "rebuttal":     builder_reb,
                "responding_to": "Skeptic",
            },
            {
                "agent":        "skeptic",
                "glyph":        "S",
                "name":         "Skeptic",
                "stance":       "contra",
                "rebuttal":     skeptic_reb,
                "responding_to": "Builder",
            },
        ],
        "verdict":     verdict,
        "duration_ms": duration_ms,
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

    # Feed FULL paper bodies (up to 4000 chars per paper × 6 papers = 24k chars).
    # The AI must be able to reference any sentence/word in the body — not just the abstract.
    src_block = ("\n\n".join(
        f"[{s['n']}] {s.get('title','').strip()}\n{(s.get('full_text') or s.get('snippet', ''))[:4000]}"
        for s in sources[:6] if s.get("full_text") or s.get("snippet")
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
            {"step": "judge_r1",    "label": "Judge: identifying conflict",           "detail": "Pinpointing the exact factual disagreement between Builder and Skeptic's Round 1 stances."},
            {"step": "rebuttals",   "label": "Round 2: agent rebuttals",              "detail": "Builder and Skeptic each respond to the other's argument — the actual debate happens here."},
            {"step": "verdict",     "label": "Judge: rendering verdict",              "detail": "Judge weighs the full exchange and produces a calibrated 2-sentence verdict."},
            {"step": "synthesis",   "label": "Final synthesis",                       "detail": "Main answer incorporates all rounds of deliberation into a unified response."},
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
        # ── Round 1: Run all 4 agents in parallel ─────────────────────────────
        agent_tasks = [
            _run_agent(key, cfg, request.message)
            for key, cfg in AGENT_CONFIGS.items()
        ]
        agent_results: List[Dict] = list(await asyncio.gather(*agent_tasks))

        # ── Deliberation: Judge identifies conflict, agents exchange rebuttals ─
        # Runs only if both Builder and Skeptic produced a real view.
        deliberation = await _run_deliberation(request.message, agent_results)

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

        # Build view synthesis block for the main answer — include deliberation
        agent_views_text = ""
        for ar in agent_results:
            if ar["synthesis"]:
                agent_views_text += f"- {ar['name']} ({ar['stance']}): {ar['synthesis']}\n"
        if deliberation.get("conflict"):
            agent_views_text += f"\nKey conflict identified: {deliberation['conflict']}\n"
        for r2 in deliberation.get("round2", []):
            if r2.get("rebuttal"):
                agent_views_text += f"- {r2['name']} (rebuttal): {r2['rebuttal']}\n"
        if deliberation.get("verdict"):
            agent_views_text += f"\nJudge verdict: {deliberation['verdict']}\n"

        # Main answer synthesis — feed FULL peS2o body text (up to 4000 chars per paper)
        # for 6 papers = 24 000 chars of evidence.  The AI must be able to cite any
        # sentence anywhere in the paper, not just the opening line.
        src_block = "\n\n".join(
            f"[{s['n']}] {s.get('title','').strip()}\n{(s.get('full_text') or s.get('snippet', ''))[:4000]}"
            for s in all_faiss_sources[:6]
            if s.get("full_text") or s.get("snippet")
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

        # Build fallback summaries from paper titles when synthesis is empty
        def _title_summary(ar: Dict, stance: str) -> str:
            titles = [s.get("title","") for s in ar.get("sources",[])[:3] if s.get("title")]
            if titles:
                t = "; ".join(f'"{t[:50]}"' for t in titles[:2])
                return f"Papers retrieved: {t}."
            return f"The {ar.get('name','Agent')} retrieved {ar.get('num_chunks',0)} papers from the peS2o corpus in a {stance} search."

        answer_obj: Dict[str, Any] = {}
        if builder_ar and (builder_ar["synthesis"] or builder_ar["sources"]):
            answer_obj["dominant_view"] = {
                "summary":              builder_ar["synthesis"] or _title_summary(builder_ar, "supporting"),
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
                "summary":              skeptic_ar["synthesis"] or _title_summary(skeptic_ar, "adversarial"),
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
        "deliberation":        deliberation if request.agents else {},
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
