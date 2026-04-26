"""
EVIRAG Main Backend — HuggingFace Spaces deployment
Slim version: uses Ollama cloud + remote FAISS search space, no local corpus.
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
        if isinstance(obj, np.bool_):   return bool(obj)
        if isinstance(obj, np.integer): return int(obj)
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

class ChatMessage(BaseModel):
    role: str
    content: str
    sources: Optional[List[Dict]] = None
    claim: Optional[str] = None
    turn: int = 0


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
        top_dist = min(scores) if scores else 768
        relevance = round(max(0.0, 1.0 - top_dist / 768), 3)
        return sources, relevance
    except Exception as e:
        print(f"[chat] FAISS backend unavailable: {e}")
        return [], 0.5


# ── View synthesis ─────────────────────────────────────────────────────────────
def _synthesize_view_reason(query: str, snippets: List[str], stance: str) -> str:
    import re as _re
    from ollama_cloud_client import get_cloud_client
    snip_text = " | ".join(s[:180] for s in snippets[:3] if s.strip())
    if not snip_text:
        return f"No papers directly address the {stance} position on this topic."

    instr = (
        "In ONE sentence, explain what these papers suggest supports or confirms the main position on this topic."
        if stance == "dominant" else
        "In ONE sentence, explain what alternative or contrasting perspective these papers raise about this topic."
    )
    system_msg = "You are a scientific analyst. Be concise. Do NOT copy paper titles verbatim. Base response ONLY on provided papers."
    user_msg   = f"Topic: {query}\nPapers: {snip_text}\n\n{instr}"

    try:
        client = get_cloud_client()
        text = client.generate(prompt=user_msg, system=system_msg, temperature=0.25, max_tokens=100)
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
        return text[:240]
    except Exception as exc:
        print(f"[chat] view reason error: {exc}")
        return f"These papers suggest a {stance} perspective on the topic."


# ── Answer synthesis ───────────────────────────────────────────────────────────
def _chat_synthesize(
    sources_block: str,
    message: str,
    chat_history: Optional[List[Dict]] = None,
) -> str:
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

    messages = [{"role": "system", "content": system_msg}]
    if chat_history:
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

    user_parts = []
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


# ── Perplexity answer builder ──────────────────────────────────────────────────
def _build_answer(message: str, faiss_sources: List[Dict], faiss_relevance: float,
                  session: Dict, turn: int):
    sources = faiss_sources or []

    # Gate: relevance + keyword match
    _raw = set()
    for w in message.lower().split():
        wc = w.strip("?!.,;:")
        _raw.add(wc)
        for part in wc.split("-"):
            if len(part) >= 3:
                _raw.add(part)
    query_terms = {t for t in _raw if len(t) >= 3}
    all_snippet_text = " ".join((s.get("snippet") or "").lower() for s in sources[:6])
    keyword_match = not query_terms or any(t in all_snippet_text for t in query_terms)
    has_relevant = faiss_relevance >= 0.70 and keyword_match and bool([s for s in sources[:8] if s.get("snippet")])

    if has_relevant:
        src_block = "\n".join(
            f"[{s['n']}] {s['snippet'][:250]}"
            for s in sources[:6] if s.get("snippet")
        )
    else:
        src_block = ""

    # Build chat history for multi-turn
    history = session.get("history", [])
    chat_history_messages: List[Dict] = []
    for h in history[-4:]:
        chat_history_messages.append({"role": "user",      "content": h["user"]})
        chat_history_messages.append({"role": "assistant", "content": h["answer"]})

    answer_md = _chat_synthesize(src_block, message, chat_history_messages)

    if not answer_md:
        answer_md = "I was unable to synthesize an answer at this time. Please try again."

    # Extract evolving claim
    claim_match = re.search(r'\*{0,2}Evolving Claim:\*{0,2}\s*(.+?)(?:\n|$)', answer_md, re.DOTALL)
    if claim_match:
        claim = claim_match.group(1).strip()
    else:
        sentences = [s.strip() for s in re.split(r'[.!?]+', answer_md) if s.strip()]
        claim = sentences[-1] if sentences else message

    # Strip evolving claim block from answer body
    answer_body = re.sub(
        r'(?i)(?:The best(?:-| )supported position is:)?\s*\*{0,2}Evolving Claim:\*{0,2}.*$',
        '', answer_md, flags=re.DOTALL
    ).strip()

    return answer_body, claim, sources


# ── App ────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[evirag-backend] Starting. Search URL: {SEARCH_URL}")
    yield

app = FastAPI(
    title="EVIRAG Backend",
    description="Evidence-Centric, Disagreement-Aware RAG — HF Spaces deployment",
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


@app.get("/api/health")
async def health():
    return {"status": "ok", "search_url": SEARCH_URL}


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

    # Enriched query for context continuity
    acc = session["accumulated_context"]
    enriched_query = f"{acc}. Now specifically: {request.message}" if acc else request.message
    new_acc = f"{acc}; {request.message}" if acc else request.message
    session["accumulated_context"] = " ".join(new_acc.split()[-120:])

    # FAISS search
    _faiss_query = enriched_query if len(request.message.split()) < 5 else request.message
    loop = asyncio.get_event_loop()
    faiss_sources, faiss_relevance = await loop.run_in_executor(
        None, _fetch_faiss_sources, _faiss_query
    )

    # Build answer
    answer_body, claim, sources = await loop.run_in_executor(
        None, _build_answer, request.message, faiss_sources, faiss_relevance, session, turn
    )

    # Build EVIRAG-style result structure
    answer_obj: Dict[str, Any] = {}

    # Views (only when relevant papers found at high confidence)
    views_relevant = faiss_relevance >= 0.75 and bool(faiss_sources)
    if views_relevant:
        def _view(srcs, name, stance):
            snips = [s["snippet"] for s in srcs if s.get("snippet")]
            if not snips: return None
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
        dom = _view(faiss_sources[:3],  "Most relevant (peS2o corpus)", "dominant")
        alt = _view(faiss_sources[3:6], "Related perspective",          "alternative")
        if dom: answer_obj["dominant_view"] = dom
        if alt: answer_obj["alternative_views"] = [alt]

    answer_obj["minority_views"] = []

    # Claims from evolving claim text
    claim_entry = {
        "id":    f"claim_{turn}",
        "text":  claim,
        "type":  "synthesis",
        "confidence": 0.8,
        "source_doc_title": "EVIRAG synthesis",
    }
    new_claims = [claim_entry]

    # Accumulate across turns
    seen_ids = {c.get("id") for c in session["acc_claims"]}
    for c in new_claims:
        if c.get("id") not in seen_ids:
            session["acc_claims"].append(c)
    session["acc_claims"] = session["acc_claims"][-60:]

    for s in sources:
        key = (s.get("title") or "")[:50]
        if key and key not in session["acc_sources"]:
            session["acc_sources"][key] = s
    if len(session["acc_sources"]) > 40:
        keys = list(session["acc_sources"].keys())
        session["acc_sources"] = {k: session["acc_sources"][k] for k in keys[-40:]}

    session["history"].append({
        "turn":    turn,
        "user":    request.message,
        "answer":  answer_body,
        "claim":   claim,
        "sources": [s["title"] for s in sources[:5]],
    })
    session["claim"] = claim
    session["turn"]  = turn
    if len(session["history"]) > 20:
        session["history"] = session["history"][-20:]

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    acc_sources_list = list(session["acc_sources"].values())

    result = {
        "claims":  session["acc_claims"],
        "sources": [
            {
                "doc_id":  s.get("doi") or s["title"][:12],
                "title":   s["title"],
                "year":    s.get("year"),
                "doi":     s.get("doi") or "",
                "source":  s.get("source") or "",
                "snippet": s.get("snippet") or "",
                "text":    s.get("snippet") or "",
                "cited_by_count": None,
            }
            for s in faiss_sources
        ],
        "answer":     answer_obj,
        "statistics": {"total_sources": len(faiss_sources)},
        "chat": {
            "session_id":   session_id,
            "answer":       answer_body,
            "claim":        claim,
            "sources":      acc_sources_list,
            "turn":         turn,
            "latency_ms":   latency_ms,
            "total_claims":  len(session["acc_claims"]),
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
        },
    }
    return _sanitize(result)


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
