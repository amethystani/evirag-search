---
title: EVIRAG Backend
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# EVIRAG Backend API

Evidence-Centric, Disagreement-Aware RAG backend for scientific literature.

## Environment Variables (Secrets)

Set these in the Space Settings → Repository secrets:

- `OLLAMA_API_KEY` — Bearer token for ollama.com cloud API
- `EVIRAG_SEARCH_URL` — URL of the EVIRAG search space (default: `http://localhost:7860`)

## Endpoints

- `GET /api/health` — Health check
- `POST /api/chat` — Multi-turn chat with FAISS retrieval + LLM synthesis
- `GET /api/sessions/{session_id}` — Get session history
