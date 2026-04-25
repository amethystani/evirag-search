---
title: EVIRAG Search
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: true
app_port: 7860
---

# EVIRAG Search API

Binary-quantized FAISS search over 40M+ scientific papers from OpenAlex.

Built for the EVIRAG (Evidence-centric, Disagreement-aware RAG) research project.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Keep-warm ping — always 200 |
| GET | `/status` | Index loading state + stats |
| POST | `/search` | Search papers by natural language query |

## Search Example

```bash
curl -X POST https://amethystani-evirag-search.hf.space/search \
  -H "Content-Type: application/json" \
  -d '{"query": "vaccine hesitancy covid misinformation", "k": 10}'
```

Response:
```json
{
  "query": "vaccine hesitancy covid misinformation",
  "results": [
    {
      "rank": 1,
      "openalex_id": "https://openalex.org/W2345678901",
      "title": "COVID-19 vaccine hesitancy...",
      "year": 2021,
      "doi": "https://doi.org/10.1016/...",
      "cited_by_count": 342,
      "source": "Vaccine",
      "score": 28.0
    }
  ],
  "latency_ms": 14.3,
  "index_size": 41230000
}
```

## Notes

- **Cold start**: ~5-15 min on first boot (downloads 4GB+ index from HF Datasets)
- **Warm queries**: 10-25ms typical latency
- **Score**: Hamming distance — lower = more similar
- **Index**: FAISS `IndexBinaryIVF` with 8192 clusters, nprobe=128
