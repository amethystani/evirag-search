# Tests

Integration and unit tests for the EVIRAG pipeline.

| File | Tests |
|------|-------|
| `smoke_test.py` | Quick end-to-end sanity check |
| `e2e_test.py` | Full pipeline end-to-end |
| `test_full_evirag.py` | Complete EVIRAG pipeline |
| `test_local_evirag.py` | Local Ollama backend |
| `test_cloud_evirag.py` | Cloud backend |
| `test_disagreement_local.py` | NLI and graph construction |
| `test_disagreement_cloud.py` | Cloud NLI |
| `test_contradiction.py` | Contradiction detection |
| `test_claims_analysis.py` | Claim extraction |
| `test_nli_diagnostic.py` | NLI pipeline diagnostics |
| `test_visual_grounding.py` | CLIP visual alignment |
| `test_synthetic_validation.py` | Synthetic gold-standard queries |
| `debug_nli_pipeline.py` | NLI debugging utilities |

Run a quick smoke test:

```bash
python tests/smoke_test.py
```
