# EVIRAG Final System Breakdown

## 1. Project Identity

**Project name:** EVIRAG  
**Expanded name:** Evidence-Centric, Disagreement-Aware Retrieval-Augmented Generation  
**Project type:** Research-oriented scientific literature QA and analysis system  
**Primary user goal:** Ask natural-language questions over a local corpus of scientific papers and receive answers that do not collapse disagreement into a single flat summary.

At a high level, EVIRAG is trying to solve a problem that normal RAG systems handle poorly: when the literature is mixed, contradictory, conditional, or context-dependent, a standard retrieve-and-summarize pipeline usually produces one smoothed-over answer. That is often fast, but epistemically weak. It hides uncertainty, erases minority views, and fails to tell the user *why* papers disagree.

The final working version of this project addresses that by introducing:

1. A **multi-agent evidence gathering pipeline** for deeper reasoning.
2. A **disagreement graph** over extracted claims and their support/contradiction relations.
3. A **fast offline claim-graph mode** that moves expensive relationship work to index time.
4. **Citations, confidence, alternative views, and minority views** in the answer surface.
5. Multiple execution modes so the system can trade off speed and reasoning depth.

The current system is therefore not just “chat with PDFs.” It is a structured evidence analysis system with a chat interface.

---

## 2. What Problem the Project Solves

### 2.1 The core problem

Most classical RAG systems do this:

1. Embed a query.
2. Retrieve top-k chunks.
3. Feed those chunks to an LLM.
4. Ask the LLM to write one answer.

That works reasonably well when:

- the evidence is internally consistent,
- the question is descriptive,
- the corpus is homogeneous,
- and a single summary is acceptable.

It works much worse when:

- the literature is mixed,
- the same intervention helps some populations and harms others,
- different papers operationalize the same concept differently,
- papers disagree because of methodology, population, measurement, time, or context,
- or the user explicitly wants nuanced, evidence-grounded reasoning.

### 2.2 Why homework is a good demonstration domain

The current corpus is about homework and academic outcomes. This is a good demonstration domain because the question is not a settled binary fact. Different papers report:

- positive associations between homework and achievement,
- conditional benefits by grade level or aptitude,
- negative effects for some groups,
- trade-offs involving stress, leisure, and mental health,
- and context-sensitive effects in science education.

That makes it a natural testbed for a system that claims to reason about **disagreement**, not just retrieve text.

### 2.3 EVIRAG’s response to that problem

EVIRAG tries to preserve the structure of the evidence rather than flatten it immediately.

It does this in two ways:

1. **Full EVIRAG pipeline**
   - Intent analysis
   - Explanation hypothesis generation
   - Multi-agent retrieval
   - Claim extraction
   - Pairwise relationship reasoning
   - Disagreement graph construction
   - Multi-view synthesis

2. **Fast claim-graph pipeline**
   - Precompute claim nodes and many claim relationships offline
   - Cache them in a persistent graph
   - At query time, retrieve a relevant subgraph
   - Synthesize multi-view answers from that subgraph

The second path is the major system-level optimization that made the project practically usable as a local corpus chat system.

---

## 3. Corpus Inventory

### 3.1 Exact file count and raw size

The current corpus directory contains **7 PDF files** with a total raw size of **2,872,027 bytes**.

That is:

- **2.74 MiB** using binary units (`2,872,027 / 1,048,576`)
- **2.87 MB** using decimal units (`2,872,027 / 1,000,000`)

### 3.2 Exact corpus file list

| File | Size (bytes) | Pages | What it contains |
|---|---:|---:|---|
| `1305.2213v1.pdf` | 725,613 | 16 | A physics education study on how homework completion relates to exam performance across different student aptitude levels. Important because it contains conditional and even negative effects for lower-aptitude students. |
| `Impact_of_Homework_on_the_Student_Academic_Perform.pdf` | 827,262 | 10 | A secondary-school study on the impact of homework on academic performance. Emphasizes homework as a school-home linkage and reports positive learning impact in its observed setting. |
| `cooperrobinsonpatall_2006.pdf` | 293,832 | 62 | A major review article, “Does Homework Improve Academic Achievement? A Synthesis of Research, 1987–2003.” This is a key survey paper and a strong backbone source for the corpus. |
| `exploring-the-impact-of-homework-assignments-on-achievement-and-attitudes-in-science-education-13058.pdf` | 345,821 | 8 | A 2023 review article focused specifically on science education, including achievement and attitudes. Important because it brings in domain-specific nuance rather than only general education outcomes. |
| `homework_achievement_mystery_trautwein.pdf` | 353,468 | 7 | A paper connected to homework, time spent, management, and academic performance, emphasizing complexity in how homework effects should be interpreted. |
| `homework_negative_effects_ERIC.pdf` | 324,706 | 15 | “Does Homework Work or Hurt? A Study on the Effects of Homework on Mental Health and Academic Performance.” Important because it explicitly introduces a harm-oriented, mental-health-sensitive viewpoint. |
| `stanford_too_much_homework_stress.pdf` | 1,325 | 1 | Not a valid research article in practice. It is an “Access Denied” page rather than usable scholarly content. It is technically present in the corpus directory, but it is effectively noise and should be treated as a malformed corpus artifact. |

### 3.3 What the corpus contains semantically

The corpus is not a random set of PDFs. It is a focused literature packet around:

- homework effectiveness,
- academic achievement,
- science education,
- student attitudes,
- differential effects by student group,
- and possible negative outcomes such as stress and mental health burden.

This matters because EVIRAG’s disagreement-aware design is easiest to justify when the domain is **not** purely consensus-driven.

### 3.4 Important corpus quality note

One file, `stanford_too_much_homework_stress.pdf`, is effectively an **Access Denied placeholder**, not a meaningful academic source. That means:

1. The raw corpus file count is 7.
2. The *substantively useful* corpus is effectively 6 real papers plus 1 noisy artifact.

This distinction should be stated clearly in any final report, because otherwise the corpus description will overstate the number of genuine research documents.

### 3.5 Indexed corpus statistics in the final working system

The processed artifacts currently show:

- **277 text chunks** in the vector index
- **631 atomic claims** in the offline claim graph
- **385 cached claim-relationship edges** in the final local-verified graph
- **384-dimensional embeddings** for both chunk and claim retrieval

These indexed statistics matter because the runtime system does not operate directly on 7 full PDFs. It operates on:

1. chunked text passages for retrieval,
2. extracted claims for evidence reasoning,
3. and graph edges for disagreement-aware answer generation.

---

## 4. Full Model Inventory

This section lists **all models used or configured in the codebase**, including local, cloud, embedding, and visual models.

## 4.1 Local language model backend

### Primary local model

- **Model:** `qwen3:0.6b`
- **Serving framework:** Ollama
- **Endpoint:** `http://localhost:11434`
- **Role in the project:** This is the main local reasoning model used across almost the entire final local stack.

The design decision here is unusually aggressive: instead of using many different local models, the system unifies most language tasks onto one small local model. That keeps deployment simple, but it also creates pressure on latency. That pressure is exactly what motivated the offline claim graph optimization.

### Local SLM task assignments

The following “small language model” tasks are configured to use `qwen3:0.6b` locally:

1. `intent_analyzer`
   - Purpose: classify the query along epistemic dimensions such as factual vs disputed and expected disagreement level.

2. `claim_filter`
   - Purpose: identify sentences in retrieved text that are worth turning into structured claims.

3. `nli_prefilter`
   - Purpose: cheaply screen relationship candidates before expensive verification.

4. `synthesizer`
   - Purpose: lighter synthesis stage when a smaller/faster answering step is appropriate.

### Local LLM task assignments

The following “larger reasoning task” roles are also configured to use the same local `qwen3:0.6b` model, but with larger token budgets:

1. `hypothesis_generator`
   - Purpose: generate an explanation-oriented hypothesis structure from the user query.

2. `claim_extractor`
   - Purpose: convert candidate evidence sentences into atomic normalized claims.

3. `nli_verifier`
   - Purpose: infer support, contradiction, or neutrality between claim pairs.

4. `synthesizer_escalation`
   - Purpose: stronger synthesis path when richer answer generation is needed.

### Local agent model assignments

All four retrieval/reasoning agents also use `qwen3:0.6b` locally:

1. `precision`
   - Purpose: seek strong supporting evidence.

2. `recall`
   - Purpose: seek broad evidence coverage.

3. `skeptic`
   - Purpose: seek contradictory or challenging evidence.

4. `counterfactual`
   - Purpose: seek alternative explanations and disproof-oriented evidence.

### Why one small local model matters architecturally

This is one of the most important points in the entire project:

- The system’s original slowness was not just “RAG is slow.”
- It was specifically that **many separate reasoning steps** were all mapped onto one small local model.
- That meant the system paid repeated inference costs for claim extraction, NLI, and agent reasoning.

The offline claim graph is therefore not a cosmetic optimization. It is the central architectural response to the constraints of a unified small-model local setup.

---

## 4.2 Cloud model backend

Even though the final current frontend is local-first and the user asked to remove Groq dependency from the main experience, the codebase still contains a complete cloud model registry and Groq client path. That should be documented because it is part of the project’s full system design.

### Cloud provider

- **Provider:** Groq
- **API endpoint:** `https://api.groq.com/openai/v1`

### Cloud SLM models

1. `llama-3.1-8b-instant`
   - Used for:
     - `intent_analyzer`
     - `claim_filter`
     - `nli_prefilter`
     - `synthesizer`
   - Purpose: fast cloud inference for lighter tasks.

### Cloud LLM models

1. `llama-3.3-70b-versatile`
   - Used for:
     - `hypothesis_generator`
     - `claim_extractor`
     - `nli_verifier`
     - `synthesizer_escalation`
   - Purpose: stronger reasoning and structured generation in cloud mode.

### Cloud agent models

1. `llama-3.3-70b-versatile`
   - Used for:
     - `precision`

2. `llama-3.1-8b-instant`
   - Used for:
     - `recall`
   - Rationale in config: faster broad retrieval.

3. `deepseek-r1-distill-llama-70b`
   - Used for:
     - `skeptic`
     - `counterfactual`
   - Purpose: stronger adversarial and alternative-explanation reasoning.

### Why document cloud mode if current UI is local

Because the system was clearly designed as a dual-backend architecture:

- **Local mode** for self-contained deployment
- **Cloud mode** for larger, stronger reasoning when latency/cost trade-offs permit it

The official report should present this honestly:

- the **current final interactive frontend is local-first**,
- but the project codebase still supports a broader backend design.

---

## 4.3 Embedding and retrieval models

### Text embedding model

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding dimension:** `384`
- **Purpose:**
  - build chunk embeddings,
  - embed user queries,
  - build claim embeddings,
  - retrieve relevant chunks and claims using FAISS.

This model is critical to both:

1. the standard vector retrieval baseline, and
2. the fast offline claim-graph path.

### Visual embedding model

- **Model:** `openai/clip-vit-base-patch16`
- **Configured dimension:** `512`
- **Purpose:** visual grounding and visual-text alignment analysis when visual mode is enabled.

In the final local chat frontend, visual grounding is disabled by default, but the code still supports it as a research feature.

---

## 4.4 Visual and auxiliary model pathways

### CLIP-based visual evidence processor

The codebase includes a visual pipeline through `vlm_module.py`, using the configured CLIP model for:

- figure processing,
- visual evidence indexing,
- and visual disagreement analysis.

This is not the dominant path in the current final demo configuration, but it is part of the system’s research scope and should be documented as an optional modality extension.

### No separate custom NLI model

An important design point is that the project does **not** use a dedicated compact NLI classifier such as a specialized DeBERTa or RoBERTa entailment model. Instead:

- claim relationship verification is performed through the same general-purpose language backend.

This simplifies the stack but is also a major reason the original online pipeline became too slow.

---

## 5. Final Working Version: What the System Does

The final working version is best understood as a **multi-mode corpus reasoning system** with one baseline mode and several EVIRAG modes.

### 5.1 Supported execution modes

The system exposes two top-level execution families:

1. `vanilla_rag`
2. `evirag`

Within `evirag`, the frontend exposes speed/depth presets:

1. **Fast**
2. **Balanced**
3. **Deep Reasoning**

The current Streamlit frontend also exposes:

4. **Vanilla RAG**

So the user-facing chat application effectively offers four modes.

### 5.2 What each mode means

#### A. Vanilla RAG

This is the baseline.

Pipeline:

1. Embed query
2. Retrieve top chunks from FAISS
3. Concatenate retrieved text
4. Ask the synthesizer model for a single answer

Output characteristics:

- one answer,
- source list,
- no structured claim graph,
- no alternative/minority viewpoint decomposition,
- no disagreement-aware reasoning.

This mode is useful as a benchmark and ablation baseline.

#### B. Fast EVIRAG

This is the key final optimization path.

Pipeline:

1. Use the offline claim graph if it is ready.
2. Retrieve a relevant **claim subgraph** instead of running the full online claim-building pipeline.
3. Run disagreement reasoning over that subgraph.
4. Return a multi-view answer with citations, claims, graph, and metrics.
5. Optionally **fallback** to the full EVIRAG pipeline if the subgraph is too weak, too sparse, or too contradictory.

Output characteristics:

- multi-view answer,
- dominant + alternative + minority views,
- citations linked to claims,
- structured claim list,
- visible graph,
- sub-second warmed performance in the benchmarked setup.

#### C. Balanced EVIRAG

This is the slower but deeper full reasoning mode.

Pipeline:

1. Intent analysis
2. Hypothesis generation
3. Multi-agent retrieval
4. Claim extraction
5. Pairwise relationship reasoning
6. Disagreement graph construction
7. Multi-view synthesis
8. Metrics and optional research modules

This mode is epistemically richer but far more expensive.

#### D. Deep Reasoning

This is effectively the same full pipeline family as Balanced, but positioned as the slow, thorough, reasoning-heavy mode.

In the current frontend, it primarily serves as a user-facing “take more time, reason more deeply” mode label.

---

## 6. End-to-End System Architecture

## 6.1 Data ingestion and indexing

Main file: [data_layer.py](/Users/animesh/Downloads/Project-2-3/data_layer.py)

This module is responsible for turning raw PDFs into structured retrieval assets.

### Responsibilities

1. **PDF metadata extraction**
   - title
   - author
   - year
   - page count
   - path

2. **Text extraction with section awareness**
   - The system tries to preserve sections such as abstract, introduction, methods, results, discussion, and conclusion.

3. **Chunking**
   - Current configured chunk size: `256`
   - Chunk overlap: `32`
   - This yields the current indexed total of `277` chunks.

4. **Figure extraction**
   - The code can extract figures and captions from PDFs.

5. **Embedding generation**
   - Uses `all-MiniLM-L6-v2`

6. **Vector index construction**
   - Uses `FAISS IndexFlatL2`

7. **Offline claim graph construction**
   - After chunk indexing, the system can also build a claim-level graph for fast reasoning.

### Why this matters

This layer is the foundation for both the baseline and advanced system:

- Vanilla RAG uses the chunk index directly.
- Fast EVIRAG uses the chunk-derived claim graph.
- Full EVIRAG uses chunks for retrieval and then constructs reasoning structures online.

---

## 6.2 Epistemic engine

Main file: [epistemic_engine.py](/Users/animesh/Downloads/Project-2-3/epistemic_engine.py)

This is the reasoning preparation layer. It converts an arbitrary question into a structured evidence-analysis problem.

### Main responsibilities

1. **Epistemic intent analysis**
   - Determines whether the query is factual vs disputed
   - Determines empirical vs theoretical character
   - Predicts expected disagreement level

2. **Explanation hypothesis generation**
   - Converts the user query into:
     - a central hypothesis,
     - supporting claims,
     - assumptions,
     - expected counterclaims

3. **Claim extraction**
   - Extracts atomic claims from text chunks
   - Normalizes them into structured units

4. **NLI reasoning**
   - Determines whether claim pairs support, contradict, or remain neutral

### Why this matters

This is what separates EVIRAG from ordinary RAG. Instead of going straight from retrieval to answering, the system inserts an **epistemic modeling layer**:

- “What kind of question is this?”
- “What explanatory structure should we test?”
- “What claims did the literature actually make?”
- “How do those claims relate?”

That is a research-oriented design decision, not just an engineering one.

---

## 6.3 Multi-agent deliberative retrieval

Main file: [multi_agent.py](/Users/animesh/Downloads/Project-2-3/multi_agent.py)

This module implements the multi-agent retrieval strategy.

### Agent roster

#### 1. Precision agent

- Objective: find high-confidence support
- Retrieval configuration:
  - `retrieval_k = 3`
- Behavior:
  - seeks strong supporting evidence
  - evaluates whether claims support the central hypothesis

#### 2. Recall agent

- Objective: broad coverage
- Retrieval configuration:
  - `retrieval_k = 5`
- Behavior:
  - casts a wider semantic net
  - prioritizes coverage over confidence

#### 3. Skeptic agent

- Objective: find contradictions
- Retrieval configuration:
  - `retrieval_k = 4`
- Behavior:
  - actively searches for claims that challenge or contradict the hypothesis

#### 4. Counterfactual agent

- Objective: alternative explanations and disproof attempts
- Retrieval configuration:
  - `retrieval_k = 3`
- Behavior:
  - explores what-if reasoning and alternative causal mechanisms

### Orchestrator behavior

The orchestrator:

1. creates shared agent state,
2. runs agents,
3. accumulates evidence,
4. tracks retrieval history,
5. and summarizes agent-level stances/confidences.

### Important implementation note

The original full pipeline executes agents **sequentially**, not fully in parallel. This was part of the original latency bottleneck on the local single-model stack.

That matters for the report because the fast claim-graph path should be positioned as the practical response to this cost structure.

---

## 6.4 Disagreement graph and answer synthesis

Main file: [disagreement.py](/Users/animesh/Downloads/Project-2-3/disagreement.py)

This is the module that turns evidence into structured viewpoints.

### What it does

1. Builds a graph where:
   - nodes are claims
   - edges are support/contradiction/neutral relations

2. Computes disagreement metrics such as:
   - disagreement density
   - conflict ratio
   - claim entropy
   - conflict centrality
   - visual-text mismatch

3. Detects communities or clusters of claims

4. Generates viewpoint-level summaries
   - dominant view
   - alternative views
   - minority views

5. Calibrates answer confidence

### Why this matters

This module is where the system becomes explicitly disagreement-aware.

Instead of writing:

> “Homework is good”  
or  
> “Homework is bad”

it tries to write something closer to:

- one dominant interpretation,
- plus additional supported interpretations,
- plus outlier but relevant positions,
- with an explicit confidence assessment.

That is the core intellectual contribution of the project.

---

## 6.5 Offline claim graph for fast reasoning

Main file: [claim_graph.py](/Users/animesh/Downloads/Project-2-3/claim_graph.py)

This is the most important systems innovation in the final working version.

### Motivation

The original online EVIRAG pipeline was too slow because it did expensive claim-pair relationship reasoning at query time. That meant the user paid the cost every time they asked a question.

### What the offline claim graph does

At index time, it:

1. extracts claims from chunks,
2. embeds claims,
3. builds a claim index,
4. constructs a sparse relationship graph,
5. optionally verifies edges offline,
6. persists claims, relationships, embeddings, and metadata to disk.

### What it stores

Artifacts currently include:

- `claim_index.faiss`
- `claims.pkl`
- `claim_relationships.pkl`
- `claim_embeddings.npy`
- `chunk_to_claims.json`
- `claim_pair_cache.json`
- `claim_graph_metadata.json`

### Final current graph config

- `enabled = True`
- `max_claims_per_chunk = 3`
- `candidate_neighbors = 6`
- `query_seed_k = 8`
- `query_neighbor_expansion = 3`
- `max_query_edges = 40`
- `verify_edges = "local"`
- `verification_budget_local = 8`
- `verification_budget_cloud = 60`
- `always_verify_contradictions = True`

### Query-time behavior

At query time, Fast EVIRAG:

1. embeds the user query,
2. retrieves seed claims,
3. expands locally through the claim graph,
4. extracts a subgraph,
5. reasons over that subgraph,
6. synthesizes the answer.

This means the expensive pairwise reasoning is shifted away from the online path.

### Why this is architecturally important

This is the central answer to the speed problem. It changes EVIRAG from:

- “expensive online reasoning over raw retrieved text”

to:

- “cheap online reasoning over a precomputed evidence graph.”

That is exactly why the final warmed fast mode becomes practically sub-second.

---

## 6.6 System orchestrator

Main file: [evirag_system.py](/Users/animesh/Downloads/Project-2-3/evirag_system.py)

This is the main control layer. It wires the entire system together.

### Core responsibilities

1. initialize all major components
2. initialize or rebuild corpus assets
3. dispatch queries to:
   - `vanilla_rag`
   - full EVIRAG
   - fast claim-graph EVIRAG
4. manage fallback logic
5. package outputs into UI/API-consumable records

### Key execution paths

#### `_vanilla_rag`

- standard vector retrieval
- single synthesized answer

#### `_evirag_full_pipeline`

- full multi-agent, claim extraction, online relationship reasoning, graph building, synthesis, metrics

#### `_evirag_fast_graph`

- claim-subgraph retrieval
- fast disagreement reasoning
- citation-enriched answer
- optional fallback to full pipeline

### Why this matters

This file is the actual “system brain” from an engineering standpoint. If the report needs one file to present as the backbone of the software architecture, this is it.

---

## 6.7 Research modules beyond basic QA

The project also includes research-oriented extensions that go beyond simple corpus chat.

### A. Epistemic divergence

Main file: [epistemic_divergence.py](/Users/animesh/Downloads/Project-2-3/epistemic_divergence.py)

Purpose:

- quantify how far viewpoints diverge,
- estimate polarization-like structure,
- measure consensus collapse.

This is meant to formalize disagreement rather than leaving it as a purely narrative phenomenon.

### B. Causal disagreement attribution

Main file: [causal_attribution.py](/Users/animesh/Downloads/Project-2-3/causal_attribution.py)

Purpose:

- explain *why* papers disagree,
- attribute disagreement to causes such as methodological differences, contextual differences, or alternative explanatory mechanisms.

### C. Temporal epistemic drift

Main file: [temporal_tracker.py](/Users/animesh/Downloads/Project-2-3/temporal_tracker.py)

Purpose:

- examine how viewpoints or disagreements evolve over time.

### D. Evaluation framework

Main file: [evaluation_framework.py](/Users/animesh/Downloads/Project-2-3/evaluation_framework.py)

Purpose:

- define evaluation metrics appropriate for disagreement-aware systems,
- not just conventional answer quality.

This is important academically, because a system like EVIRAG should not be judged only by “did it sound good?”

---

## 7. User Interfaces and External Access

## 7.1 Streamlit frontend

Main file: [streamlit_app.py](/Users/animesh/Downloads/Project-2-3/streamlit_app.py)

### What it provides

1. A normal chat-style interface over the corpus
2. Mode selection in the sidebar
3. Warm/rebuild controls
4. Reasoning display for deeper modes
5. Claim graph visualization
6. Evidence, metrics, and source inspection

### User-facing modes in the frontend

1. Fast
2. Balanced
3. Deep Reasoning
4. Vanilla RAG

### Final frontend behavior

The final frontend is no longer a narrow benchmark harness. It is intended to behave like a corpus chat application:

- the user asks normal questions,
- the system answers,
- the selected mode determines how much reasoning depth and structure is exposed.

This is the correct framing for the final report.

---

## 7.2 FastAPI backend

Main file: [fastapi_service.py](/Users/animesh/Downloads/Project-2-3/fastapi_service.py)

### Purpose

The FastAPI service provides programmatic access to the EVIRAG system and supports frontend/backend decoupling.

### Main API routes

1. `/api/initialize`
2. `/api/process_query`
3. `/api/rebuild_index`
4. `/api/batch_process`
5. `/api/evaluate_modes`

### Why it matters

This means the project is not only a notebook or one-off script. It has a service interface suitable for integration, testing, and frontend connection.

---

## 7.3 CLI entrypoint

Main file: [run.py](/Users/animesh/Downloads/Project-2-3/run.py)

Purpose:

- quick start,
- CLI demo,
- manual testing of system behavior outside the web frontend.

This is useful for reproducibility, demos, and debugging.

---

## 8. Main Functionalities in the Final Working Version

The final system supports all of the following major capabilities.

### 8.1 Ask questions over a local corpus

The user can ask natural language questions such as:

- “Does homework improve academic achievement?”
- “Is homework bad?”
- “What are the trade-offs of homework for student attitudes?”

The system will answer using the indexed local corpus rather than open-web knowledge.

### 8.2 Return structured, disagreement-aware answers

Instead of only one answer string, EVIRAG can return:

- dominant view,
- alternative views,
- minority views,
- confidence level,
- confidence score,
- confidence reasoning.

### 8.3 Show citations linked to claims

The system enriches viewpoints with source-linked claim citations. This is critical for research credibility and report writing because the answer is not supposed to look like an unsupported LLM opinion.

### 8.4 Show structured evidence

The final UI/API can expose:

- claims,
- sources,
- graph nodes,
- graph edges,
- metrics,
- reasoning trace.

### 8.5 Fast online answering via offline graph retrieval

This is the system’s main performance feature:

- expensive graph-building work is moved offline,
- online answering becomes claim-subgraph retrieval plus synthesis.

### 8.6 Automatic fallback for accuracy protection

Fast mode can automatically switch to the full pipeline when:

- too few claims are retrieved,
- too few relationships are available,
- contradiction density is too high,
- or the evidence looks too small/mixed to trust the fast path.

This matters because pure speed without guardrails would hurt answer quality.

### 8.7 Optional visual grounding

The system can process figures and visual evidence, although this is not the default path in the current local frontend.

### 8.8 Evaluation and benchmarking support

The project includes scripts and utilities for:

- fast-graph benchmarking,
- mode comparison,
- broader evaluation experimentation.

---

## 9. Why the Fast Graph Version Is the Final Practical Architecture

This section is worth stating explicitly because it is the main systems argument of the project.

### 9.1 The original bottleneck

The expensive part of the old pipeline was online claim-pair reasoning:

- claim extraction per retrieved chunk,
- pairwise NLI checks among claims,
- and sequential multi-agent behavior on a local small model.

That made a “simple query” take many minutes.

### 9.2 The key redesign

The redesigned architecture performs:

- **claim extraction offline**
- **claim embedding offline**
- **claim relationship estimation/caching offline**
- **graph persistence offline**

Then at query time it performs:

1. query embedding,
2. claim-subgraph lookup,
3. disagreement reasoning,
4. citation-enriched answer synthesis.

### 9.3 Why this is a principled redesign, not just an optimization hack

This is important for your report:

- It changes *where* the expensive reasoning happens.
- It changes the **unit of retrieval** from chunk to claim.
- It changes the answering substrate from flat retrieved text to a structured evidence graph.
- It preserves disagreement structure while improving latency.

That is an architectural contribution.

---

## 10. Main Code Sections and Their Roles

Below is the most useful “code map” for the final project.

| File | Role in final system |
|---|---|
| [config.py](/Users/animesh/Downloads/Project-2-3/config.py) | Global configuration, model registry, prompts, thresholds, graph settings, UI/system constants |
| [data_layer.py](/Users/animesh/Downloads/Project-2-3/data_layer.py) | PDF ingestion, metadata extraction, chunking, embedding, vector indexing, figure extraction |
| [claim_graph.py](/Users/animesh/Downloads/Project-2-3/claim_graph.py) | Offline claim graph build/load/query, cached relationships, claim-level FAISS index |
| [epistemic_engine.py](/Users/animesh/Downloads/Project-2-3/epistemic_engine.py) | Intent analysis, hypothesis generation, claim extraction, NLI reasoning, Ollama client |
| [multi_agent.py](/Users/animesh/Downloads/Project-2-3/multi_agent.py) | Precision/recall/skeptic/counterfactual agents and orchestrator |
| [disagreement.py](/Users/animesh/Downloads/Project-2-3/disagreement.py) | Disagreement graph, metrics, community detection, viewpoint synthesis, confidence calibration |
| [evirag_system.py](/Users/animesh/Downloads/Project-2-3/evirag_system.py) | End-to-end orchestration, mode routing, fast/full pipeline execution, fallback control, API/UI-ready outputs |
| [streamlit_app.py](/Users/animesh/Downloads/Project-2-3/streamlit_app.py) | Main chat frontend with modes, reasoning display, graph visualization, evidence panels |
| [fastapi_service.py](/Users/animesh/Downloads/Project-2-3/fastapi_service.py) | API service for initialization, querying, rebuilding, benchmarking |
| [run.py](/Users/animesh/Downloads/Project-2-3/run.py) | CLI entrypoint/demo |
| [epistemic_divergence.py](/Users/animesh/Downloads/Project-2-3/epistemic_divergence.py) | Quantitative disagreement/divergence analysis |
| [causal_attribution.py](/Users/animesh/Downloads/Project-2-3/causal_attribution.py) | Attribution of why sources disagree |
| [temporal_tracker.py](/Users/animesh/Downloads/Project-2-3/temporal_tracker.py) | Temporal drift analysis over claims/evidence |
| [evaluation_framework.py](/Users/animesh/Downloads/Project-2-3/evaluation_framework.py) | Research evaluation metrics and benchmark framing |
| [benchmark_fast_graph.py](/Users/animesh/Downloads/Project-2-3/benchmark_fast_graph.py) | Fast-path runtime benchmarking |
| [mode_comparison.py](/Users/animesh/Downloads/Project-2-3/mode_comparison.py) | Comparison of fast/full behavior across queries |

---

## 11. Important Configuration Values in the Final Version

These values are especially important to document because they shape both behavior and performance.

### Retrieval and chunking

- `chunk_size = 256`
- `chunk_overlap = 32`
- `embedding_dim = 384`
- `FAISS index type = IndexFlatL2`
- `n_probes = 10`

### Agent retrieval sizes

- `precision.retrieval_k = 3`
- `recall.retrieval_k = 5`
- `skeptic.retrieval_k = 4`
- `counterfactual.retrieval_k = 3`

### Full graph limits

- `max_nodes = 50`
- `max_pairs = 30`
- `max_pairs_cloud = 200`

### System constraints

- `max_concurrent_agents = 1`
- `max_retrieval_chunks = 15`
- `max_claims_per_query = 15`
- `max_claims_cloud = 60`

### Offline claim-graph values

- `max_claims_per_chunk = 3`
- `candidate_neighbors = 6`
- `query_seed_k = 8`
- `query_neighbor_expansion = 3`
- `max_query_edges = 40`
- `verify_edges = local`

These numbers are not just implementation details. They reflect the system’s central trade-off:

- keep online work small enough for local usability,
- but retain enough evidence structure to support multi-view answers.

---

## 12. Benchmark Summary to Preserve for the Final Report

The following results should be preserved almost verbatim because they are highly report-relevant.

### 12.1 Benchmark setup

A local 3-query benchmark was run against the project’s normal baseline, `vanilla_rag`, using:

- the same local corpus,
- the same local model stack,
- and the final fast claim-graph EVIRAG path.

### 12.2 Steady-state results after warmup

- **Fast EVIRAG:** `0.431s` average per query
- **Vanilla RAG:** `10.623s` average per query
- **Mean speedup:** `25.62x`

### 12.3 Per-query results

| Query | Fast EVIRAG | Vanilla RAG | Speedup |
|---|---:|---:|---:|
| Does homework improve academic achievement? | 0.553 s | 12.268 s | 22.18x |
| Is homework effective for science learning outcomes? | 0.411 s | 7.841 s | 19.08x |
| What are the trade-offs of homework for student achievement and attitudes? | 0.330 s | 11.761 s | 35.60x |

### 12.4 Behavioral interpretation of the benchmark

The speedup is important, but the behavioral differences are equally important:

1. **Fast mode returned multi-view answers**
   - It did not simply become a shallow summary engine.
   - It still surfaced multiple perspectives.

2. **Fast mode used cached graph reasoning with citations**
   - This matters because the speedup did not come from dropping evidence structure entirely.

3. **No fallback to the full pipeline was triggered on any of the 3 benchmarked queries**
   - That means the offline claim graph was sufficiently informative for those questions.

4. **Vanilla RAG remained a single-answer baseline**
   - It is useful as a speed and simplicity reference, but not as a disagreement-aware answering system.

### 12.5 One-time costs

- **Offline local-verified graph init/rebuild:** `157.9s`

This cost should be interpreted correctly:

- It is an **offline or infrequent preprocessing cost**.
- It is not the steady-state user-facing latency.
- Once the graph is built and warm, query latency becomes sub-second in Fast EVIRAG mode.

### 12.6 Why these benchmark results matter

These results support the project’s main systems claim:

> By moving disagreement structure construction offline, EVIRAG can preserve multi-view evidence-aware behavior while achieving practical local query latency.

That is the most report-worthy performance conclusion in the project.

---

## 13. Concise End Summary

EVIRAG is a disagreement-aware scientific literature QA system built over a focused homework-related research corpus.

In its final working version:

1. It indexes a corpus of **7 PDF files** totaling **2,872,027 bytes**.
2. It builds a chunk-level vector index with **277 chunks**.
3. It builds a claim-level offline evidence graph with **631 claims** and **385 cached relationships**.
4. It supports:
   - a baseline `vanilla_rag` mode,
   - a full multi-agent EVIRAG pipeline,
   - and a fast offline claim-graph EVIRAG path.
5. It includes:
   - citations,
   - confidence,
   - alternative and minority views,
   - claim graph inspection,
   - and optional research modules for divergence, causal disagreement attribution, and temporal drift.
6. It currently uses a **local-first architecture** centered on `qwen3:0.6b` via Ollama, plus `all-MiniLM-L6-v2` for text embeddings, while still retaining a cloud-mode model registry in the codebase.
7. Its strongest practical result is that the final fast path achieves **0.431 seconds average warmed query time**, versus **10.623 seconds** for the baseline `vanilla_rag`, for a mean **25.62x speedup**, while still returning structured multi-view answers.

The most important conceptual point is this:

EVIRAG is not merely a faster chatbot over PDFs. The final system is a structured evidence reasoning architecture that attempts to preserve disagreement, expose supporting evidence, and make local corpus chat both useful and fast enough to be usable.

