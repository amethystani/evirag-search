/* global React */
// ─────────────────────────────────────────────────────────────────────────────
// EVIRAG · api.jsx
// All backend communication + data transformation lives here.
// The UI files (results.jsx, home.jsx, …) import nothing — they read from
// window.* globals set at the bottom of this file.
// ─────────────────────────────────────────────────────────────────────────────

const BACKEND_URL = "http://localhost:8000";

function _parseYear(text) {
  const match = String(text || "").match(/\b(19|20)\d{2}\b/);
  return match ? Number(match[0]) : null;
}

// ── Stable pseudo-random from string (for deterministic graph layout) ─────────
function _hashStr(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  return Math.abs(h) / 2147483647;
}

// ── Graph node layout ─────────────────────────────────────────────────────────
// Produces x,y ∈ [0,1] for each node, clustered by view kind.
function _layoutNodes(nodes) {
  // Seed initial positions by kind
  const positioned = nodes.map((n) => {
    const sx = _hashStr(n.id + "x");
    const sy = _hashStr(n.id + "y");
    if (n.kind === "dom") {
      return { ...n, x: 0.08 + sx * 0.30, y: 0.12 + sy * 0.76 };
    } else if (n.kind === "alt") {
      return { ...n, x: 0.50 + sx * 0.28, y: 0.08 + sy * 0.50 };
    } else {
      return { ...n, x: 0.48 + sx * 0.30, y: 0.55 + sy * 0.35 };
    }
  });

  // Minimal repulsion so nodes don't stack
  const MIN_DIST = 0.14;
  for (let iter = 0; iter < 8; iter++) {
    for (let i = 0; i < positioned.length; i++) {
      for (let j = i + 1; j < positioned.length; j++) {
        const dx = positioned[i].x - positioned[j].x;
        const dy = positioned[i].y - positioned[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 0.001;
        if (dist < MIN_DIST) {
          const push = ((MIN_DIST - dist) / dist) * 0.4;
          positioned[i].x += dx * push;
          positioned[i].y += dy * push;
          positioned[j].x -= dx * push;
          positioned[j].y -= dy * push;
        }
      }
    }
    positioned.forEach((p) => {
      p.x = Math.max(0.04, Math.min(0.93, p.x));
      p.y = Math.max(0.05, Math.min(0.93, p.y));
    });
  }
  return positioned;
}

// ── Map backend relationship label → frontend edge kind ───────────────────────
function _rel(r) {
  if (r === "supports") return "support";
  if (r === "contradicts") return "contra";
  return "neutral";
}

// ── Map confidence score → controversy level string ───────────────────────────
function controversyLevel(edScore) {
  if (!edScore && edScore !== 0) return "med";
  if (edScore > 0.60) return "high";
  if (edScore > 0.30) return "med";
  return "low";
}

// ── Build calibration factors from available metrics ─────────────────────────
function _buildCalibration(answer, metrics) {
  const cr = metrics.conflict_ratio || 0;
  const vr = metrics.visual_text_mismatch || 0;
  const conf = answer.confidence_score || 0.5;
  return [
    ["Claim agreement",       Math.max(0.05, Math.min(0.98, 1 - cr * 0.7)),     "",        "How aligned the kept claims are with each other."],
    ["Source diversity",      Math.max(0.05, Math.min(0.98, conf * 0.95 + 0.1)),"accent",  "Spread of retrieved sources across venues and years."],
    ["Contradiction severity",Math.max(0.05, Math.min(0.98, cr)),                "contra",  "Severity of opposing claims (higher = more conflict)."],
    ["Unverified assump.",    Math.max(0.05, Math.min(0.98, 0.50 - cr * 0.3)),  "muted",   "Hypothesis assumptions not validated against evidence."],
    ["Visual alignment",      Math.max(0.05, Math.min(0.98, 1 - vr)),            "support", "CLIP-based agreement between figures and textual claims."]
  ];
}

// ── Transform a single view object (dom/alt/min) ──────────────────────────────
function _transformView(view, kind, claimLookup) {
  if (!view) return null;
  const citations = view.citations || [];
  const claimIds  = view.supporting_claim_ids || [];

  const claims = claimIds.slice(0, 5).map((cid, i) => {
    const cl  = claimLookup[cid] || {};
    const cit = citations[i]     || {};
    return {
      n:     "C-" + String(i + 1).padStart(2, "0"),
      text:  cl.text || cit.text || cid,
      src:   cl.citation_label || cit.citation_label || cit.source_doc_title || "–",
      cites: [],
      context: cl.context || "",
      claimId: cid,
      docId: cl.source_doc_id || cit.source_doc_id || "",
      sourcePath: cl.source_path || "",
      docTitle: cl.source_doc_title || cit.source_doc_title || ""
    };
  });

  const labelMap = { dom: "Dominant view", alt: "Alternative view", min: "Minority / contrarian" };
  return {
    kind,
    label:      labelMap[kind] || "View",
    sources:    view.source_count || 0,
    confidence: view.confidence  || 0.5,
    summary:    view.summary     || "",
    claims
  };
}

function transformVanillaResult(data, query) {
  const sources = (data.sources || []).map((s, i) => ({
    n: i + 1,
    title: s.title || "Unknown source",
    venue: "",
    year: _parseYear(s.title),
    stance: "neutral",
    snippet: s.text || "",
    docId: "",
    path: "",
    claimCount: 0
  }));

  return {
    query,
    directAnswer: typeof data.answer === "string" ? data.answer : "No answer returned.",
    intent: {
      dispute: "single_view_answer",
      domain: "retrieval",
      expectedDisagreement: "low"
    },
    metrics: {
      confidence: 0.5,
      confidenceLabel: "LOW",
      disagreementDensity: 0,
      conflictRatio: 0,
      claimEntropy: 0,
      visualMismatch: 0,
      claims: 0,
      sources: sources.length,
      figuresAligned: 0,
      controversyClass: "stable"
    },
    views: [
      {
        kind: "dom",
        label: "Retrieved answer",
        sources: sources.length,
        confidence: 0.5,
        summary: typeof data.answer === "string" ? data.answer : "No answer returned.",
        claims: []
      }
    ],
    agents: [],
    sources,
    graph: { nodes: [], edges: [] },
    calibration: [
      ["Claim agreement", 0.5, "", "Vanilla mode does not compute inter-claim agreement."],
      ["Source diversity", Math.min(0.95, 0.35 + sources.length * 0.08), "accent", "Breadth of retrieved documents."],
      ["Contradiction severity", 0.0, "contra", "Vanilla mode does not build a contradiction graph."],
      ["Unverified assump.", 0.5, "muted", "No explicit hypothesis verification is run."],
      ["Visual alignment", 0.0, "support", "Visual grounding is disabled in vanilla mode."]
    ],
    hypothesis: {
      centralHypothesis: query,
      supportingClaims: [],
      assumptions: [],
      expectedCounterclaims: []
    },
    epistemic: {
      edScore: 0,
      polarizationIndex: 0,
      consensusCollapseScore: 0,
      reasoning: "",
      controversyClass: "stable"
    },
    temporal: null,
    confidenceReasoning: "Vanilla retrieval returned a single synthesized answer without disagreement-aware decomposition.",
    visual: {
      enabled: false,
      mismatchScore: 0,
      weakSupportCount: 0,
      items: []
    },
    raw: data
  };
}

function buildDirectAnswerFallback(query, answer, metrics) {
  const dominant = (answer.dominant_view || {}).summary || "";
  if (!dominant) {
    return "The backend did not return enough corpus evidence to answer this query directly.";
  }
  const alternative = ((answer.alternative_views || [])[0] || {}).summary || ((answer.minority_views || [])[0] || {}).summary || "";
  const conflictRatio = metrics.conflict_ratio || 0;
  const confidence = answer.confidence_score || 0.5;
  const level = answer.overall_confidence || "medium";
  let text = `Best-supported answer from the retrieved corpus: ${dominant}`;
  const artifact = /doi:|\.docx|\.qxd|\.indd|microsoft word|access denied|copyright|downloaded from/i.test(alternative);
  const realCaveat = /however|but|although|negative|burden|harm|worse|reduce|denies|contradict|opposes|ineffective|mixed|limited|no evidence/i.test(alternative);
  if (alternative && conflictRatio >= 0.15 && !artifact && realCaveat) {
    text += ` Main caveat: ${alternative}`;
  } else if (conflictRatio >= 0.15) {
    text += " The claim graph still shows material disagreement, so treat this as a corpus-grounded position rather than a settled fact.";
  }
  return `${text} Confidence is ${level} (${confidence.toFixed(2)}).`;
}

// ── Main transform: backend JSON → UI result object ───────────────────────────
function transformResult(data, query) {
  if ((data.mode || "") === "vanilla_rag") {
    return transformVanillaResult(data, query);
  }

  const answer  = data.answer     || {};
  const metrics = data.metrics    || {};
  const stats   = data.statistics || {};
  const ed      = data.epistemic_divergence || {};

  // Claim lookup by claim_id
  const claimLookup = {};
  (data.claims || []).forEach((c) => { claimLookup[c.claim_id] = c; });

  // ── Intent ──────────────────────────────────────────────────────────────────
  const intentRaw = data.intent || {};
  const intent = {
    dispute:              intentRaw.factual_vs_disputed   || "somewhat_disputed",
    domain:               intentRaw.empirical_vs_theoretical || "empirical",
    expectedDisagreement: intentRaw.disagreement_level    || "medium"
  };

  // ── Metrics ─────────────────────────────────────────────────────────────────
  const metricsObj = {
    confidence:           answer.confidence_score || 0.5,
    confidenceLabel:      (answer.overall_confidence || "medium").toUpperCase(),
    disagreementDensity:  metrics.disagreement_density   || 0,
    conflictRatio:        metrics.conflict_ratio         || 0,
    claimEntropy:         metrics.claim_entropy          || 0,
    visualMismatch:       metrics.visual_text_mismatch   || 0,
    claims:               stats.total_claims || (data.claims || []).length,
    sources:              (data.sources || []).length,
    figuresAligned:       0,
    controversyClass:     ed.controversy_class || "stable"
  };

  // ── Views ────────────────────────────────────────────────────────────────────
  const views = [];
  const dom = _transformView(answer.dominant_view, "dom", claimLookup);
  if (dom) views.push(dom);
  (answer.alternative_views || []).slice(0, 2).forEach((v) => {
    const vv = _transformView(v, "alt", claimLookup);
    if (vv) views.push(vv);
  });
  (answer.minority_views || []).slice(0, 1).forEach((v) => {
    const vv = _transformView(v, "min", claimLookup);
    if (vv) views.push(vv);
  });

  // ── Graph ────────────────────────────────────────────────────────────────────
  const backGraph = data.graph || { nodes: [], edges: [] };

  // Determine kind per node from view membership
  const domIds = new Set((answer.dominant_view || {}).supporting_claim_ids || []);
  const altIds = new Set(
    (answer.alternative_views || []).flatMap((v) => v.supporting_claim_ids || [])
  );
  const minIds = new Set(
    (answer.minority_views || []).flatMap((v) => v.supporting_claim_ids || [])
  );

  // Map: backend node id → { frontId, label }
  const idMap    = {};
  const labelMap = {};
  backGraph.nodes.forEach((n, i) => {
    idMap[n.id]    = "n" + (i + 1);
    labelMap[n.id] = "C-" + String(i + 1).padStart(2, "0");
  });

  const rawNodes = backGraph.nodes.map((n, i) => ({
    id:        idMap[n.id],
    originalId: n.id,
    label:     labelMap[n.id],
    title:     n.text     || "",
    docTitle:  n.doc_title || "",
    context:   (claimLookup[n.id] || {}).context || "",
    docId:     (claimLookup[n.id] || {}).source_doc_id || "",
    sourcePath:(claimLookup[n.id] || {}).source_path || "",
    kind:      domIds.has(n.id) ? "dom" : altIds.has(n.id) ? "alt" : minIds.has(n.id) ? "min" : "min"
  }));

  const layoutNodes = _layoutNodes(rawNodes);

  const edges = backGraph.edges
    .filter((e) => idMap[e.source] && idMap[e.target])
    .map((e) => ({
      a:          idMap[e.source],
      b:          idMap[e.target],
      kind:       _rel(e.relationship),
      confidence: e.confidence || 0.5,
      // keep originals for modal detail
      srcLabel:   labelMap[e.source] || "",
      tgtLabel:   labelMap[e.target] || "",
      srcText:    claimLookup[e.source] ? claimLookup[e.source].text : "",
      tgtText:    claimLookup[e.target] ? claimLookup[e.target].text : ""
    }));

  // ── Agents ───────────────────────────────────────────────────────────────────
  const agentsUsed = stats.agents_used || 0;
  const traceModels = (((data.trace || {}).models || {}).query_time_models || {});
  const agentModelMap = traceModels.agents || {};
  const agentDefs = [
    { key: "precision", glyph: "P", name: "Precision",      role: "Strict semantic · high-confidence support",  model: "unknown", k: 3, retr: "cosine ≥ 0.78" },
    { key: "recall", glyph: "R", name: "Recall",            role: "Expansive semantic · broad coverage",         model: "unknown", k: 5, retr: "MMR λ=0.3"      },
    { key: "skeptic", glyph: "S", name: "Skeptic",          role: "Adversarial · contradiction-seeking",         model: "unknown", k: 4, retr: "neg-augmented"   },
    { key: "counterfactual", glyph: "C", name: "Counterfactual", role: "Alternative explanations · what-if reasoning", model: "unknown", k: 3, retr: "hyp. perturb" }
  ];
  const agentDefMap = Object.fromEntries(agentDefs.map((a) => [a.key, a]));
  const agentDetails = (data.agent_details || ((data.agent_outputs || {}).agents) || []);
  const agents = agentDetails.map((detail) => {
    const key = detail.agent_name || detail.key || "";
    const def = agentDefMap[key] || {
      key,
      glyph: key ? key[0].toUpperCase() : "A",
      name: key ? key.replace(/_/g, " ") : "Agent",
      role: "Deliberative retrieval agent",
      model: "unknown",
      k: 0,
      retr: "backend"
    };
    return {
      ...def,
      model: agentModelMap[key] || def.model,
      retrieved: detail.num_chunks || 0,
      kept: detail.num_claims || 0,
      stance: detail.stance || "neutral",
      confidence: detail.confidence || 0,
      reasoning: detail.reasoning || "",
      status: detail.status || "ok",
      durationMs: detail.duration_ms || 0,
      retrievalQuery: detail.retrieval_query || "",
      parallel: !!((data.agent_outputs || {}).parallel)
    };
  });
  const agentSummary = data.agent_outputs || {};

  // ── Sources ──────────────────────────────────────────────────────────────────
  const claimKinds = {};
  domIds.forEach((id) => { claimKinds[id] = "support"; });
  altIds.forEach((id) => { claimKinds[id] = "contra"; });
  minIds.forEach((id) => { claimKinds[id] = claimKinds[id] || "contra"; });

  const sourceMap = new Map();
  (data.claims || []).forEach((claim) => {
    const key = claim.source_doc_id || claim.source_doc_title || claim.claim_id;
    if (!sourceMap.has(key)) {
      sourceMap.set(key, {
        title: claim.source_doc_title || "Unknown source",
        venue: claim.section ? String(claim.section).replace(/_/g, " ") : "",
        year: _parseYear(claim.source_doc_title),
        path: claim.source_path || "",
        docId: claim.source_doc_id || "",
        snippet: claim.text || "",
        claimCount: 0,
        supportCount: 0,
        contraCount: 0
      });
    }
    const entry = sourceMap.get(key);
    entry.claimCount += 1;
    if (!entry.snippet && claim.text) entry.snippet = claim.text;
    if (claimKinds[claim.claim_id] === "support") entry.supportCount += 1;
    if (claimKinds[claim.claim_id] === "contra") entry.contraCount += 1;
  });

  (data.sources || []).forEach((source, i) => {
    const key = source.doc_id || source.title || source.doc_title || String(i);
    if (!sourceMap.has(key)) {
      sourceMap.set(key, {
        title: source.doc_title || source.title || "Unknown source",
        venue: source.section || "",
        year: source.year || _parseYear(source.doc_title || source.title),
        path: source.path || "",
        docId: source.doc_id || source.source_doc_id || "",
        snippet: source.text || "",
        claimCount: 0,
        supportCount: 0,
        contraCount: 0
      });
    }
  });

  const sources = Array.from(sourceMap.values())
    .sort((a, b) => b.claimCount - a.claimCount || a.title.localeCompare(b.title))
    .map((source, i) => ({
      n: i + 1,
      title: source.title,
      venue: source.venue,
      year: source.year,
      stance: source.supportCount > source.contraCount ? "support" : source.contraCount > source.supportCount ? "contra" : "neutral",
      snippet: source.snippet ? source.snippet.slice(0, 220) + (source.snippet.length > 220 ? "…" : "") : "",
      docId: source.docId,
      path: source.path,
      claimCount: source.claimCount
    }));

  // ── Calibration sidebar factors ───────────────────────────────────────────────
  const calibration = _buildCalibration(answer, metrics);

  // ── Hypothesis ────────────────────────────────────────────────────────────────
  const hyp = data.hypothesis || {};
  const hypothesis = {
    centralHypothesis:    hyp.central_hypothesis    || query,
    supportingClaims:     hyp.supporting_claims     || [],
    assumptions:          hyp.assumptions           || [],
    expectedCounterclaims:hyp.expected_counterclaims|| []
  };

  // ── Epistemic divergence extras ───────────────────────────────────────────────
  const epistemic = {
    edScore:                ed.ed_score                 || 0,
    polarizationIndex:      ed.polarization_index       || 0,
    consensusCollapseScore: ed.consensus_collapse_score || 0,
    reasoning:              ed.reasoning                || "",
    controversyClass:       ed.controversy_class        || "stable"
  };

  // ── Temporal drift ────────────────────────────────────────────────────────────
  const temporal = data.temporal_drift || {};

  // ── Confidence reasoning ──────────────────────────────────────────────────────
  const confidenceReasoning = answer.confidence_reasoning || "";
  const directAnswer = answer.direct_answer || buildDirectAnswerFallback(query, answer, metrics);

  // ── Visual analysis ───────────────────────────────────────────────────────
  const visualAnalysis = data.visual_analysis || {};
  const alignedFigures = visualAnalysis.aligned_figures || [];
  const crossComparisons = visualAnalysis.cross_comparisons || [];
  const visual = {
    enabled: !!stats.visual_grounding,
    mismatchScore: visualAnalysis.mismatch_score || 0,
    weakSupportCount: visualAnalysis.claims_with_weak_support || 0,
    totalIndexedFigures: visualAnalysis.total_indexed_figures || 0,
    items: alignedFigures.slice(0, 12).map((af, i) => ({
      i: i + 1,
      figureId: af.figure_id,
      claimId: af.claim_id,
      claimText: af.claim_text || "",
      paper: af.doc_id || "—",
      align: (af.alignment_score || 0).toFixed(3),
      alignmentScore: af.alignment_score || 0,
      caption: af.caption || "No caption",
      pageNum: af.page_num,
      width: af.width || 0,
      height: af.height || 0,
      imageUrl: af.figure_id ? `${BACKEND_URL}/api/visual/figure/${af.figure_id}` : null,
    })),
    crossComparisons: crossComparisons.slice(0, 6).map((cc) => ({
      claim1Id: cc.claim1_id,
      claim2Id: cc.claim2_id,
      interpretationConflict: cc.interpretation_conflict,
      similarEvidenceCount: cc.similar_evidence_count || 0,
    })),
    mismatches: (visualAnalysis.mismatches || []).slice(0, 6).map((mismatch, i) => ({
      i: i + 1,
      paper: mismatch.doc_id || "—",
      align: Math.max(0, 1 - (mismatch.mismatch_severity || 0)).toFixed(2),
      caption: mismatch.claim,
      claim: mismatch.claim,
      severity: mismatch.mismatch_severity || 0,
    })),
  };

  metricsObj.sources = sources.length;
  metricsObj.figuresAligned = visual.enabled ? visual.items.length : 0;

  return {
    query,
    directAnswer,
    intent,
    metrics:   metricsObj,
    views,
    agents,
    agentSummary,
    sources,
    graph:     { nodes: layoutNodes, edges },
    calibration,
    hypothesis,
    epistemic,
    temporal,
    confidenceReasoning,
    visual,
    raw: data
  };
}

// ── API: submit a query ────────────────────────────────────────────────────────
function buildQueryConfig(options = {}) {
  const mode = (options.mode === "Vanilla") ? "vanilla_rag" : "evirag";
  const wantsVlm    = !!options.vlm;
  const wantsAgents = !!options.agents;
  const wantsFullPipeline = mode === "evirag" && wantsAgents;
  return {
    mode,
    backend:                  "local",
    depth_vs_speed:           wantsFullPipeline ? "balanced" : "fast",
    auto_fallback_to_full:    wantsFullPipeline,
    enabled_agents:           wantsFullPipeline ? ["precision", "recall", "skeptic", "counterfactual"] : [],
    use_epistemic_divergence: true,
    use_causal_attribution:   false,
    use_temporal_tracking:    true,
    use_visual_grounding:     wantsVlm,
    rebuild_index:            false,
  };
}

async function queryBackend(query, options = {}) {
  const config = buildQueryConfig(options);
  const resp = await fetch(`${BACKEND_URL}/api/process_query`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ query, config })
  });
  if (!resp.ok) {
    const txt = await resp.text().catch(() => "");
    throw new Error(`Backend ${resp.status}: ${txt.slice(0, 200)}`);
  }
  return resp.json();
}

async function fetchQueryTracePlan(query, options = {}) {
  const resp = await fetch(`${BACKEND_URL}/api/query_trace_plan`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ query, config: buildQueryConfig(options) })
  });
  if (!resp.ok) {
    const txt = await resp.text().catch(() => "");
    throw new Error(`Backend ${resp.status}: ${txt.slice(0, 200)}`);
  }
  return resp.json();
}

// ── API: corpus stats ─────────────────────────────────────────────────────────
async function fetchCorpusStats() {
  try {
    const r = await fetch(`${BACKEND_URL}/api/corpus/stats?backend=local`);
    if (!r.ok) return null;
    return r.json();
  } catch { return null; }
}

// ── API: health ───────────────────────────────────────────────────────────────
async function fetchHealth() {
  try {
    const ctrl = new AbortController();
    setTimeout(() => ctrl.abort(), 3000);
    const r = await fetch(`${BACKEND_URL}/health`, { signal: ctrl.signal });
    if (!r.ok) return { status: "unreachable" };
    return r.json();
  } catch { return { status: "unreachable" }; }
}

async function fetchJson(path) {
  const r = await fetch(`${BACKEND_URL}${path}`);
  if (!r.ok) {
    const txt = await r.text().catch(() => "");
    throw new Error(`Backend ${r.status}: ${txt.slice(0, 200)}`);
  }
  return r.json();
}

async function fetchUIBootstrap() {
  return fetchJson("/api/ui/bootstrap?backend=local");
}

async function fetchCorpusDocuments() {
  return fetchJson("/api/corpus/documents?backend=local");
}

async function fetchGraphOverview() {
  return fetchJson("/api/graph/overview?backend=local");
}

async function fetchAgentsStatus() {
  return fetchJson("/api/agents/status?backend=local");
}

// ── Expose to window (consumed by all JSX files) ─────────────────────────────
window.BACKEND_URL         = BACKEND_URL;
window.queryBackend        = queryBackend;
window.fetchQueryTracePlan = fetchQueryTracePlan;
window.transformResult     = transformResult;
window.fetchCorpusStats    = fetchCorpusStats;
window.fetchHealth         = fetchHealth;
window.fetchUIBootstrap    = fetchUIBootstrap;
window.fetchCorpusDocuments= fetchCorpusDocuments;
window.fetchGraphOverview  = fetchGraphOverview;
window.fetchAgentsStatus   = fetchAgentsStatus;
window.controversyLevel    = controversyLevel;
