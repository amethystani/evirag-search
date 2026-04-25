/* global React, Icon */
const { useState, useMemo, useEffect } = React;

const requestLabels = (options = {}) => {
  const mode = String(options.mode || "EVIRAG");
  if (mode.toLowerCase() === "vanilla") {
    return { mode: "Vanilla RAG", path: "corpus retrieval" };
  }
  if (options.agents || options.vlm) {
    return { mode: "EVIRAG", path: options.vlm ? "full graph + visual" : "full graph + agents" };
  }
  return { mode: "EVIRAG", path: "fast claim graph" };
};

const normalizeTraceSteps = (tracePlan) => {
  if (!tracePlan || !Array.isArray(tracePlan.steps) || !tracePlan.steps.length) {
    return null;
  }
  return tracePlan.steps.map((item) => [
    item.label || item.step || "Backend step",
    item.detail || item.description || "Running backend pipeline step."
  ]);
};

const ConfidenceBar = ({ value }) => (
  <div className="bar"><span style={{ width: (value * 100).toFixed(0) + "%" }}/></div>
);

const ViewBlock = ({ v, idx, onClaim }) => (
  <div className={"view-block " + v.kind + " fade-up"} style={{ animationDelay: (0.05 + idx * 0.06) + "s" }}>
    <div className="stripe"/>
    <div className="view-head">
      <div className="view-tag">
        <span className="swatch"/>
        {v.label} · {v.sources} sources
      </div>
      <div className="view-stats">
        <span className="conf">
          <span>conf</span>
          <ConfidenceBar value={v.confidence}/>
          <strong style={{ color: "var(--ink)" }}>{v.confidence.toFixed(2)}</strong>
        </span>
      </div>
    </div>
    <p className="view-summary">{v.summary}</p>
    <div className="view-claims">
      {v.claims.map((c, i) => (
        <div key={i} className="claim-row" onClick={() => onClaim({ ...c, view: v.label, kind: v.kind })} style={{ gridTemplateColumns: "28px minmax(0, 1fr) minmax(100px, 30%)", alignItems: "start" }}>
          <div className="num">{c.n}</div>
          <div style={{ minWidth: 0, paddingRight: 10 }}>{c.text}</div>
          <div className="src" style={{ display: "flex", flexWrap: "nowrap", overflow: "hidden", alignItems: "center", justifyContent: "flex-end" }}>
            <span style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={c.src}>{c.src}</span>
            {c.cites.map(s => <span key={s} className="cite" style={{ flexShrink: 0 }} onClick={(e) => { e.stopPropagation(); onClaim({ ...c, view: v.label, kind: v.kind }); }}>{s}</span>)}
          </div>
        </div>
      ))}
    </div>
  </div>
);

const DisagreementGraph = ({ graph, onNode }) => {
  const W = 640, H = 320, pad = 28;
  const pos = (n) => ({ x: pad + n.x * (W - pad * 2), y: pad + n.y * (H - pad * 2) });
  const nodeMap = useMemo(() => Object.fromEntries(graph.nodes.map(n => [n.id, n])), [graph]);
  const edgeStyle = (k) => {
    if (k === "support") return { stroke: "oklch(0.50 0.07 150)", strokeWidth: 1.4, strokeDasharray: "" };
    if (k === "contra") return { stroke: "oklch(0.50 0.13 30)", strokeWidth: 1.4, strokeDasharray: "4 4" };
    return { stroke: "var(--muted-2)", strokeWidth: 1, strokeDasharray: "1 4" };
  };
  return (
    <div className="graph-wrap fade-up d2">
      <div className="graph-head">
        <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--muted)", fontWeight: 600 }}>Disagreement graph</div>
        <div className="legend">
          <span className="leg-item"><span className="ln support"/> support</span>
          <span className="leg-item"><span className="ln contra"/> contradict</span>
          <span className="leg-item"><span className="ln neutral"/> neutral</span>
        </div>
      </div>
      <svg className="graph-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
        <defs>
          <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
            <path d="M32 0H0V32" fill="none" stroke="var(--hair)" strokeWidth="0.5"/>
          </pattern>
        </defs>
        <rect x="0" y="0" width={W} height={H} fill="url(#grid)" opacity="0.5"/>
        {graph.edges.map((e, i) => {
          const a = pos(nodeMap[e.a]); const b = pos(nodeMap[e.b]);
          const st = edgeStyle(e.kind);
          return <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} {...st} />;
        })}
        {graph.nodes.map(n => {
          const p = pos(n);
          return (
            <g key={n.id} className={"node " + n.kind} transform={`translate(${p.x},${p.y})`} style={{ cursor: "pointer" }} onClick={() => onNode(n)}>
              <circle r="20"/>
              <text textAnchor="middle" dy="3.5" fontSize="10">{n.label}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
};

const Results = ({ result, loading, error, pendingQuery, pendingOptions, tracePlan, onRetry, onBack, onOpenSource, onOpenClaim, onOpenAgent, onOpenFigure, onOpenMetric, onOpenHypothesis }) => {
  const [tab, setTab] = useState("answer");
  const [loadingStep, setLoadingStep] = useState(0);
  const pendingMode = pendingOptions && pendingOptions.mode;
  const pendingAgents = !!(pendingOptions && pendingOptions.agents);
  const pendingVlm = !!(pendingOptions && pendingOptions.vlm);
  const loadingOptions = { mode: pendingMode, agents: pendingAgents, vlm: pendingVlm };
  const backendSteps = normalizeTraceSteps(tracePlan);
  const loadingSteps = backendSteps || [
    ["Initializing pipeline execution", "Analyzing epistemic intent and routing query through the active retrieval topology."]
  ];
  const activeLoadingStep = Math.min(loadingStep, loadingSteps.length - 1);
  const fallbackLabels = requestLabels(loadingOptions);
  const labels = tracePlan ? {
    mode: tracePlan.mode_label || fallbackLabels.mode,
    path: tracePlan.path_label || fallbackLabels.path
  } : fallbackLabels;
  const tracePlanKey = tracePlan && Array.isArray(tracePlan.steps)
    ? tracePlan.steps.map((item) => item.step || item.label).join("|")
    : "";

  useEffect(() => {
    if (!loading) {
      setLoadingStep(0);
      return undefined;
    }

    setLoadingStep(0);
    const stepCount = (
      tracePlan && Array.isArray(tracePlan.steps) && tracePlan.steps.length
        ? tracePlan.steps.length
        : 1
    );
    const id = window.setInterval(() => {
      setLoadingStep((step) => Math.min(step + 1, stepCount - 1));
    }, 1300);
    return () => window.clearInterval(id);
  }, [loading, pendingQuery, pendingMode, pendingAgents, pendingVlm, tracePlanKey]);

  const r = result;
  if (loading) {
    return (
      <div className="results">
        <div className="result-main">
          <div className="q-line fade-up">
            <button className="back" onClick={onBack} title="Back"><Icon name="arrow-left" size={16}/></button>
            <div style={{ flex: 1 }}>
              <div className="q-text serif">{pendingQuery || "EVIRAG inquiry"}</div>
              <div className="q-meta">
                <span className="pill"><span className="lbl">status</span> reasoning</span>
                <span className="pill"><span className="lbl">mode</span> {labels.mode}</span>
                <span className="pill"><span className="lbl">path</span> {labels.path}</span>
                {pendingVlm && <span className="pill"><span className="lbl">visual</span> requested</span>}
              </div>
            </div>
          </div>

          <div className="card fade-up reasoning-card">
            <div className="reason-title">
              <h3>Reasoning through corpus evidence <span className="small">{activeLoadingStep + 1}/{loadingSteps.length}</span></h3>
            </div>
            <div className="reason-pulse"><span/></div>
            <p className="reason-lede">{loadingSteps[activeLoadingStep][1]}</p>
            <div className="reason-steps" aria-label="Backend pipeline trace">
              {loadingSteps.map(([label, detail], i) => (
                <div
                  key={label}
                  className={"reason-step " + (i < activeLoadingStep ? "is-done" : i === activeLoadingStep ? "is-active" : "is-pending")}
                >
                  <span className="reason-dot"/>
                  <div>
                    <div className="reason-label">{label}</div>
                    <div className="reason-detail">{detail}</div>
                  </div>
                </div>
              ))}
            </div>
            <div className="reason-note">
              <Icon name="library" size={14}/>
              <span>
                {tracePlan
                  ? `Plan ready: ${tracePlan.corpus ? `${tracePlan.corpus.total_documents} documents, ${tracePlan.corpus.total_claims} claims, ${tracePlan.corpus.total_claim_edges} graph edges` : "corpus metadata attached"}.`
                  : "Determining optimal retrieval path and index routing."}
              </span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error || !r) {
    return (
      <div className="results">
        <div className="result-main">
          <div className="q-line fade-up">
            <button className="back" onClick={onBack} title="Back"><Icon name="arrow-left" size={16}/></button>
            <div style={{ flex: 1 }}>
              <div className="q-text serif">{pendingQuery || "EVIRAG inquiry"}</div>
              <div className="q-meta">
                <span className="pill"><span className="lbl">status</span> failed</span>
              </div>
            </div>
          </div>
          <div className="card fade-up">
            <h3>Backend request failed</h3>
            <p>{error || "No result was returned."}</p>
            <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
              <button className="chip" onClick={onRetry}>Retry</button>
              <button className="chip" onClick={onBack}>Back</button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const supportEdges = r.graph.edges.filter(e => e.kind === "support").length;
  const contraEdges = r.graph.edges.filter(e => e.kind === "contra").length;
  const neutralEdges = r.graph.edges.filter(e => e.kind === "neutral").length;
  const total = r.graph.edges.length;
  const calibration = r.calibration || [];
  const visibleClaims = r.views.flatMap(v => v.claims.map(c => ({ ...c, view: v.label, kind: v.kind })));
  const disputedLabel = String(r.intent.dispute || "contested").replace(/_/g, " ");

  return (
    <div className="results">
      <div className="result-main">
        <div className="q-line fade-up">
          <button className="back" onClick={onBack} title="Back"><Icon name="arrow-left" size={16}/></button>
          <div style={{ flex: 1 }}>
            <div className="q-text serif">{r.query}</div>
            <div className="q-meta">
              <span className="pill warn" onClick={onOpenHypothesis}><Icon name="warn" size={11}/> {r.intent.dispute.replace("_", " ")}</span>
              <span className="pill" onClick={() => onOpenMetric({ label: "Domain classification", value: r.intent.domain, formula: "Topic-classifier (phi3:mini)", desc: "Domain inferred from query embedding and corpus distribution.", interp: "Used to bias retrieval toward domain-specific corpora." })}><span className="lbl">domain</span> {r.intent.domain}</span>
              <span className="pill" onClick={() => setTab("claims")}><span className="lbl">claims</span> {r.metrics.claims}</span>
              <span className="pill" onClick={() => setTab("sources")}><span className="lbl">sources</span> {r.metrics.sources}</span>
              <span className="pill" onClick={() => onOpenMetric({ label: "Confidence", value: r.metrics.confidence + " (" + r.metrics.confidenceLabel + ")", formula: "0.30·agree + 0.20·div + 0.25·(1−contra) + 0.15·(1−unver) + 0.10·visual", desc: "Calibrated confidence combining five orthogonal evidence factors.", interp: "Below 0.50 means the system recommends suspending judgement." })}><span className="lbl">conf</span> {r.metrics.confidenceLabel} · {r.metrics.confidence.toFixed(2)}</span>
              <span className="pill" onClick={() => onOpenMetric({ label: "Disagreement density", value: (r.metrics.disagreementDensity*100).toFixed(1)+"%", formula: "|contradict_edges| / |edges|", desc: "Fraction of claim-pair edges classified as contradicting.", interp: ">15% indicates active scientific controversy." })}><span className="lbl">density</span> {(r.metrics.disagreementDensity*100).toFixed(1)}%</span>
            </div>
          </div>
          <button className="icon-btn" title="Share"><Icon name="share" size={15}/></button>
          <button className="icon-btn" title="Copy"><Icon name="copy" size={15}/></button>
        </div>

        <div className="tabbar">
          <button className={tab==="answer" ? "is-active" : ""} onClick={() => setTab("answer")}>
            <span className="ico"><Icon name="sparkle" size={13}/></span> Multi-view answer
          </button>
          <button className={tab==="graph" ? "is-active" : ""} onClick={() => setTab("graph")}>
            <span className="ico"><Icon name="graph" size={13}/></span> Graph <span className="count">{r.graph.nodes.length}</span>
          </button>
          <button className={tab==="claims" ? "is-active" : ""} onClick={() => setTab("claims")}>
            <span className="ico"><Icon name="doc" size={13}/></span> Claims <span className="count">{r.metrics.claims}</span>
          </button>
          <button className={tab==="visual" ? "is-active" : ""} onClick={() => setTab("visual")}>
            <span className="ico"><Icon name="beaker" size={13}/></span> Visual <span className="count">{r.metrics.figuresAligned}</span>
          </button>
          <button className={tab==="sources" ? "is-active" : ""} onClick={() => setTab("sources")}>
            <span className="ico"><Icon name="library" size={13}/></span> Sources <span className="count">{r.metrics.sources}</span>
          </button>
        </div>

        {tab === "answer" && (
          <div>
            <div className="card fade-up direct-answer">
              <h3>Answer <span className="small">from retrieved corpus evidence</span></h3>
              <p>{r.directAnswer}</p>
            </div>
            <div className="callout fade-up" onClick={onOpenHypothesis} style={{ cursor: "pointer" }}>
              <div className="ico"><Icon name="warn" size={18}/></div>
              <div>
                <h4>This question is <em>{disputedLabel}</em>. EVIRAG answers first, then shows where the corpus disagrees.</h4>
                <p>{r.confidenceReasoning || `Synthesis retained ${r.views.length} positions from ${r.metrics.claims} extracted claims. Tap to inspect the X-IR hypothesis trace.`}</p>
              </div>
            </div>
            {r.views.map((v, i) => <ViewBlock key={i} v={v} idx={i} onClaim={onOpenClaim}/>)}
          </div>
        )}

        {tab === "graph" && (
          <div>
            <DisagreementGraph graph={r.graph} onNode={(n) => onOpenClaim({
              n: n.label,
              text: n.title || ("Claim " + n.label),
              src: n.docTitle || "graph",
              cites: [],
              view: n.kind === "dom" ? "Dominant" : n.kind === "alt" ? "Alternative" : "Minority",
              kind: n.kind,
              context: n.context || "",
              claimId: n.originalId || "",
              docId: n.docId || "",
              sourcePath: n.sourcePath || "",
              docTitle: n.docTitle || ""
            })}/>
            <div className="card fade-up d3">
              <h3>Edge composition <span className="small">{total} edges</span></h3>
              <div className="bar-row" onClick={() => onOpenMetric({ label: "Support edges", value: supportEdges, formula: "NLI ENTAILS", desc: "Pairs of claims where one entails the other.", interp: "Higher = consensus or redundancy." })} style={{ cursor: "pointer" }}>
                <span className="name">Support</span>
                <div className="track"><div className="fill support" style={{ width: (total ? supportEdges / total * 100 : 0) + "%" }}/></div>
                <span className="num">{supportEdges}</span>
              </div>
              <div className="bar-row" onClick={() => onOpenMetric({ label: "Contradict edges", value: contraEdges, formula: "NLI CONTRADICTS", desc: "Pairs of claims in direct logical opposition.", interp: "The signal EVIRAG cares about most." })} style={{ cursor: "pointer" }}>
                <span className="name">Contradict</span>
                <div className="track"><div className="fill contra" style={{ width: (total ? contraEdges / total * 100 : 0) + "%" }}/></div>
                <span className="num">{contraEdges}</span>
              </div>
              <div className="bar-row" onClick={() => onOpenMetric({ label: "Neutral edges", value: neutralEdges, formula: "NLI NEUTRAL", desc: "Claim pairs that neither support nor contradict.", interp: "High share suggests semantic drift between subdomains." })} style={{ cursor: "pointer" }}>
                <span className="name">Neutral</span>
                <div className="track"><div className="fill neutral" style={{ width: (total ? neutralEdges / total * 100 : 0) + "%" }}/></div>
                <span className="num">{neutralEdges}</span>
              </div>
            </div>
          </div>
        )}

        {tab === "claims" && (
          <div className="card fade-up">
            <h3>Atomic claims <span className="small">{r.metrics.claims} extracted · {visibleClaims.length} shown</span></h3>
            {visibleClaims.map((c, i) => (
              <div key={i} className="claim-row" onClick={() => onOpenClaim(c)} style={{ gridTemplateColumns: "32px minmax(0, 1fr) minmax(100px, 30%)", padding: "10px 0", gap: "12px", alignItems: "start" }}>
                <div className="num">{c.n}</div>
                <div style={{ minWidth: 0 }}>
                  <div style={{ color: "var(--ink)" }}>{c.text}</div>
                  <div style={{ fontSize: 10.5, color: "var(--muted-2)", marginTop: 4, fontFamily: "JetBrains Mono, monospace" }}>
                    view: {c.view.toLowerCase()} · cited in {(c.cites && c.cites.length) ? c.cites.join(", ") : "—"}
                  </div>
                </div>
                <div className="src" style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", textAlign: "right" }} title={c.src}>{c.src}</div>
              </div>
            ))}
          </div>
        )}

        {tab === "visual" && (
          <div>
            <div className="card fade-up">
              <h3>Visual evidence (CLIP ViT-B/16) <span className="small">{r.visual.enabled ? `${r.visual.items.length} aligned · mismatch ${r.visual.mismatchScore.toFixed(2)} · ${r.visual.totalIndexedFigures || 0} indexed figures` : "disabled"}</span></h3>
              {r.visual.enabled && r.visual.items.length > 0 ? (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 14, marginTop: 10 }}>
                  {r.visual.items.map((item) => (
                    <button key={item.i} onClick={() => onOpenFigure(item)} style={{ background: "var(--bg)", border: "1px solid var(--hair)", borderRadius: 10, overflow: "hidden", textAlign: "left", cursor: "pointer", padding: 0, transition: "border-color 0.15s" }}
                      onMouseOver={(e) => e.currentTarget.style.borderColor = "var(--accent)"}
                      onMouseOut={(e) => e.currentTarget.style.borderColor = "var(--hair)"}
                    >
                      <div style={{ height: 140, background: "var(--panel)", display: "grid", placeItems: "center", overflow: "hidden", position: "relative" }}>
                        {item.imageUrl ? (
                          <img src={item.imageUrl} alt={item.caption} style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }} onError={(e) => { e.target.style.display = "none"; e.target.nextSibling && (e.target.nextSibling.style.display = "block"); }}/>
                        ) : null}
                        <span style={{ display: item.imageUrl ? "none" : "block", fontFamily: "JetBrains Mono, monospace", fontSize: 10, color: "var(--muted)", padding: 12, textAlign: "center" }}>Figure {item.figureId || item.i}</span>
                        <div style={{ position: "absolute", top: 6, right: 6, background: item.alignmentScore > 0.3 ? "oklch(0.35 0.08 150 / 0.9)" : item.alignmentScore > 0.2 ? "oklch(0.35 0.08 60 / 0.9)" : "oklch(0.30 0.08 30 / 0.9)", color: "#fff", borderRadius: 6, padding: "2px 7px", fontSize: 10.5, fontFamily: "JetBrains Mono, monospace", fontWeight: 600 }}>
                          {item.align}
                        </div>
                      </div>
                      <div style={{ padding: "9px 11px", fontSize: 11.5, color: "var(--ink-2)", borderTop: "1px solid var(--hair)" }}>
                        <div style={{ fontWeight: 500, color: "var(--ink)", marginBottom: 3, lineHeight: 1.35, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>{item.claimText || item.caption}</div>
                        <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 10, color: "var(--muted)", marginTop: 4, display: "flex", gap: 8, flexWrap: "wrap" }}>
                          <span>p.{(item.pageNum || 0) + 1}</span>
                          <span>{item.width}×{item.height}px</span>
                          <span>doc {(item.paper || "").slice(0, 8)}</span>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <div style={{ marginTop: 8, color: "var(--muted)" }}>
                  {r.visual.enabled
                    ? "No visual evidence was aligned for this query. Try enabling multi-agent mode for richer claim extraction."
                    : "Visual grounding was disabled for this run. Enable the VLM toggle in the sidebar to activate CLIP-based figure analysis."}
                </div>
              )}
            </div>
            {r.visual.enabled && (r.visual.crossComparisons || []).length > 0 && (
              <div className="card fade-up d2" style={{ marginTop: 12 }}>
                <h3>Cross-paper figure comparisons <span className="small">{r.visual.crossComparisons.length} pairs</span></h3>
                <p style={{ fontSize: 12, color: "var(--muted)", marginBottom: 8 }}>When two papers contradict, CLIP compares their figures to determine if the disagreement is about interpretation (similar figures) or methodology (different figures).</p>
                {r.visual.crossComparisons.map((cc, i) => (
                  <div key={i} style={{ padding: "8px 0", borderBottom: "1px solid var(--hair)", display: "flex", gap: 10, alignItems: "center", fontSize: 12 }}>
                    <div style={{ width: 8, height: 8, borderRadius: "50%", background: cc.interpretationConflict ? "oklch(0.55 0.18 30)" : "oklch(0.55 0.10 150)", flexShrink: 0 }}/>
                    <div style={{ flex: 1 }}>
                      <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 10.5, color: "var(--ink-2)" }}>
                        {cc.claim1Id.slice(0, 8)} ↔ {cc.claim2Id.slice(0, 8)}
                      </span>
                      <span style={{ marginLeft: 8, color: cc.interpretationConflict ? "oklch(0.55 0.18 30)" : "oklch(0.55 0.10 150)", fontWeight: 500 }}>
                        {cc.interpretationConflict
                          ? `Interpretation conflict (${cc.similarEvidenceCount} similar figures)`
                          : "Methodological difference (distinct figures)"}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {tab === "sources" && (
          <div className="card fade-up">
            <h3>Sources <span className="small">{r.metrics.sources} retrieved · {r.sources.length} shown</span></h3>
            {r.sources.map(s => (
              <button key={s.n} className="src-card" onClick={() => onOpenSource(s)}>
                <div className="num">{s.n}</div>
                <div>
                  <div className="src-title">{s.title}</div>
                  <div className="src-meta">
                    <span>{s.venue || s.docId || "source document"}</span>
                    {s.year ? <span>· {s.year}</span> : null}
                    <span className={"stance " + s.stance}>{s.stance}</span>
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      <aside className="result-side">
        <div className="card fade-up d1">
          <h3>Confidence calibration <span className="small">{r.metrics.confidenceLabel}</span></h3>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 10 }}>
            <div className="serif" style={{ fontSize: 38, lineHeight: 1, letterSpacing: "-0.02em" }}>{r.metrics.confidence.toFixed(2)}</div>
            <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 11, color: "var(--muted)" }}>/ 1.00</div>
          </div>
          {calibration.map(([k, v, c, desc], i) => (
            <div key={i} className="bar-row" onClick={() => onOpenMetric({ label: k, value: v, formula: "weighted factor", desc, interp: "Contributes to overall confidence." })} style={{ cursor: "pointer" }}>
              <span className="name">{k}</span>
              <div className="track"><div className={"fill " + (c || "")} style={{ width: (v*100)+"%", background: c === "muted" ? "var(--muted-2)" : undefined }}/></div>
              <span className="num">.{(v*100).toFixed(0)}</span>
            </div>
          ))}
        </div>

        <div className="card fade-up d2">
          <h3>
            Multi-agent retrieval
            <span className="small">
              {r.agentSummary && r.agentSummary.parallel
                ? `parallel · median ${(r.agentSummary.median_confidence || 0).toFixed(2)}`
                : "deliberative"}
            </span>
          </h3>
          {r.agents.length ? r.agents.map((a, i) => (
            <div key={i} className="agent" onClick={() => onOpenAgent(a)}>
              <div className="glyph">{a.glyph}</div>
              <div>
                <div className="name">{a.name}</div>
                <div className="role">{a.role}</div>
                <div className="role">{a.stance} · conf {a.confidence.toFixed(2)}</div>
              </div>
              <div className="stat">
                <strong>{a.kept}</strong> / {a.retrieved}
                <div style={{ marginTop: 2 }}>{a.status === "error" ? "error" : `${(a.durationMs / 1000).toFixed(1)}s`}</div>
              </div>
            </div>
          )) : (
            <div style={{ color: "var(--muted)" }}>
              This run used the fast claim-graph path, so no separate retrieval agents were launched.
            </div>
          )}
        </div>

        <div className="card fade-up d3">
          <h3>Disagreement metrics</h3>
          {[
            ["Disagreement density", (r.metrics.disagreementDensity*100).toFixed(1)+"%", "|contradict| / |edges|", "Active controversy when > 15%."],
            ["Conflict ratio", r.metrics.conflictRatio.toFixed(2), "|contradict| / |support|", "Balance of opposing vs. agreeing edges."],
            ["Claim entropy", r.metrics.claimEntropy.toFixed(2)+" bits", "−Σ p log p over stances", "Diversity of stance distribution."],
            ["Visual–text mismatch", r.metrics.visualMismatch.toFixed(2), "1 − mean(CLIP align)", "Penalises figure–text contradictions."],
            ["Controversy class", r.metrics.controversyClass, "temporal density derivative", "Stable | narrowing | open."]
          ].map(([k, v, f, d], i) => (
            <div key={i} className="metric" onClick={() => onOpenMetric({ label: k, value: v, formula: f, desc: d, interp: d })}>
              <span className="lbl">{k}</span><span className="val">{v}</span>
            </div>
          ))}
        </div>

        <div className="card fade-up d4" onClick={onOpenHypothesis} style={{ cursor: "pointer" }}>
          <h3>Hypothesis <span className="small">X-IR · click to expand</span></h3>
          <div style={{ fontFamily: "Newsreader, serif", fontSize: 14.5, lineHeight: 1.5, color: "var(--ink-2)", fontStyle: "italic" }}>
            "{r.hypothesis.centralHypothesis}"
          </div>
          <div style={{ borderTop: "1px solid var(--hair)", marginTop: 10, paddingTop: 8, fontSize: 11.5, color: "var(--muted)" }}>
            <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 10.5, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 4 }}>expected counterclaims</div>
            {(r.hypothesis.expectedCounterclaims || []).length ? (
              (r.hypothesis.expectedCounterclaims || []).slice(0, 3).map((claim, i) => (
                <div key={i}>· {claim}</div>
              ))
            ) : (
              <div>· No explicit counterclaims were returned for this run.</div>
            )}
          </div>
        </div>
      </aside>
    </div>
  );
};

window.Results = Results;
