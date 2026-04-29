/* global React, Icon */
const { useState, useMemo, useEffect, useRef } = React;

// Rotating thinking words shown during loading — Perplexity-style
const THINKING_WORDS = [
  "flabbergasting","epistemological","disambiguating","triangulating","corroborating",
  "scrutinizing","adjudicating","deliberating","substantiating","interrogating",
  "extrapolating","calibrating","synthesizing","verifying","deducing",
  "cross-referencing","contextualizing","unpacking","reasoning","mapping",
];
function useRotatingWord(active, intervalMs = 900) {
  const [idx, setIdx] = useState(0);
  useEffect(() => {
    if (!active) { setIdx(0); return; }
    const id = setInterval(() => setIdx(i => (i + 1) % THINKING_WORDS.length), intervalMs);
    return () => clearInterval(id);
  }, [active]);
  return THINKING_WORDS[idx];
}

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

// ── Agent Deliberation Panel ───────────────────────────────────────────────────
// Redesigned multi-round debate UI showing the complete deliberation pipeline:
//   [R1] 4 independent retrievals → [Judge] conflict detection →
//   [R2] Builder ↔ Skeptic rebuttals → [Verdict] calibrated ruling
const STANCE_COLORS = {
  support: { border: "var(--support)", badge: "oklch(0.50 0.07 150)", text: "oklch(0.38 0.07 150)", bg: "oklch(0.97 0.02 150)" },
  contra:  { border: "var(--contra)",  badge: "oklch(0.50 0.13 30)",  text: "oklch(0.42 0.13 30)",  bg: "oklch(0.97 0.03 30)"  },
  neutral: { border: "var(--hair-2)",  badge: "var(--muted-2)",       text: "var(--muted)",          bg: "var(--panel)"         },
};

// Small glyph badge used in multiple places
const AgentBadge = ({ glyph, stance, size = 26 }) => {
  const c = STANCE_COLORS[stance] || STANCE_COLORS.neutral;
  return (
    <div style={{
      width: size, height: size, flexShrink: 0,
      borderRadius: Math.round(size * 0.27),
      background: c.badge, display: "grid", placeItems: "center",
      fontFamily: "Newsreader, serif", fontStyle: "italic",
      fontSize: size * 0.54, color: "#fff", fontWeight: 600,
    }}>{glyph}</div>
  );
};

// Pipeline stage indicator — shows the deliberation flow at a glance
const DelibPipeline = ({ hasDebate, hasVerdict, hasConflict, section, onSection }) => {
  const stages = [
    { id: "r1",      label: "Round 1", sub: "Retrieval",  active: true,       dot: "B S A J" },
    { id: "conflict",label: "Judge",   sub: "Conflict",   active: hasConflict, dot: "J"      },
    { id: "debate",  label: "Round 2", sub: "Rebuttals",  active: hasDebate,   dot: "B↔S"   },
    { id: "verdict", label: "Verdict", sub: "Ruling",     active: hasVerdict,  dot: "J"      },
  ];
  return (
    <div style={{
      display: "flex", alignItems: "stretch",
      borderBottom: "1px solid var(--hair)", background: "var(--bg)",
      overflowX: "auto",
    }}>
      {stages.map((st, i) => {
        const isCurrent = section === st.id || (section === "debate" && st.id === "conflict");
        return (
          <React.Fragment key={st.id}>
            {i > 0 && (
              <div style={{ display: "flex", alignItems: "center", padding: "0 2px", color: "var(--hair-2)", fontSize: 14, flexShrink: 0 }}>›</div>
            )}
            <button
              onClick={() => st.active && st.id !== "conflict" && onSection(st.id)}
              style={{
                flex: "1 0 auto", padding: "9px 10px 7px",
                display: "flex", flexDirection: "column", alignItems: "center", gap: 2,
                borderBottom: isCurrent ? "2px solid var(--accent)" : "2px solid transparent",
                opacity: st.active ? 1 : 0.35,
                cursor: st.active && st.id !== "conflict" ? "pointer" : "default",
                transition: "all 0.15s",
                minWidth: 64,
              }}
            >
              <span style={{
                fontSize: 10, fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase",
                color: isCurrent ? "var(--accent)" : st.active ? "var(--ink)" : "var(--muted-2)",
              }}>{st.label}</span>
              <span style={{ fontSize: 9, color: "var(--muted-2)", fontFamily: "JetBrains Mono, monospace" }}>{st.sub}</span>
            </button>
          </React.Fragment>
        );
      })}
    </div>
  );
};

const AgentDeliberationPanel = ({ agents, deliberation, onOpenAgent }) => {
  const [section, setSection] = useState("r1"); // "r1" | "debate" | "verdict"
  if (!agents || agents.length === 0) return null;

  const totalPapers = agents.reduce((s, a) => s + (a.retrieved || 0), 0);
  const totalSources= agents.reduce((s, a) => s + (a.num_chunks || 0), 0);
  const delib = deliberation || {};
  const hasDebate  = delib.round2  && delib.round2.length > 0 && delib.round2.some(r => r.rebuttal);
  const hasVerdict = !!(delib.verdict);
  const hasConflict= !!(delib.conflict);

  return (
    <div className="agent-delib fade-up" style={{ animationDelay: "0.08s" }}>

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="agent-delib-hd">
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ color: "var(--accent)", display: "flex", alignItems: "center" }}>
            <Icon name="cpu" size={13}/>
          </span>
          <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--ink-2)" }}>
            4-Agent Deliberation
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span className="delib-stat">{totalSources || totalPapers} papers</span>
          {hasDebate  && <span className="delib-stat">2 rounds</span>}
          {hasVerdict && <span className="delib-stat" style={{ color: "var(--accent)", borderColor: "var(--accent)" }}>verdict ✓</span>}
        </div>
      </div>

      {/* ── Pipeline timeline ─────────────────────────────────────────────── */}
      <DelibPipeline
        hasDebate={hasDebate}
        hasVerdict={hasVerdict}
        hasConflict={hasConflict}
        section={section}
        onSection={setSection}
      />

      {/* ── Round 1: independent retrieval + stance formation ────────────── */}
      {section === "r1" && (
        <>
          {/* Agent grid */}
          <div className="agent-delib-grid">
            {agents.map((a, i) => {
              const c = STANCE_COLORS[a.stance] || STANCE_COLORS.neutral;
              // Known backend fallback / placeholder strings that should not render as content
              const FALLBACK_PREFIXES = [
                "No synthesis returned",
                "Mainstream evidence on this topic is supported",
                "Several methodological limitations and heterogeneous",
                "Synthesis temporarily unavailable",
                "No papers directly address",
              ];
              const hasView = !!(a.reasoning && !FALLBACK_PREFIXES.some(p => a.reasoning.startsWith(p)));
              const agentRole = {
                builder:  "Builds the strongest supporting case from the literature",
                skeptic:  "Searches for counter-evidence and contradictions",
                archivist:"Grounds claims in citation metadata and source diversity",
                judge:    "Calibrates confidence and detects factual conflicts",
              }[a.agent_name || a.name?.toLowerCase()] || a.role || "";
              return (
                <div key={a.key || i} className="agent-delib-card"
                  style={{ borderTop: `3px solid ${c.border}`, background: c.bg }}
                  onClick={() => onOpenAgent(a)}
                  title={`Click to inspect ${a.name} agent — ${agentRole}`}>
                  <div style={{ display: "flex", alignItems: "flex-start", gap: 8, marginBottom: 6 }}>
                    <AgentBadge glyph={a.glyph} stance={a.stance}/>
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                        <span style={{ fontSize: 12.5, fontWeight: 700, color: "var(--ink)", lineHeight: 1.3 }}>{a.name}</span>
                        <span style={{ fontSize: 9, fontFamily: "JetBrains Mono, monospace", color: "var(--muted-2)", fontWeight: 500 }}>
                          {(a.confidence||0).toFixed(2)} conf
                        </span>
                      </div>
                      <div style={{ fontSize: 10, color: c.text, fontWeight: 600, letterSpacing: "0.04em", textTransform: "uppercase", marginTop: 1 }}>{a.stance}</div>
                    </div>
                  </div>
                  {/* Agent role description */}
                  <div style={{ fontSize: 10, color: "var(--muted)", marginBottom: 7, lineHeight: 1.4, fontFamily: "JetBrains Mono, monospace" }}>
                    {agentRole}
                  </div>
                  {/* Synthesis */}
                  {hasView ? (
                    <div style={{ fontSize: 12, color: "var(--ink-2)", lineHeight: 1.55, marginBottom: 8, fontFamily: "Newsreader, serif", fontStyle: "italic", minHeight: 44 }}>
                      "{a.reasoning}"
                    </div>
                  ) : (
                    <div style={{ fontSize: 11, color: "var(--muted-2)", lineHeight: 1.5, marginBottom: 8, fontStyle: "italic", minHeight: 44, display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ opacity: 0.5, fontSize: 16, lineHeight: 1 }}>—</span>
                      <span>{a.retrieved > 0 ? "Synthesis unavailable for this query." : "No corpus match found."}</span>
                    </div>
                  )}
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap", fontFamily: "JetBrains Mono, monospace", fontSize: 10, color: "var(--muted-2)", borderTop: "1px solid var(--hair)", paddingTop: 6, marginTop: "auto" }}>
                    <span><strong style={{ color: "var(--ink)" }}>{a.retrieved || a.num_chunks || 0}</strong> papers</span>
                    {a.durationMs > 0 && <span>{(a.durationMs/1000).toFixed(1)}s</span>}
                  </div>
                  {a.topSources && a.topSources.filter(s => s.title && !["Unknown","(no title)"].includes(s.title)).length > 0 && (
                    <div style={{ marginTop: 5 }}>
                      {a.topSources.filter(s => s.title && !["Unknown","(no title)"].includes(s.title)).slice(0,2).map((s,si) => (
                        <div key={si} style={{ fontSize: 10, color: "var(--muted)", lineHeight: 1.35, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          · {s.title}{s.year ? ` (${s.year})` : ""}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Round 1 → next step hint */}
          {hasConflict && (
            <div
              style={{ margin: "0 14px 14px", padding: "9px 12px", borderRadius: 8, background: "oklch(0.97 0.025 50)", border: "1px solid oklch(0.87 0.04 50)", display: "flex", alignItems: "flex-start", gap: 10, cursor: "pointer" }}
              onClick={() => setSection("debate")}
            >
              <span style={{ display: "flex", alignItems: "center", flexShrink: 0, color: "var(--accent)", marginTop: 1 }}>
                <Icon name="warn" size={14}/>
              </span>
              <div>
                <div style={{ fontSize: 10.5, fontWeight: 700, letterSpacing: "0.04em", textTransform: "uppercase", color: "var(--accent)", marginBottom: 3 }}>
                  Judge detected a conflict — Round 2 triggered
                </div>
                <div style={{ fontSize: 12, color: "var(--ink-2)", lineHeight: 1.45, fontFamily: "Newsreader, serif", fontStyle: "italic" }}>
                  {delib.conflict}
                </div>
              </div>
              <span style={{ display: "flex", alignItems: "center", flexShrink: 0, color: "var(--muted-2)" }}>
                <Icon name="chevron-right" size={14}/>
              </span>
            </div>
          )}
        </>
      )}

      {/* ── Round 2: Judge conflict + Builder↔Skeptic rebuttals ──────────── */}
      {section === "debate" && (
        <div style={{ padding: "14px 14px 10px" }}>

          {/* What the Judge found */}
          {hasConflict && (
            <div style={{ marginBottom: 16 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <AgentBadge glyph="J" stance="neutral" size={24}/>
                <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase", color: "var(--muted)" }}>
                  Judge · conflict identified
                </span>
              </div>
              <div style={{ padding: "10px 12px", background: "oklch(0.97 0.025 50)", border: "1px solid oklch(0.85 0.04 50)", borderRadius: 8, display: "flex", gap: 10, alignItems: "flex-start" }}>
                <span style={{ fontSize: 11, fontFamily: "JetBrains Mono, monospace", fontWeight: 700, color: "var(--accent)", letterSpacing: "0.06em", whiteSpace: "nowrap", marginTop: 1 }}>
                  CONFLICT
                </span>
                <div style={{ fontSize: 12.5, color: "var(--ink-2)", lineHeight: 1.5, fontFamily: "Newsreader, serif", fontStyle: "italic" }}>
                  {delib.conflict}
                </div>
              </div>
              <div style={{ marginTop: 8, fontSize: 11, color: "var(--muted)", lineHeight: 1.5 }}>
                The Judge sent this conflict back to Builder and Skeptic for a direct rebuttal exchange.
              </div>
            </div>
          )}

          {/* Rebuttals — Builder and Skeptic respond to each other */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {(delib.round2 || []).filter(r => r.rebuttal).map((r, i) => {
              const c  = STANCE_COLORS[r.stance] || STANCE_COLORS.neutral;
              const r1 = agents.find(a => a.agent_name === r.agent || a.key === r.agent);
              const r1View = r1 ? (r1.reasoning || "") : "";
              return (
                <div key={i} style={{ borderLeft: `3px solid ${c.border}`, padding: "10px 12px", background: c.bg, borderRadius: "0 8px 8px 0" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                    <AgentBadge glyph={r.glyph} stance={r.stance} size={22}/>
                    <span style={{ fontSize: 11.5, fontWeight: 700, color: "var(--ink)" }}>{r.name}</span>
                    <span style={{ fontSize: 10, color: "var(--muted-2)", fontFamily: "JetBrains Mono, monospace" }}>
                      → responding to {r.respondingTo || r.responding_to}
                    </span>
                  </div>
                  {r1View && (
                    <div style={{ fontSize: 11, color: "var(--muted)", fontStyle: "italic", marginBottom: 6, lineHeight: 1.4 }}>
                      Round 1: "{r1View}"
                    </div>
                  )}
                  <div style={{ fontSize: 13, color: "var(--ink-2)", lineHeight: 1.55, fontFamily: "Newsreader, serif" }}>
                    → {r.rebuttal}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Hint to verdict */}
          {hasVerdict && (
            <div
              style={{ marginTop: 12, padding: "8px 12px", borderRadius: 8, border: "1px solid var(--hair)", background: "var(--panel)", display: "flex", alignItems: "center", justifyContent: "space-between", cursor: "pointer" }}
              onClick={() => setSection("verdict")}
            >
              <span style={{ fontSize: 11.5, color: "var(--muted)", fontWeight: 500 }}>Judge has issued a final verdict</span>
              <span style={{ display: "flex", alignItems: "center", gap: 4, color: "var(--accent)", fontSize: 11, fontWeight: 600 }}>
                View ruling <Icon name="chevron-right" size={12}/>
              </span>
            </div>
          )}
        </div>
      )}

      {/* ── Judge verdict ─────────────────────────────────────────────────── */}
      {section === "verdict" && (
        <div style={{ padding: "14px 14px 12px" }}>
          {/* What went in */}
          <div style={{ marginBottom: 14, padding: "8px 12px", borderRadius: 8, background: "var(--bg)", fontSize: 11, color: "var(--muted)", lineHeight: 1.5, fontFamily: "JetBrains Mono, monospace" }}>
            Synthesized from: Builder (support) + Skeptic (contra) + Archivist (neutral) + R2 rebuttals
          </div>
          {/* Judge ruling */}
          <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
            <AgentBadge glyph="J" stance="neutral" size={34}/>
            <div style={{ flex: 1 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase", color: "var(--muted)" }}>
                  Judge · calibrated ruling
                </span>
                {delib.durationMs > 0 && (
                  <span style={{ fontSize: 10, fontFamily: "JetBrains Mono, monospace", color: "var(--muted-2)" }}>
                    {(delib.durationMs / 1000).toFixed(1)}s
                  </span>
                )}
              </div>
              <div style={{ fontSize: 14.5, fontFamily: "Newsreader, serif", color: "var(--ink)", lineHeight: 1.7, padding: "10px 14px", background: "var(--panel-2)", borderRadius: 8, borderLeft: "3px solid var(--accent)" }}>
                {delib.verdict}
              </div>
            </div>
          </div>
          {/* Back to round 2 */}
          {hasDebate && (
            <button
              style={{ marginTop: 12, fontSize: 11, color: "var(--muted)", display: "flex", alignItems: "center", gap: 4 }}
              onClick={() => setSection("debate")}
            >
              <Icon name="arrow-left" size={11}/> Back to Round 2 exchange
            </button>
          )}
        </div>
      )}

    </div>
  );
};

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

const TypingIndicator = ({ pendingQuery, thinkingWord }) => (
  <>
    {pendingQuery && (
      <div className="chat-msg user fade-up" style={{ marginBottom: 12 }}>
        <div className="chat-user-bubble">
          <strong>Q:</strong> {pendingQuery}
        </div>
      </div>
    )}
    <div className="chat-msg assistant fade-up">
      <div className="chat-bot-bubble card thinking-bubble" style={{ marginBottom: 0 }}>
        <div className="thinking-header">
          <span className="thinking-dot"/>
          <span className="thinking-dot"/>
          <span className="thinking-dot"/>
          <span className="thinking-word">{thinkingWord}</span>
        </div>
      </div>
    </div>
  </>
);

const Results = ({ result, loading, error, pendingQuery, pendingOptions, tracePlan, onRetry, onBack, onSubmit, onOpenSource, onOpenClaim, onOpenAgent, onOpenFigure, onOpenMetric, onOpenHypothesis }) => {
  const [drawer, setDrawer] = useState(null); // null | 'sources'|'claims'|'graph'|'views'|'analysis'
  const [drawerTab, setDrawerTab] = useState('sources');
  const [followUp, setFollowUp] = useState("");
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

  const thinkingWord = useRotatingWord(loading);
  const r = result;


  // First query loading (no prior result) — minimal full-screen state
  if (loading && !result) {
    const hasPdfPending = !!(pendingOptions && pendingOptions.pdfContext);
    return (
      <div className="results results-chat-first">
        <div className="chat-col">
          <div className="q-line fade-up">
            <button className="back" onClick={onBack} title="Back"><Icon name="arrow-left" size={16}/></button>
            <div style={{ flex: 1 }}>
              <div className="q-text serif">{pendingQuery || "EVIRAG inquiry"}</div>
              {hasPdfPending && (
                <div className="q-meta" style={{ marginTop: 6 }}>
                  <span className="pill" style={{ gap: 4, display: "inline-flex", alignItems: "center" }}>
                    <Icon name="doc" size={10}/> PDF context attached
                  </span>
                </div>
              )}
            </div>
          </div>
          <div className="chat-history">
            <TypingIndicator pendingQuery={pendingQuery} thinkingWord={thinkingWord}/>
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
  const contraEdges  = r.graph.edges.filter(e => e.kind === "contra").length;
  const neutralEdges = r.graph.edges.filter(e => e.kind === "neutral").length;
  const total        = r.graph.edges.length;
  const calibration  = r.calibration || [];
  // Use accumulated backend claims (grow per turn) as primary; fall back to view claims
  const accClaims = Array.isArray(r.claims) ? r.claims : [];
  const viewClaims = r.views.flatMap(v => v.claims.map(c => ({ ...c, view: v.label, kind: v.kind })));
  const visibleClaims = accClaims.length > 0
    ? accClaims.map((c, i) => ({
        n: `C-${String(i + 1).padStart(2, "0")}`,
        text: c.text || c.claim || "(no text)",
        src: c.source_doc_title || c.doc_title || "—",
        cites: [],
        view: c.stance || "claim",
        kind: c.kind || "neutral",
        context: c.context || "",
        claimId: c.id || "",
        docId: c.source_doc_id || "",
      }))
    : viewClaims;
  const totalAccClaims = r.chat?.total_claims ?? visibleClaims.length;
  const totalAccSources = r.chat?.total_sources ?? r.metrics.sources;
  const disputedLabel = String(r.intent.dispute || "contested").replace(/_/g, " ");

  const openDrawer = (panel) => { setDrawerTab(panel); setDrawer(panel); };

  // ── Drawer panel content ─────────────────────────────────────────────────────
  const FALLBACK_PREFIXES_D = [
    "No synthesis returned","Mainstream evidence on this topic is supported",
    "Several methodological limitations","Synthesis temporarily unavailable","No papers directly address",
  ];
  const DrawerContent = () => {
    const hasAgents = r.agents && r.agents.length > 0;
    const panels = [
      { id: "sources",      icon: "library", label: "Sources", count: totalAccSources },
      { id: "claims",       icon: "doc",     label: "Claims",  count: totalAccClaims  },
      { id: "graph",        icon: "graph",   label: "Graph",   count: r.graph.nodes.length },
      ...(hasAgents ? [{ id: "deliberation", icon: "cpu", label: "Agents", count: r.agents.length }] : []),
      { id: "analysis",     icon: "chart",   label: "Analysis",count: null },
    ];
    return (
      <div className="drawer-overlay" onClick={() => setDrawer(null)}>
        <div className="drawer" onClick={e => e.stopPropagation()}>
          <div className="drawer-header">
            {/* Close button FIRST so it's never pushed off-screen by tab overflow */}
            <button className="drawer-close" onClick={() => setDrawer(null)} style={{ marginRight: 4 }}>
              <Icon name="close" size={16}/>
            </button>
            <div className="drawer-tabs">
              {panels.map(p => (
                <button key={p.id}
                  className={"drawer-tab" + (drawerTab === p.id ? " is-active" : "")}
                  onClick={() => setDrawerTab(p.id)}>
                  <Icon name={p.icon} size={12}/>
                  {p.label}
                  {p.count != null && <span className="count">{p.count}</span>}
                </button>
              ))}
            </div>
          </div>
          <div className="drawer-body">
            {drawerTab === "sources" && (
              <div>
                {/* Prefer FAISS chat sources (peS2o corpus); fall back to EVIRAG sources */}
                {(r.chat?.sources?.length ? r.chat.sources : r.sources).map((s, idx) => (
                  <button key={s.n ?? idx} className="src-card" onClick={() => { setDrawer(null); onOpenSource(s); }}>
                    <div className="num">{s.n ?? idx + 1}</div>
                    <div>
                      <div className="src-title">{s.title}</div>
                      <div className="src-meta">
                        <span>{s.source || s.venue || s.doi || "peS2o corpus"}</span>
                        {s.year ? <span>· {s.year}</span> : null}
                      </div>
                      {s.snippet && <div className="src-snippet">{s.snippet.slice(0, 200)}{s.snippet.length > 200 ? "…" : ""}</div>}
                    </div>
                  </button>
                ))}
              </div>
            )}
            {drawerTab === "claims" && (
              <div>
                {visibleClaims.map((c, i) => (
                  <div key={i} className="claim-row" onClick={() => { setDrawer(null); onOpenClaim(c); }}
                    style={{ gridTemplateColumns: "32px minmax(0,1fr) minmax(80px,25%)", padding: "10px 0", gap: 12, alignItems: "start" }}>
                    <div className="num">{c.n}</div>
                    <div style={{ minWidth: 0 }}>
                      <div style={{ color: "var(--ink)" }}>{c.text}</div>
                      <div style={{ fontSize: 10.5, color: "var(--muted-2)", marginTop: 4, fontFamily: "JetBrains Mono, monospace" }}>
                        {c.view.toLowerCase()} · {(c.cites && c.cites.length) ? c.cites.join(", ") : "—"}
                      </div>
                    </div>
                    <div className="src" style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", textAlign: "right" }}
                      title={c.src}>{c.src}</div>
                  </div>
                ))}
              </div>
            )}
            {drawerTab === "graph" && (
              <div>
                <DisagreementGraph graph={r.graph} onNode={(n) => { setDrawer(null); onOpenClaim({
                  n: n.label, text: n.title || ("Claim " + n.label), src: n.docTitle || "graph",
                  cites: [], view: n.kind === "dom" ? "Dominant" : n.kind === "alt" ? "Alternative" : "Minority",
                  kind: n.kind, context: n.context || "", claimId: n.originalId || "",
                  docId: n.docId || "", sourcePath: n.sourcePath || "", docTitle: n.docTitle || ""
                }); }}/>
                <div style={{ marginTop: 12, borderTop: "1px solid var(--hair)", paddingTop: 12 }}>
                  {[["Support", supportEdges, "support", "NLI ENTAILS"], ["Contradict", contraEdges, "contra", "NLI CONTRADICTS"], ["Neutral", neutralEdges, "neutral", "NLI NEUTRAL"]].map(([lbl, ct, cls, f]) => (
                    <div key={lbl} className="bar-row" style={{ cursor: "pointer" }}
                      onClick={() => onOpenMetric({ label: lbl + " edges", value: ct, formula: f, desc: "", interp: "" })}>
                      <span className="name">{lbl}</span>
                      <div className="track"><div className={"fill " + cls} style={{ width: (total ? ct/total*100 : 0)+"%" }}/></div>
                      <span className="num">{ct}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {drawerTab === "deliberation" && r.agents && (
              <div style={{ paddingBottom: 8 }}>
                {/* Round 1 — agent cards, vertical */}
                <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--muted-2)", marginBottom: 10, paddingBottom: 6, borderBottom: "1px solid var(--hair)" }}>
                  Round 1 · Independent Retrieval
                </div>
                {r.agents.map((a, i) => {
                  const c = STANCE_COLORS[a.stance] || STANCE_COLORS.neutral;
                  const hasView = !!(a.reasoning && !FALLBACK_PREFIXES_D.some(p => a.reasoning.startsWith(p)));
                  return (
                    <div key={i} style={{ borderLeft: `3px solid ${c.border}`, background: c.bg, borderRadius: "0 8px 8px 0", padding: "11px 14px", marginBottom: 8, cursor: "pointer" }}
                      onClick={() => { setDrawer(null); onOpenAgent(a); }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: hasView ? 7 : 0 }}>
                        <AgentBadge glyph={a.glyph} stance={a.stance} size={22}/>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <span style={{ fontSize: 12.5, fontWeight: 700, color: "var(--ink)" }}>{a.name}</span>
                          <span style={{ fontSize: 10, color: c.text, fontWeight: 600, letterSpacing: "0.04em", textTransform: "uppercase", marginLeft: 8 }}>{a.stance}</span>
                        </div>
                        <span style={{ fontSize: 10, fontFamily: "JetBrains Mono, monospace", color: "var(--muted-2)", flexShrink: 0 }}>
                          {(a.confidence||0).toFixed(2)} · {a.retrieved||0}p
                        </span>
                      </div>
                      {hasView ? (
                        <div style={{ fontSize: 13, color: "var(--ink-2)", lineHeight: 1.6, fontFamily: "Newsreader, serif", fontStyle: "italic" }}>
                          "{a.reasoning}"
                        </div>
                      ) : (
                        <div style={{ fontSize: 11, color: "var(--muted-2)", fontStyle: "italic" }}>
                          — Synthesis unavailable for this query.
                        </div>
                      )}
                      {a.topSources && a.topSources.filter(s => s.title && !["Unknown","(no title)"].includes(s.title)).length > 0 && (
                        <div style={{ marginTop: 7, paddingTop: 5, borderTop: "1px solid var(--hair)" }}>
                          {a.topSources.filter(s => s.title && !["Unknown","(no title)"].includes(s.title)).slice(0,2).map((s,si) => (
                            <div key={si} style={{ fontSize: 10, color: "var(--muted)", lineHeight: 1.35, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                              · {s.title}{s.year ? ` (${s.year})` : ""}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}

                {/* Judge conflict */}
                {r.deliberation?.conflict && (
                  <div style={{ margin: "16px 0 8px" }}>
                    <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--muted-2)", marginBottom: 8, paddingBottom: 6, borderBottom: "1px solid var(--hair)" }}>
                      Judge · Conflict Detected
                    </div>
                    <div style={{ padding: "10px 14px", background: "oklch(0.97 0.025 50)", border: "1px solid oklch(0.87 0.04 50)", borderRadius: 8 }}>
                      <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase", color: "var(--accent)", marginBottom: 5 }}>CONFLICT</div>
                      <div style={{ fontSize: 13, color: "var(--ink-2)", lineHeight: 1.55, fontFamily: "Newsreader, serif", fontStyle: "italic" }}>{r.deliberation.conflict}</div>
                    </div>
                  </div>
                )}

                {/* Round 2 rebuttals */}
                {r.deliberation?.round2?.some(r2 => r2.rebuttal) && (
                  <div style={{ margin: "16px 0 8px" }}>
                    <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--muted-2)", marginBottom: 8, paddingBottom: 6, borderBottom: "1px solid var(--hair)" }}>
                      Round 2 · Rebuttals
                    </div>
                    {r.deliberation.round2.filter(r2 => r2.rebuttal).map((r2, i) => {
                      const c2 = STANCE_COLORS[r2.stance] || STANCE_COLORS.neutral;
                      return (
                        <div key={i} style={{ borderLeft: `3px solid ${c2.border}`, padding: "9px 12px", background: c2.bg, borderRadius: "0 8px 8px 0", marginBottom: 8 }}>
                          <div style={{ fontSize: 11, fontWeight: 700, color: "var(--ink)", marginBottom: 4 }}>{r2.name}</div>
                          <div style={{ fontSize: 13, color: "var(--ink-2)", lineHeight: 1.55, fontFamily: "Newsreader, serif" }}>→ {r2.rebuttal}</div>
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Verdict */}
                {r.deliberation?.verdict && (
                  <div style={{ margin: "16px 0 4px" }}>
                    <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--muted-2)", marginBottom: 8, paddingBottom: 6, borderBottom: "1px solid var(--hair)" }}>
                      Judge · Final Verdict
                    </div>
                    <div style={{ padding: "12px 14px", background: "var(--panel-2)", borderLeft: "3px solid var(--accent)", borderRadius: "0 8px 8px 0" }}>
                      <div style={{ fontSize: 14.5, fontFamily: "Newsreader, serif", color: "var(--ink)", lineHeight: 1.7 }}>
                        {r.deliberation.verdict}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {drawerTab === "analysis" && (
              <div>
                <div style={{ marginBottom: 20 }}>
                  <div className="drawer-section-title">Confidence calibration
                    <span style={{ marginLeft: 8, fontFamily: "JetBrains Mono, monospace", fontSize: 11, color: "var(--muted)" }}>{r.metrics.confidenceLabel}</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "baseline", gap: 8, margin: "8px 0 12px" }}>
                    <div className="serif" style={{ fontSize: 36, lineHeight: 1, letterSpacing: "-0.02em" }}>{r.metrics.confidence.toFixed(2)}</div>
                    <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 11, color: "var(--muted)" }}>/ 1.00</div>
                  </div>
                  {calibration.map(([k, v, c, desc], i) => (
                    <div key={i} className="bar-row" style={{ cursor: "pointer" }}
                      onClick={() => onOpenMetric({ label: k, value: v, formula: "weighted factor", desc, interp: "Contributes to overall confidence." })}>
                      <span className="name">{k}</span>
                      <div className="track"><div className={"fill " + (c || "")} style={{ width: (v*100)+"%" }}/></div>
                      <span className="num">.{(v*100).toFixed(0)}</span>
                    </div>
                  ))}
                </div>
                <div style={{ marginBottom: 20, borderTop: "1px solid var(--hair)", paddingTop: 16 }}>
                  <div className="drawer-section-title">Disagreement metrics</div>
                  {[
                    ["Disagreement density", (r.metrics.disagreementDensity*100).toFixed(1)+"%", "|contradict| / |edges|", "Active controversy when > 15%."],
                    ["Conflict ratio",        r.metrics.conflictRatio.toFixed(2),                "|contradict| / |support|", "Balance of opposing vs. agreeing edges."],
                    ["Claim entropy",          r.metrics.claimEntropy.toFixed(2)+" bits",         "−Σ p log p over stances", "Diversity of stance distribution."],
                    ["Visual–text mismatch",   r.metrics.visualMismatch.toFixed(2), "1 − term coverage", "Query term coverage in retrieved papers. 0 = all key terms found; 1 = none found."],
                    ["Controversy class",      r.metrics.controversyClass,                         "temporal density derivative", "Stable | narrowing | open."]
                  ].map(([k, v, f, d], i) => (
                    <div key={i} className="metric" style={{ cursor: "pointer" }}
                      onClick={() => onOpenMetric({ label: k, value: v, formula: f, desc: d, interp: d })}>
                      <span className="lbl">{k}</span><span className="val">{v}</span>
                    </div>
                  ))}
                </div>
                <div style={{ borderTop: "1px solid var(--hair)", paddingTop: 16 }}>
                  <div className="drawer-section-title">Hypothesis <span style={{ fontWeight: 400, color: "var(--muted)", fontSize: 11 }}>X-IR</span></div>
                  <div style={{ fontFamily: "Newsreader, serif", fontSize: 14, fontStyle: "italic", color: "var(--ink-2)", margin: "8px 0 12px", lineHeight: 1.5 }}>
                    "{r.hypothesis.centralHypothesis}"
                  </div>
                  {(r.hypothesis.expectedCounterclaims || []).slice(0, 3).map((c, i) => (
                    <div key={i} style={{ fontSize: 12, color: "var(--muted)", marginBottom: 4 }}>· {c}</div>
                  ))}
                </div>
                {r.agents.length > 0 && (
                  <div style={{ borderTop: "1px solid var(--hair)", paddingTop: 16, marginTop: 16 }}>
                    <div className="drawer-section-title">Agent runs</div>
                    {r.agents.map((a, i) => (
                      <div key={i} className="agent" onClick={() => { setDrawer(null); onOpenAgent(a); }}>
                        <div className="glyph">{a.glyph}</div>
                        <div><div className="name">{a.name}</div><div className="role">{a.role}</div></div>
                        <div className="stat"><strong>{a.kept}</strong> / {a.retrieved}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="results results-chat-first">
      {/* ── Chat column ───────────────────────────────────────────────────────── */}
      <div className="chat-col">
        <div className="q-line fade-up">
          <button className="back" onClick={onBack} title="Back"><Icon name="arrow-left" size={16}/></button>
          <div style={{ flex: 1 }}>
            <div className="q-text serif">{r.query}</div>
            <div className="q-meta">
              <span className="pill warn" onClick={() => openDrawer("analysis")}><Icon name="warn" size={11}/> {r.intent.dispute.replace("_", " ")}</span>
              <span className="pill" onClick={() => openDrawer("sources")}><span className="lbl">sources</span> {r.metrics.sources}</span>
              <span className="pill" onClick={() => openDrawer("analysis")}><span className="lbl">conf</span> {r.metrics.confidenceLabel} · {r.metrics.confidence.toFixed(2)}</span>
            </div>
          </div>
        </div>

        {/* Chat history */}
        {r.chat && r.chat.history && r.chat.history.length > 0 ? (
          <div className="chat-history">
            {r.chat.history.map((msg, i) => (
              <div key={i} className={`chat-msg ${msg.role} fade-up`} style={{ animationDelay: `${i * 0.05}s` }}>
                {msg.role === "user" ? (() => {
                  // Strip any PDF preamble injected by api.jsx before displaying
                  const cleanContent = msg.content
                    .replace(/^\[Uploaded document context[^\]]*\]\n[\s\S]*?\[User question\]\n/, '')
                    .replace(/^\[Uploaded document context[^\]]*\][\s\S]*?\[User question\]\n/, '');
                  const hasPdf = msg.content !== cleanContent;
                  return (
                  <div className="chat-user-bubble">
                    {hasPdf && (
                      <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 6, display: "flex", alignItems: "center", gap: 4 }}>
                        <Icon name="doc" size={10}/> PDF context
                      </div>
                    )}
                    <strong>Q:</strong> {cleanContent}
                  </div>
                  );
                })() : (
                  <div className="chat-bot-bubble card">
                    {msg.claim && (
                      <div className="evolving-claim" style={{ marginBottom: 12, paddingBottom: 12, borderBottom: "1px solid var(--hair)", color: "var(--accent)" }}>
                        <Icon name="sparkle" size={14} style={{ marginRight: 6 }}/>
                        <em>{msg.claim}</em>
                      </div>
                    )}
                    <div className="markdown-body" dangerouslySetInnerHTML={{ __html: msg.content
                      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                      .replace(/\*([^*\n]+?)\*/g, '<em>$1</em>')
                      .replace(/\n/g, '<br/>')
                      .replace(/\[(\d+)\]/g, '<span class="cite">[$1]</span>') }} />
                  </div>
                )}
              </div>
            ))}
            {loading && <TypingIndicator pendingQuery={pendingQuery} thinkingWord={thinkingWord}/>}
          </div>
        ) : (
          <>
            <div className="card fade-up direct-answer">
              <h3>Answer <span className="small">from retrieved corpus evidence</span></h3>
              <p>{r.directAnswer}</p>
            </div>
            {loading && <div className="chat-history" style={{marginTop: 16}}><TypingIndicator pendingQuery={pendingQuery} thinkingWord={thinkingWord}/></div>}
          </>
        )}

        {/* ── Sticky footer: insight bar + follow-up composer ─────────────────
             position:sticky + bottom:0 keeps this pinned to the bottom of the
             viewport as the user scrolls up through chat history, mirroring the
             ChatGPT / Perplexity UX pattern. */}
        <div className="chat-footer">
          {/* Insight strip */}
          {(() => {
            const totalSources = r.chat?.total_sources ?? r.metrics.sources;
            const totalClaims  = r.chat?.total_claims  ?? r.metrics.claims;
            const hasAgents    = r.agents && r.agents.length > 0;
            const hasVerdict   = !!(r.deliberation?.verdict);
            return (
              <div className="insight-bar fade-up">
                <button className="insight-chip" onClick={() => openDrawer("sources")}>
                  <Icon name="library" size={11}/> {totalSources} sources
                </button>
                {totalClaims > 0 && (
                  <button className="insight-chip" onClick={() => openDrawer("claims")}>
                    <Icon name="doc" size={11}/> {totalClaims} claims
                  </button>
                )}
                {r.graph.nodes.length > 0 && (
                  <button className="insight-chip" onClick={() => openDrawer("graph")}>
                    <Icon name="graph" size={11}/> claim graph
                  </button>
                )}
                {hasAgents && (
                  <button className="insight-chip insight-chip--delib" onClick={() => openDrawer("deliberation")}>
                    <Icon name="cpu" size={11}/> deliberation{hasVerdict ? " ✓" : ""}
                  </button>
                )}
                <button className="insight-chip insight-chip--conf" onClick={() => openDrawer("analysis")}>
                  <Icon name="chart" size={11}/> {r.metrics.confidence.toFixed(2)} conf
                </button>
              </div>
            );
          })()}

          {/* Follow-up composer */}
          <div className="composer fade-up" style={{ marginTop: 8 }}>
            <input
              type="text"
              className="composer-input"
              style={{ padding: "14px 16px", fontSize: 14 }}
              placeholder="Ask a follow-up question..."
              value={followUp}
              onChange={e => setFollowUp(e.target.value)}
              onKeyDown={e => {
                if (e.key === "Enter" && followUp.trim()) { onSubmit(followUp); setFollowUp(""); }
              }}
            />
            <button
              className="send-btn"
              style={{ position: "absolute", right: 8, top: "50%", transform: "translateY(-50%)" }}
              onClick={() => { if (followUp.trim()) { onSubmit(followUp); setFollowUp(""); } }}
              disabled={!followUp.trim()}
            >
              <Icon name="arrow-up" size={16} stroke={2}/>
            </button>
          </div>
        </div>
      </div>

      {/* ── Drawer overlay ─────────────────────────────────────────────────────── */}
      {drawer && <DrawerContent/>}
    </div>
  );

};

window.Results = Results;
