/* global React, Icon */
const { useState, useMemo } = React;

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
        <div key={i} className="claim-row" onClick={() => onClaim({ ...c, view: v.label, kind: v.kind })}>
          <div className="num">{c.n}</div>
          <div>{c.text}</div>
          <div className="src">
            {c.src}
            {c.cites.map(s => <span key={s} className="cite" onClick={(e) => { e.stopPropagation(); onClaim({ ...c, view: v.label, kind: v.kind }); }}>{s}</span>)}
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

const Results = ({ result, onBack, onOpenSource, onOpenClaim, onOpenAgent, onOpenFigure, onOpenMetric, onOpenHypothesis }) => {
  const r = result;
  const [tab, setTab] = useState("answer");
  const supportEdges = r.graph.edges.filter(e => e.kind === "support").length;
  const contraEdges = r.graph.edges.filter(e => e.kind === "contra").length;
  const neutralEdges = r.graph.edges.filter(e => e.kind === "neutral").length;
  const total = r.graph.edges.length;

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
            <div className="callout fade-up" onClick={onOpenHypothesis} style={{ cursor: "pointer" }}>
              <div className="ico"><Icon name="warn" size={18}/></div>
              <div>
                <h4>This question is <em>highly disputed</em>. EVIRAG is surfacing — not resolving — the disagreement.</h4>
                <p>Synthesis was suppressed: 3 distinct positions retained from 47 atomic claims. Tap to inspect the X-IR hypothesis trace.</p>
              </div>
            </div>
            {r.views.map((v, i) => <ViewBlock key={i} v={v} idx={i} onClaim={onOpenClaim}/>)}
          </div>
        )}

        {tab === "graph" && (
          <div>
            <DisagreementGraph graph={r.graph} onNode={(n) => onOpenClaim({ n: n.label, text: "Claim " + n.label + " — open to see extracted text.", src: "graph", cites: [], view: n.kind === "dom" ? "Dominant" : n.kind === "alt" ? "Alternative" : "Minority", kind: n.kind })}/>
            <div className="card fade-up d3">
              <h3>Edge composition <span className="small">{total} edges</span></h3>
              <div className="bar-row" onClick={() => onOpenMetric({ label: "Support edges", value: supportEdges, formula: "NLI ENTAILS", desc: "Pairs of claims where one entails the other.", interp: "Higher = consensus or redundancy." })} style={{ cursor: "pointer" }}>
                <span className="name">Support</span>
                <div className="track"><div className="fill support" style={{ width: (supportEdges/total*100)+"%" }}/></div>
                <span className="num">{supportEdges}</span>
              </div>
              <div className="bar-row" onClick={() => onOpenMetric({ label: "Contradict edges", value: contraEdges, formula: "NLI CONTRADICTS", desc: "Pairs of claims in direct logical opposition.", interp: "The signal EVIRAG cares about most." })} style={{ cursor: "pointer" }}>
                <span className="name">Contradict</span>
                <div className="track"><div className="fill contra" style={{ width: (contraEdges/total*100)+"%" }}/></div>
                <span className="num">{contraEdges}</span>
              </div>
              <div className="bar-row" onClick={() => onOpenMetric({ label: "Neutral edges", value: neutralEdges, formula: "NLI NEUTRAL", desc: "Claim pairs that neither support nor contradict.", interp: "High share suggests semantic drift between subdomains." })} style={{ cursor: "pointer" }}>
                <span className="name">Neutral</span>
                <div className="track"><div className="fill neutral" style={{ width: (neutralEdges/total*100)+"%" }}/></div>
                <span className="num">{neutralEdges}</span>
              </div>
            </div>
          </div>
        )}

        {tab === "claims" && (
          <div className="card fade-up">
            <h3>Atomic claims <span className="small">{r.metrics.claims} extracted · {r.views.flatMap(v => v.claims).length} shown</span></h3>
            {r.views.flatMap(v => v.claims.map(c => ({ ...c, view: v.label, kind: v.kind }))).map((c, i) => (
              <div key={i} className="claim-row" onClick={() => onOpenClaim(c)} style={{ gridTemplateColumns: "32px 1fr auto", padding: "10px 0" }}>
                <div className="num">{c.n}</div>
                <div>
                  <div style={{ color: "var(--ink)" }}>{c.text}</div>
                  <div style={{ fontSize: 10.5, color: "var(--muted-2)", marginTop: 4, fontFamily: "JetBrains Mono, monospace" }}>
                    view: {c.view.toLowerCase()} · cited in {c.cites.join(", ")}
                  </div>
                </div>
                <div className="src">{c.src}</div>
              </div>
            ))}
          </div>
        )}

        {tab === "visual" && (
          <div className="card fade-up">
            <h3>Visual evidence (CLIP ViT-B/16) <span className="small">12 figures · mismatch 0.30</span></h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 12, marginTop: 8 }}>
              {[1,2,3,4,5,6].map(i => (
                <button key={i} onClick={() => onOpenFigure({ i, paper: i+2, align: (0.62 + i*0.04).toFixed(2), caption: "Test error vs. width" })} style={{ background: "var(--bg)", border: "1px solid var(--hair)", borderRadius: 8, overflow: "hidden", textAlign: "left", cursor: "pointer", padding: 0 }}>
                  <div style={{ height: 110, background: "repeating-linear-gradient(135deg, var(--panel), var(--panel) 6px, var(--panel-2) 6px, var(--panel-2) 12px)", display: "grid", placeItems: "center" }}>
                    <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 10, color: "var(--muted)" }}>fig {i} · double-descent</span>
                  </div>
                  <div style={{ padding: "8px 10px", fontSize: 11.5, color: "var(--ink-2)" }}>
                    <div>Test error vs. width</div>
                    <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 10.5, color: "var(--muted)", marginTop: 3 }}>
                      align {(0.62 + i*0.04).toFixed(2)} · paper #{i+2}
                    </div>
                  </div>
                </button>
              ))}
            </div>
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
                    <span>{s.venue}</span>
                    <span>· {s.year}</span>
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
          {[
            ["Claim agreement", 0.58, "", "How aligned the kept claims are with each other."],
            ["Source diversity", 0.74, "accent", "Spread of retrieved sources across venues and years."],
            ["Contradiction", 0.65, "contra", "Severity of opposing claims (higher = more conflict)."],
            ["Unverified assump.", 0.43, "muted", "Hypothesis assumptions not validated against evidence."],
            ["Visual alignment", 0.70, "support", "CLIP-based agreement between figures and textual claims."]
          ].map(([k, v, c, desc], i) => (
            <div key={i} className="bar-row" onClick={() => onOpenMetric({ label: k, value: v, formula: "weighted factor", desc, interp: "Contributes to overall confidence." })} style={{ cursor: "pointer" }}>
              <span className="name">{k}</span>
              <div className="track"><div className={"fill " + (c || "")} style={{ width: (v*100)+"%", background: c === "muted" ? "var(--muted-2)" : undefined }}/></div>
              <span className="num">.{(v*100).toFixed(0)}</span>
            </div>
          ))}
        </div>

        <div className="card fade-up d2">
          <h3>Multi-agent retrieval <span className="small">deliberative</span></h3>
          {r.agents.map((a, i) => (
            <div key={i} className="agent" onClick={() => onOpenAgent(a)}>
              <div className="glyph">{a.glyph}</div>
              <div>
                <div className="name">{a.name}</div>
                <div className="role">{a.role}</div>
              </div>
              <div className="stat">
                <strong>{a.kept}</strong> / {a.retrieved}
                <div style={{ marginTop: 2 }}>{a.model}</div>
              </div>
            </div>
          ))}
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
            “Overparameterization can improve generalization through implicit regularization, conditional on data structure and optimizer choice.”
          </div>
          <div style={{ borderTop: "1px solid var(--hair)", marginTop: 10, paddingTop: 8, fontSize: 11.5, color: "var(--muted)" }}>
            <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 10.5, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 4 }}>expected counterclaims</div>
            <div>· classical bias–variance still applies</div>
            <div>· width is a coarse proxy for capacity</div>
            <div>· benchmark contamination inflates results</div>
          </div>
        </div>
      </aside>
    </div>
  );
};

window.Results = Results;
