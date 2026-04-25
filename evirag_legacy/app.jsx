/* global React, ReactDOM, Sidebar, Home, Results, Icon, RECENT_THREADS, DEMO_RESULT, Pages, Modal */
const { useState, useEffect } = React;

const TopBar = ({ tab, setTab, onMenu }) => {
  const tabs = ["Discover", "Frontier", "Health", "Climate", "Method"];
  return (
    <>
      <div className="mobile-bar">
        <button className="icon-btn" onClick={onMenu} title="Menu">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"><path d="M4 6h16M4 12h16M4 18h16"/></svg>
        </button>
        <div className="mark">evirag<span style={{ display: "inline-block", width: 6, height: 6, borderRadius: 3, background: "var(--accent)", marginLeft: 4, transform: "translateY(-6px)" }}/></div>
        <button className="icon-btn" title="Search"><Icon name="search" size={16}/></button>
      </div>
      <div className="topbar">
        <div className="tabs">
          {tabs.map(t => (
            <button
              key={t}
              className={"tab" + (t === tab ? " is-active" : "")}
              onClick={() => setTab(t)}
            >{t}</button>
          ))}
        </div>
        <div className="right">
          <button className="icon-btn" title="Search"><Icon name="search" size={16}/></button>
          <button className="icon-btn" title="About"><Icon name="info" size={16}/></button>
        </div>
      </div>
    </>
  );
};

const App = () => {
  const [authStage, setAuthStage] = useState(() => {
    try { return localStorage.getItem("evirag_authed") === "1" ? "app" : "splash"; } catch { return "splash"; }
  });
  const [view, setView] = useState("home"); // home | results | corpus | graph | agents | calibration | history | discover
  const [topTab, setTopTab] = useState("Discover");
  const [result, setResult] = useState(null);
  const [threads, setThreads] = useState(RECENT_THREADS);
  const [sbOpen, setSbOpen] = useState(false);
  const [modal, setModal] = useState(null); // { kind, data }

  const closeModal = () => setModal(null);

  const handleSubmit = (q) => {
    const r = { ...DEMO_RESULT, query: q };
    setResult(r);
    setView("results");
    setSbOpen(false);
    setThreads(prev => {
      if (prev.some(t => t.title === q)) return prev;
      return [{ id: "t" + Date.now(), title: q, time: "now", level: "high" }, ...prev].slice(0, 6);
    });
    window.scrollTo({ top: 0, behavior: "instant" });
  };

  const goView = (v) => { setView(v); setSbOpen(false); window.scrollTo({ top: 0, behavior: "instant" }); };

  // map top-tab clicks to category routes
  useEffect(() => {
    if (topTab === "Discover") return;
    // when user picks a top tab, switch to discover-like route filtered (kept simple)
    if (view === "home" || view === "results") return;
  }, [topTab]);

  const handleTopTab = (t) => {
    setTopTab(t);
    if (t === "Discover") goView("discover");
    else goView("home");
  };

  const handleNew = () => goView("home");
  const handlePickThread = (t) => handleSubmit(t.title);

  const screenLabel = (
    view === "home" ? "Home" :
    view === "results" ? "Results" :
    view === "corpus" ? "Corpus" :
    view === "graph" ? "Graph" :
    view === "agents" ? "Agents" :
    view === "calibration" ? "Calibration" :
    view === "history" ? "History" :
    view === "discover" ? "Discover" : "Page"
  );

  return (
    <div className="app" data-screen-label={screenLabel}>
      {authStage === "splash" && <Splash onGetStarted={() => setAuthStage("login")}/>}
      {authStage === "login" && <Login onBack={() => setAuthStage("splash")} onLogin={() => { try { localStorage.setItem("evirag_authed", "1"); } catch {} setAuthStage("app"); }}/>}
      {sbOpen && <div className="sb-overlay" onClick={() => setSbOpen(false)}/>}
      <Sidebar
        active={view}
        onNew={handleNew}
        onGo={goView}
        threads={threads}
        onPickThread={handlePickThread}
        isOpen={sbOpen}
        onClose={() => setSbOpen(false)}
      />
      <div className="main">
        <TopBar tab={topTab} setTab={handleTopTab} onMenu={() => setSbOpen(true)}/>
        {view === "home" && <Home onSubmit={handleSubmit}/>}
        {view === "results" && (
          <Results
            result={result}
            onBack={handleNew}
            onOpenSource={(s) => setModal({ kind: "source", data: s })}
            onOpenClaim={(c) => setModal({ kind: "claim", data: c })}
            onOpenAgent={(a) => setModal({ kind: "agent", data: a })}
            onOpenFigure={(f) => setModal({ kind: "figure", data: f })}
            onOpenMetric={(m) => setModal({ kind: "metric", data: m })}
            onOpenHypothesis={() => setModal({ kind: "hypothesis" })}
          />
        )}
        {view === "corpus" && <Pages.Corpus onOpenDoc={(d) => setModal({ kind: "doc", data: d })}/>}
        {view === "graph" && <Pages.Graph onOpen={(g) => handleSubmit(g.t)}/>}
        {view === "agents" && <Pages.Agents onOpen={(a) => setModal({ kind: "agent", data: a })}/>}
        {view === "calibration" && <Pages.Calibration/>}
        {view === "history" && <Pages.History threads={threads} onOpen={(t) => handleSubmit(t.title)}/>}
        {view === "discover" && <Pages.Discover onOpen={(it) => handleSubmit(it.t)}/>}
      </div>

      {modal && modal.kind === "source" && (
        <Modal open onClose={closeModal} title={modal.data.title} sub={`${modal.data.venue} · ${modal.data.year} · stance: ${modal.data.stance}`}>
          <p>This source was retrieved by 3 of 4 agents and contributed to {modal.data.stance === "support" ? "supporting" : modal.data.stance === "contra" ? "contradicting" : "neutral"} edges in the disagreement graph.</p>
          <div className="kv">
            <span className="k">retrieval rank</span><span className="v">#{modal.data.n} of 34</span>
            <span className="k">cosine score</span><span className="v">0.{(82 - modal.data.n).toString().padStart(2,"0")}</span>
            <span className="k">claims extracted</span><span className="v">{6 + modal.data.n} atomic</span>
            <span className="k">stance</span><span className="v">{modal.data.stance}</span>
            <span className="k">section</span><span className="v">Results §3.2, Discussion §4.1</span>
          </div>
          <div style={{ marginTop: 12, padding: 12, background: "var(--bg)", borderRadius: 8, fontFamily: "Newsreader, serif", fontStyle: "italic", fontSize: 14, color: "var(--ink-2)", lineHeight: 1.6 }}>
            “…the second-descent regime appears robust across width, depth, and label-noise levels, suggesting an implicit-regularization mechanism rather than capacity matching alone.”
          </div>
          <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
            <button className="chip">Open PDF</button>
            <button className="chip">Cite (BibTeX)</button>
            <button className="chip">View claims</button>
          </div>
        </Modal>
      )}

      {modal && modal.kind === "claim" && (
        <Modal open onClose={closeModal} title={modal.data.text} sub={`${modal.data.n} · view: ${modal.data.view || "—"} · source: ${modal.data.src}`}>
          <p>This atomic claim was extracted by the SLM filter (phi3:mini) and verified against retrieved evidence by the LLM extractor (llama3).</p>
          <div className="kv">
            <span className="k">stance</span><span className="v">{modal.data.kind === "dom" ? "supports dominant" : modal.data.kind === "alt" ? "supports alternative" : "minority/contrarian"}</span>
            <span className="k">cited in</span><span className="v">{(modal.data.cites || []).join(", ") || "—"}</span>
            <span className="k">edges</span><span className="v">3 support · 2 contradict</span>
            <span className="k">centrality</span><span className="v">0.42 (top-decile)</span>
            <span className="k">verifier</span><span className="v">llama3 · NLI head v2</span>
          </div>
          <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
            <button className="chip">Trace neighbors</button>
            <button className="chip">View source ¶</button>
          </div>
        </Modal>
      )}

      {modal && modal.kind === "agent" && (
        <Modal open onClose={closeModal} title={modal.data.name + " agent"} sub={modal.data.role}>
          <p>The {modal.data.name} agent runs an independent retrieval pass with its own query-rewriting policy and verification threshold, then writes its kept evidence into a shared claim pool for synthesis.</p>
          <div className="kv">
            <span className="k">model</span><span className="v">{modal.data.model}</span>
            <span className="k">retrieved</span><span className="v">{modal.data.retrieved || modal.data.k}</span>
            <span className="k">kept</span><span className="v">{modal.data.kept ?? Math.round((modal.data.k || 5) * 0.85)}</span>
            <span className="k">policy</span><span className="v">{modal.data.retr || "cosine + verifier gate"}</span>
            <span className="k">temperature</span><span className="v">{modal.data.name === "Skeptic" ? "0.4" : "0.2"}</span>
          </div>
          <div style={{ marginTop: 12, padding: 12, background: "var(--bg)", borderRadius: 8, fontFamily: "JetBrains Mono, monospace", fontSize: 11, color: "var(--ink-2)", lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
{`SYSTEM
You are the ${modal.data.name} agent in a deliberative RAG.
Objective: ${(modal.data.role || "").toLowerCase()}.
Reject claims with verifier score < 0.62.
Always emit stance ∈ {support, contradict, neutral}.`}
          </div>
        </Modal>
      )}

      {modal && modal.kind === "figure" && (
        <Modal open onClose={closeModal} title={"Figure " + modal.data.i + " — " + modal.data.caption} sub={`paper #${modal.data.paper} · CLIP alignment ${modal.data.align}`}>
          <div style={{ height: 220, background: "repeating-linear-gradient(135deg, var(--panel), var(--panel) 8px, var(--panel-2) 8px, var(--panel-2) 16px)", borderRadius: 10, display: "grid", placeItems: "center", color: "var(--muted)", fontFamily: "JetBrains Mono, monospace", fontSize: 11 }}>
            figure placeholder · {modal.data.caption}
          </div>
          <div className="kv" style={{ marginTop: 14 }}>
            <span className="k">visual–text align</span><span className="v">{modal.data.align}</span>
            <span className="k">supports claim</span><span className="v">C-09 (double-descent regime)</span>
            <span className="k">extracted via</span><span className="v">PyMuPDF · CLIP ViT-B/16</span>
          </div>
        </Modal>
      )}

      {modal && modal.kind === "metric" && (
        <Modal open onClose={closeModal} title={modal.data.label} sub={modal.data.formula}>
          <p>{modal.data.desc}</p>
          <div className="kv">
            <span className="k">current value</span><span className="v">{modal.data.value}</span>
            <span className="k">formula</span><span className="v">{modal.data.formula}</span>
            <span className="k">interpretation</span><span className="v">{modal.data.interp}</span>
          </div>
        </Modal>
      )}

      {modal && modal.kind === "hypothesis" && (
        <Modal open onClose={closeModal} title="X-IR hypothesis trace" sub="Explanation-first verification · pre-retrieval">
          <p>EVIRAG generates a structured hypothesis <em>before</em> retrieval, then evaluates retrieved evidence against it. Unverified assumptions become a confidence penalty.</p>
          <div className="kv">
            <span className="k">central</span><span className="v">Overparameterization can improve generalization through implicit regularization.</span>
            <span className="k">supporting</span><span className="v">double-descent · NTK · low-norm bias</span>
            <span className="k">assumptions</span><span className="v">SGD dynamics · structured data · interpolation reachable</span>
            <span className="k">expected counters</span><span className="v">classical bias–variance · width is coarse proxy · benchmark contamination</span>
            <span className="k">verified</span><span className="v">4 of 7 supporting · 2 of 3 counters</span>
          </div>
        </Modal>
      )}

      {modal && modal.kind === "doc" && (
        <Modal open onClose={closeModal} title={modal.data.title} sub={`${modal.data.venue} · ${modal.data.year} · ${modal.data.pages} pages`}>
          <p>This paper contributes {modal.data.claims} atomic claims to the corpus. Section-aware chunking preserved 4 logical sections; CLIP aligned 6 figures.</p>
          <div className="kv">
            <span className="k">indexed</span><span className="v">2 days ago</span>
            <span className="k">embedding</span><span className="v">MiniLM-L6-v2 · 384d</span>
            <span className="k">chunks</span><span className="v">{Math.round(modal.data.pages * 4.2)}</span>
            <span className="k">figures</span><span className="v">6 aligned · 2 unmatched</span>
          </div>
          <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
            <button className="chip">Re-index</button>
            <button className="chip">View claims</button>
            <button className="chip" onClick={() => { closeModal(); handleSubmit("Findings from " + modal.data.title); }}>Query this paper</button>
          </div>
        </Modal>
      )}
    </div>
  );
};

ReactDOM.createRoot(document.getElementById("root")).render(<App/>);
