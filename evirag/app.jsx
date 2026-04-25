/* global React, ReactDOM, Sidebar, Home, Results, Icon, Pages, Modal, BACKEND_URL, queryBackend, transformResult, fetchUIBootstrap, fetchQueryTracePlan, Splash, Login, TopBar */
const { useState, useEffect, useRef } = React;

const AppInner = ({ onLogout }) => {
  const [view, setView] = useState("home"); // home | results | corpus | graph | agents | calibration | history | discover
  const [result, setResult] = useState(null);
  const [threads, setThreads] = useState([]);
  const [bootstrap, setBootstrap] = useState(null);
  const [bootstrapError, setBootstrapError] = useState("");
  const [sbOpen, setSbOpen] = useState(false);
  const [sbCollapsed, setSbCollapsed] = useState(false);
  const [modal, setModal] = useState(null); // { kind, data }
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [lastRequest, setLastRequest] = useState(null);
  const [tracePlan, setTracePlan] = useState(null);
  const requestSeq = useRef(0);

  const closeModal = () => setModal(null);
  const openPdf = (docId) => {
    if (!docId) return;
    window.open(`${BACKEND_URL}/api/corpus/documents/${docId}/pdf`, "_blank", "noopener");
  };

  useEffect(() => {
    let active = true;
    fetchUIBootstrap()
      .then((data) => {
        if (active) {
          setBootstrap(data);
          setBootstrapError("");
        }
      })
      .catch((err) => {
        if (active) setBootstrapError(err instanceof Error ? err.message : "Failed to load corpus data.");
      });
    return () => {
      active = false;
    };
  }, []);

  const classifyThreadLevel = (nextResult) => {
    const label = String(
      nextResult?.intent?.expectedDisagreement ||
      nextResult?.intent?.dispute ||
      nextResult?.epistemic?.controversyClass ||
      ""
    ).toLowerCase();
    if (label.includes("high")) return "high";
    if (label.includes("med") || label.includes("disputed") || label.includes("contested") || label.includes("narrow")) return "med";
    return "low";
  };

  const handleSubmit = async (q, options = {}) => {
    const query = String(q || "").trim();
    if (!query) return;

    const requestId = requestSeq.current + 1;
    requestSeq.current = requestId;
    setLastRequest({ query, options });
    setTracePlan(null);
    setError("");
    setLoading(true);
    setResult(null);
    setView("results");
    setSbOpen(false);
    window.scrollTo({ top: 0, behavior: "instant" });

    fetchQueryTracePlan(query, options)
      .then((plan) => {
        if (requestSeq.current === requestId) setTracePlan(plan);
      })
      .catch((err) => {
        if (requestSeq.current === requestId) {
          setTracePlan({
            mode_label: options.mode || "EVIRAG",
            path_label: "trace plan unavailable",
            steps: [{
              step: "query_execution",
              label: "Running backend query",
              detail: err instanceof Error ? err.message : "The trace-plan endpoint did not return before query execution."
            }]
          });
        }
      });

    try {
      const raw = await queryBackend(query, options);
      if (requestSeq.current !== requestId) return;
      const transformed = transformResult(raw, query);
      setResult(transformed);
      setThreads((prev) => {
        if (prev.some((t) => t.title === query)) return prev;
        return [{ id: "t" + Date.now(), title: query, time: "now", level: classifyThreadLevel(transformed) }, ...prev].slice(0, 6);
      });
    } catch (err) {
      if (requestSeq.current === requestId) {
        setError(err instanceof Error ? err.message : "Failed to reach the EVIRAG backend.");
      }
    } finally {
      if (requestSeq.current === requestId) setLoading(false);
    }
  };

  const goView = (v) => { setView(v); setSbOpen(false); window.scrollTo({ top: 0, behavior: "instant" }); };

  const handleNew = () => goView("home");
  const handlePickThread = (t) => handleSubmit(t.title);
  const openSidebar = () => {
    const isMobile = window.matchMedia && window.matchMedia("(max-width: 900px)").matches;
    setSbCollapsed(false);
    setSbOpen(!!isMobile);
  };
  const closeSidebar = () => { setSbOpen(false); setSbCollapsed(true); };

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
    <div className={"app" + (sbCollapsed ? " sb-collapsed" : "") + (sbOpen ? " sb-open" : "")} data-screen-label={screenLabel}>
      <button className="sb-reopen" onClick={openSidebar} title="Open sidebar">
        <Icon name="panel" size={16}/>
      </button>
      {sbOpen && <div className="sb-overlay" onClick={() => setSbOpen(false)}/>}
      <Sidebar
        active={view}
        onNew={handleNew}
        onGo={goView}
        threads={threads}
        stats={bootstrap && bootstrap.stats}
        onPickThread={handlePickThread}
        isOpen={sbOpen}
        onClose={closeSidebar}
        onLogout={onLogout}
      />
      <div className="main">
        {view === "home" && <Home onSubmit={handleSubmit} bootstrap={bootstrap} error={bootstrapError}/>}
        {view === "results" && (
          <Results
            result={result}
            loading={loading}
            error={error}
            pendingQuery={(result && result.query) || (lastRequest && lastRequest.query) || ""}
            pendingOptions={(lastRequest && lastRequest.options) || {}}
            tracePlan={tracePlan}
            onBack={handleNew}
            onRetry={() => lastRequest && handleSubmit(lastRequest.query, lastRequest.options)}
            onOpenSource={(s) => setModal({ kind: "source", data: s })}
            onOpenClaim={(c) => setModal({ kind: "claim", data: c })}
            onOpenAgent={(a) => setModal({ kind: "agent", data: a })}
            onOpenFigure={(f) => setModal({ kind: "figure", data: f })}
            onOpenMetric={(m) => setModal({ kind: "metric", data: m })}
            onOpenHypothesis={() => setModal({ kind: "hypothesis", data: result ? result.hypothesis : null })}
          />
        )}
        {view === "corpus" && <Pages.Corpus bootstrap={bootstrap} onOpenDoc={(d) => setModal({ kind: "doc", data: d })}/>}
        {view === "graph" && <Pages.Graph bootstrap={bootstrap} onOpen={(g) => handleSubmit(g.query || g.t || g.title)}/>}
        {view === "agents" && <Pages.Agents bootstrap={bootstrap} onOpen={(a) => setModal({ kind: "agent", data: a })}/>}
        {view === "calibration" && <Pages.Calibration bootstrap={bootstrap}/>}
        {view === "history" && <Pages.History threads={threads} onOpen={(t) => handleSubmit(t.title)}/>}
        {view === "discover" && <Pages.Discover bootstrap={bootstrap} onOpen={(it) => handleSubmit(it.q || it.t || it.title)}/>}
      </div>

      {modal && modal.kind === "source" && (
        <Modal
          open
          onClose={closeModal}
          title={modal.data.title}
          sub={[
            modal.data.venue,
            modal.data.year,
            modal.data.stance ? `stance: ${modal.data.stance}` : null
          ].filter(Boolean).join(" · ")}
        >
          <p>
            This source contributed {modal.data.claimCount || 0} extracted claims and was classified as
            {" "}
            {modal.data.stance === "support" ? "supporting" : modal.data.stance === "contra" ? "contradicting" : "neutral"}
            {" "}
            in the current view split.
          </p>
          <div className="kv">
            <span className="k">source rank</span><span className="v">#{modal.data.n}</span>
            <span className="k">claims extracted</span><span className="v">{modal.data.claimCount || 0} atomic</span>
            <span className="k">stance</span><span className="v">{modal.data.stance}</span>
            <span className="k">document id</span><span className="v">{modal.data.docId || modal.data.doc_id || "—"}</span>
          </div>
          <div style={{ marginTop: 12, padding: 12, background: "var(--bg)", borderRadius: 8, fontFamily: "Newsreader, serif", fontStyle: "italic", fontSize: 14, color: "var(--ink-2)", lineHeight: 1.6 }}>
            {modal.data.snippet || "No source snippet was returned for this document."}
          </div>
          <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
            <button className="chip" onClick={() => openPdf(modal.data.docId || modal.data.doc_id)}>Open PDF</button>
            <button className="chip" onClick={() => { closeModal(); handleSubmit(`Findings from ${modal.data.title}`); }}>Query source</button>
          </div>
        </Modal>
      )}

      {modal && modal.kind === "claim" && (
        <Modal open onClose={closeModal} title={modal.data.text} sub={`${modal.data.n} · view: ${modal.data.view || "—"} · source: ${modal.data.src}`}>
          <p>This atomic claim was returned by the active EVIRAG pipeline from the indexed corpus and is tied back to its source document metadata.</p>
          <div className="kv">
            <span className="k">stance</span><span className="v">{modal.data.kind === "dom" ? "supports dominant" : modal.data.kind === "alt" ? "supports alternative" : "minority/contrarian"}</span>
            <span className="k">cited in</span><span className="v">{(modal.data.cites || []).join(", ") || "—"}</span>
            <span className="k">claim id</span><span className="v">{modal.data.claimId || "—"}</span>
            <span className="k">document</span><span className="v">{modal.data.docTitle || modal.data.src || "—"}</span>
            <span className="k">verifier</span><span className="v">{
              (((result || {}).raw || {}).trace || {}).models &&
              ((((result.raw.trace.models.query_time_models || {}).nli_verifier) || (result.raw.trace.models.claim_graph_verification || {}).mode)
              || "claim graph")
            }</span>
          </div>
          {modal.data.context && (
            <div style={{ marginTop: 12, padding: 12, background: "var(--bg)", borderRadius: 8, fontFamily: "Newsreader, serif", fontStyle: "italic", fontSize: 14, color: "var(--ink-2)", lineHeight: 1.6 }}>
              {modal.data.context}
            </div>
          )}
          <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
            <button className="chip" onClick={() => handleSubmit(`What evidence supports this claim: ${modal.data.text}`)}>Query claim</button>
            <button className="chip" onClick={() => openPdf(modal.data.docId)}>Open source</button>
          </div>
        </Modal>
      )}

      {modal && modal.kind === "agent" && (
        <Modal open onClose={closeModal} title={modal.data.name + " agent"} sub={modal.data.role}>
          <p>The {modal.data.name} agent runs an independent retrieval pass with its own query-rewriting policy and verification threshold, then writes its kept evidence into the shared claim pool for cross-agent synthesis.</p>
          <div className="kv">
            <span className="k">model</span><span className="v">{modal.data.model}</span>
            <span className="k">status</span><span className="v">{modal.data.status || "configured"}</span>
            <span className="k">retrieved chunks</span><span className="v">{modal.data.retrieved ?? modal.data.retrieval_k ?? modal.data.k ?? "—"}</span>
            <span className="k">extracted claims</span><span className="v">{modal.data.kept ?? "—"}</span>
            <span className="k">stance</span><span className="v">{modal.data.stance || "—"}</span>
            <span className="k">confidence</span><span className="v">{typeof modal.data.confidence === "number" ? modal.data.confidence.toFixed(2) : "—"}</span>
            <span className="k">runtime</span><span className="v">{modal.data.durationMs ? `${(modal.data.durationMs / 1000).toFixed(1)}s` : "—"}</span>
            <span className="k">confidence gate</span><span className="v">{modal.data.confidence_threshold ?? "—"}</span>
            <span className="k">policy</span><span className="v">{modal.data.search_strategy || modal.data.retr || "—"}</span>
            <span className="k">temperature</span><span className="v">{modal.data.temperature ?? "—"}</span>
          </div>
          {modal.data.retrievalQuery && (
            <div style={{ marginTop: 12, padding: 12, background: "var(--bg)", borderRadius: 8, fontFamily: "JetBrains Mono, monospace", fontSize: 11, color: "var(--ink-2)", lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
{`RETRIEVAL QUERY
${modal.data.retrievalQuery}

AGENT REASONING
${modal.data.reasoning || "No agent reasoning returned."}`}
            </div>
          )}
          <div style={{ marginTop: 12, padding: 12, background: "var(--bg)", borderRadius: 8, fontFamily: "JetBrains Mono, monospace", fontSize: 11, color: "var(--ink-2)", lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
{`SYSTEM
You are the ${modal.data.name} agent in a deliberative RAG.
Objective: ${(modal.data.role || "").toLowerCase()}.
Reject claims below configured threshold ${modal.data.confidence_threshold ?? "—"}.
Always emit stance ∈ {support, contradict, neutral}.`}
          </div>
        </Modal>
      )}

      {modal && modal.kind === "figure" && (
        <Modal open onClose={closeModal} title={"Figure " + modal.data.i + " — " + modal.data.caption} sub={`paper ${modal.data.paper} · CLIP alignment ${modal.data.align}`}>
          <div style={{ height: 220, background: "repeating-linear-gradient(135deg, var(--panel), var(--panel) 8px, var(--panel-2) 8px, var(--panel-2) 16px)", borderRadius: 10, display: "grid", placeItems: "center", color: "var(--muted)", fontFamily: "JetBrains Mono, monospace", fontSize: 11 }}>
            visual evidence summary · {modal.data.caption}
          </div>
          <div className="kv" style={{ marginTop: 14 }}>
            <span className="k">visual–text align</span><span className="v">{modal.data.align}</span>
            <span className="k">supports claim</span><span className="v">{modal.data.claim || modal.data.caption}</span>
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
          
          {modal.data && (!modal.data.supportingClaims || modal.data.supportingClaims.length === 0) ? (
            <div style={{ marginTop: 12, padding: 12, background: "var(--bg)", borderRadius: 8, fontFamily: "Newsreader, serif", fontSize: 14, color: "var(--ink-2)", lineHeight: 1.6 }}>
              <strong>Fast Path Optimization</strong><br />
              Pre-retrieval LLM hypothesis generation was bypassed for this query to minimize latency. Switch to <strong>4 agents</strong> mode to enable the full deliberative planning step.
            </div>
          ) : (
            <div className="kv">
              <span className="k">central</span><span className="v">{(modal.data && modal.data.centralHypothesis) || "—"}</span>
              <span className="k">supporting</span><span className="v">{(modal.data && modal.data.supportingClaims && modal.data.supportingClaims.join(" · ")) || "—"}</span>
              <span className="k">assumptions</span><span className="v">{(modal.data && modal.data.assumptions && modal.data.assumptions.join(" · ")) || "—"}</span>
              <span className="k">expected counters</span><span className="v">{(modal.data && modal.data.expectedCounterclaims && modal.data.expectedCounterclaims.join(" · ")) || "—"}</span>
              <span className="k">verified</span><span className="v">{result ? `${result.metrics.claims} claims evaluated` : "—"}</span>
            </div>
          )}
        </Modal>
      )}

      {modal && modal.kind === "doc" && (
        <Modal open onClose={closeModal} title={modal.data.title} sub={`${modal.data.year || "n.d."} · ${modal.data.pages || 0} pages`}>
          <p>This paper contributes {modal.data.claims || 0} atomic claims to the corpus. Section-aware chunking produced {modal.data.chunks || 0} retrievable chunks and {modal.data.figures || 0} extracted figures.</p>
          <div className="kv">
            <span className="k">document id</span><span className="v">{modal.data.id}</span>
            <span className="k">filename</span><span className="v">{modal.data.filename || "—"}</span>
            <span className="k">chunks</span><span className="v">{modal.data.chunks || 0}</span>
            <span className="k">figures</span><span className="v">{modal.data.figures || 0}</span>
            <span className="k">edges</span><span className="v">{modal.data.edge_count || 0}</span>
          </div>
          {(modal.data.top_claims || []).length > 0 && (
            <div style={{ marginTop: 12 }}>
              {(modal.data.top_claims || []).map((claim) => (
                <div key={claim.claim_id} className="claim-row" style={{ gridTemplateColumns: "32px 1fr", padding: "8px 0" }}>
                  <div className="num">C</div>
                  <div>{claim.text}</div>
                </div>
              ))}
            </div>
          )}
          <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
            <button className="chip" onClick={() => openPdf(modal.data.id)}>Open PDF</button>
            <button className="chip" onClick={() => { closeModal(); handleSubmit("Findings from " + modal.data.title); }}>Query this paper</button>
          </div>
        </Modal>
      )}
    </div>
  );
};

const App = () => {
  const [authStage, setAuthStage] = useState(() => {
    try { return localStorage.getItem("evirag_authed") === "1" ? "app" : "splash"; } catch { return "splash"; }
  });

  if (authStage === "splash") {
    return <Splash onGetStarted={() => setAuthStage("login")} />;
  }
  if (authStage === "login") {
    return <Login onBack={() => setAuthStage("splash")} onLogin={() => { try { localStorage.setItem("evirag_authed", "1"); } catch {} setAuthStage("app"); }} />;
  }
  const handleLogout = () => {
    try { localStorage.removeItem("evirag_authed"); } catch {}
    setAuthStage("splash");
  };

  return <AppInner onLogout={handleLogout} />;
};

ReactDOM.createRoot(document.getElementById("root")).render(<App/>);
