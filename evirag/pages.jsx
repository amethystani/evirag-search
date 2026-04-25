/* global React, Icon */
const { useState } = React;

const LoadingPage = ({ title }) => (
  <div className="page">
    <h1>{title}</h1>
    <div className="lede">Loading live corpus data from the EVIRAG API.</div>
  </div>
);

const Pages = {
  Corpus: ({ bootstrap, onOpenDoc }) => {
    const [section, setSection] = useState("All");
    if (!bootstrap) return <LoadingPage title="Corpus"/>;

    const docs = bootstrap.documents || [];
    const stats = bootstrap.stats || {};
    const sectionCounts = {};
    docs.forEach((doc) => {
      (doc.sections || []).slice(0, 2).forEach((item) => {
        const name = item.name || "unknown";
        sectionCounts[name] = (sectionCounts[name] || 0) + item.chunks;
      });
    });
    const sections = ["All", ...Object.keys(sectionCounts).sort((a, b) => sectionCounts[b] - sectionCounts[a]).slice(0, 4)];
    const visibleDocs = section === "All"
      ? docs
      : docs.filter((doc) => (doc.sections || []).some((item) => item.name === section));

    return (
      <div className="page">
        <h1>Corpus</h1>
        <div className="lede">
          {Number(stats.total_documents || 0).toLocaleString()} papers indexed · {Number(stats.total_claims || 0).toLocaleString()} claims · {Number(stats.total_figures || 0).toLocaleString()} figures · FAISS index. Click any paper to inspect its claims, figures, and stance distribution.
        </div>
        <div style={{ display: "flex", gap: 8, marginBottom: 18 }}>
          {sections.map((name) => (
            <button
              key={name}
              className={"chip" + (section === name ? " is-active" : "")}
              onClick={() => setSection(name)}
            >
              {name === "All" ? "All" : name}
            </button>
          ))}
        </div>
        <div className="grid-2">
          {visibleDocs.map((d) => (
            <button key={d.id} className="tile" onClick={() => onOpenDoc(d)}>
              <h4>{d.title}</h4>
              <p>
                {d.year || "n.d."} · {d.pages || 0} pages · {d.claims || 0} extracted claims
              </p>
              <div className="meta">
                section_aware · {d.chunks || 0} chunks · {d.figures || 0} figures
              </div>
            </button>
          ))}
        </div>
      </div>
    );
  },

  Graph: ({ bootstrap, onOpen }) => {
    if (!bootstrap) return <LoadingPage title="Disagreement graph"/>;
    const graph = bootstrap.graph || {};
    const clusters = graph.clusters || [];
    return (
      <div className="page">
        <h1>Disagreement graph</h1>
        <div className="lede">
          A persistent meta-graph across the indexed corpus: {graph.nodes || 0} atomic claims and {graph.edges || 0} entailment edges classified by the claim graph.
        </div>
        <div className="grid-3">
          {clusters.map((g) => (
            <button key={g.doc_id} className="tile" onClick={() => onOpen(g)}>
              <h4>{g.title}</h4>
              <p>{g.claims} claims · {g.edges} edges · density {(g.density * 100).toFixed(1)}% · {g.class}</p>
              <div className="swatch-row">
                <span className="sw" style={{ background: "var(--ink)" }}/>
                <span className="sw" style={{ background: "var(--accent)" }}/>
                <span className="sw" style={{ background: "var(--muted-2)" }}/>
              </div>
            </button>
          ))}
        </div>
      </div>
    );
  },

  Agents: ({ bootstrap, onOpen }) => {
    if (!bootstrap) return <LoadingPage title="Agents"/>;
    const agents = bootstrap.agents || [];
    return (
      <div className="page">
        <h1>Agents</h1>
        <div className="lede">Four deliberative agents run through the configured retrieval objectives and local model settings exposed by the backend.</div>
        <div className="grid-2">
          {agents.map((a) => (
            <button key={a.key} className="tile" onClick={() => onOpen(a)} style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
              <div className="agent" style={{ padding: 0, border: 0 }}>
                <div className="glyph" style={{ width: 40, height: 40, fontSize: 20 }}>{a.glyph}</div>
              </div>
              <div style={{ flex: 1 }}>
                <h4 style={{ marginBottom: 2 }}>{a.name}</h4>
                <p>{a.role}</p>
                <div className="meta">{a.model} · {a.search_strategy} · k={a.retrieval_k}</div>
              </div>
            </button>
          ))}
        </div>
      </div>
    );
  },

  Calibration: ({ bootstrap }) => {
    if (!bootstrap) return <LoadingPage title="Calibration"/>;
    const factors = bootstrap.calibration || [];
    return (
      <div className="page">
        <h1>Calibration</h1>
        <div className="lede">Tune the weights that combine into final confidence. Values are loaded from the backend confidence configuration.</div>
        <div className="card" style={{ maxWidth: 560 }}>
          <h3>Confidence weights</h3>
          {factors.map((factor, i) => {
            const color = factor.key === "contradiction_severity" ? "contra" : factor.key === "source_diversity" ? "accent" : factor.key === "unverified_assumptions" ? "neutral" : "support";
            return (
              <div key={factor.key || i} className="bar-row" style={{ gridTemplateColumns: "150px 1fr 36px" }}>
                <span className="name">{factor.label}</span>
                <div className="track"><div className={"fill " + color} style={{ width: (factor.weight * 100)+"%" }}/></div>
                <span className="num">{Number(factor.weight).toFixed(2)}</span>
              </div>
            );
          })}
        </div>
      </div>
    );
  },

  History: ({ threads, onOpen }) => (
    <div className="page">
      <h1>History</h1>
      <div className="lede">All inquiries opened in this browser session are listed with their returned disagreement level.</div>
      <div className="grid-2">
        {threads.length ? threads.map(t => (
          <button key={t.id} className="tile" onClick={() => onOpen(t)}>
            <h4>{t.title}</h4>
            <p>{t.time} · {t.level === "high" ? "high disagreement" : t.level === "med" ? "contested" : "narrowing"}</p>
            <div className="meta">re-open inquiry →</div>
          </button>
        )) : (
          <div className="tile">
            <h4>No inquiries yet</h4>
            <p>Submit a corpus query to create the first history entry.</p>
          </div>
        )}
      </div>
    </div>
  ),

  Discover: ({ bootstrap, onOpen }) => {
    if (!bootstrap) return <LoadingPage title="Discover"/>;
    const tabs = ((bootstrap.topics || {}).tabs || []);
    const items = tabs.flatMap((tab) => (tab.items || []).map((item) => ({ ...item, tab: tab.label }))).slice(0, 8);
    return (
      <div className="page">
        <h1>Discover</h1>
        <div className="lede">Corpus-derived inquiries ranked from indexed documents, claims, and graph clusters.</div>
        <div className="grid-2">
          {items.map((it, i) => (
            <button key={it.claim_id || it.doc_id || i} className="tile" onClick={() => onOpen(it)}>
              <h4>{it.title || it.q}</h4>
              <p>{it.meta}</p>
            </button>
          ))}
        </div>
      </div>
    );
  }
};

window.Pages = Pages;
