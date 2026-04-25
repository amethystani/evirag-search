/* global React, Icon */

const Pages = {
  Corpus: ({ onOpenDoc }) => {
    const docs = [
      { id: 1, title: "Reconciling modern ML practice and the bias–variance trade-off", venue: "PNAS", year: 2019, pages: 6, claims: 12 },
      { id: 2, title: "Deep double descent: where bigger models and more data hurt", venue: "ICLR", year: 2020, pages: 24, claims: 18 },
      { id: 3, title: "Neural tangent kernel: convergence and generalization", venue: "NeurIPS", year: 2018, pages: 14, claims: 9 },
      { id: 4, title: "ResNet strikes back: improved training in timm", venue: "arXiv", year: 2022, pages: 22, claims: 11 },
      { id: 5, title: "ConViT: soft convolutional inductive biases for ViTs", venue: "ICML", year: 2021, pages: 18, claims: 8 },
      { id: 6, title: "Do ImageNet classifiers generalize to ImageNet?", venue: "ICML", year: 2019, pages: 16, claims: 14 },
      { id: 7, title: "On width and implicit regularization in deep linear nets", venue: "JMLR", year: 2023, pages: 32, claims: 21 },
      { id: 8, title: "Statins for primary prevention: USPSTF evidence review", venue: "JAMA", year: 2022, pages: 28, claims: 19 }
    ];
    return (
      <div className="page">
        <h1>Corpus</h1>
        <div className="lede">1,284 papers indexed · section-aware chunking · MiniLM embeddings · FAISS index. Click any paper to inspect its claims, figures, and stance distribution.</div>
        <div style={{ display: "flex", gap: 8, marginBottom: 18 }}>
          <button className="chip is-active">All</button>
          <button className="chip">ML / AI</button>
          <button className="chip">Health</button>
          <button className="chip">Climate</button>
          <button className="chip">Method</button>
        </div>
        <div className="grid-2">
          {docs.map(d => (
            <button key={d.id} className="tile" onClick={() => onOpenDoc(d)}>
              <h4>{d.title}</h4>
              <p>{d.venue} · {d.year} · {d.pages} pages · {d.claims} extracted claims</p>
              <div className="meta">section_aware · {Math.round(d.pages * 4.2)} chunks · clip-aligned</div>
            </button>
          ))}
        </div>
      </div>
    );
  },
  Graph: ({ onOpen }) => (
    <div className="page">
      <h1>Disagreement graph</h1>
      <div className="lede">A persistent meta-graph across all your inquiries. Each node is an atomic claim; edges are entailment relations classified by an NLI head. Open any cluster to drill into its sources.</div>
      <div className="grid-3">
        {[
          { t: "Overparameterization & generalization", n: 47, d: 0.183, c: "stable" },
          { t: "Statins primary prevention", n: 41, d: 0.221, c: "narrowing" },
          { t: "Climate sensitivity 2× CO₂", n: 29, d: 0.094, c: "narrowing" },
          { t: "Minimum wage employment", n: 64, d: 0.276, c: "stable" },
          { t: "p < 0.005 thresholds", n: 19, d: 0.140, c: "open" },
          { t: "Mech-interp faithfulness", n: 24, d: 0.198, c: "open" }
        ].map((g, i) => (
          <button key={i} className="tile" onClick={() => onOpen(g)}>
            <h4>{g.t}</h4>
            <p>{g.n} claims · density {(g.d*100).toFixed(1)}% · {g.c}</p>
            <div className="swatch-row">
              <span className="sw" style={{ background: "var(--ink)" }}/>
              <span className="sw" style={{ background: "var(--accent)" }}/>
              <span className="sw" style={{ background: "var(--muted-2)" }}/>
            </div>
          </button>
        ))}
      </div>
    </div>
  ),
  Agents: ({ onOpen }) => {
    const agents = [
      { name: "Precision", role: "Strict semantic · high-confidence support", model: "llama3:latest", k: 5, retr: "cosine ≥ 0.78", glyph: "P" },
      { name: "Recall", role: "Expansive semantic · broad coverage", model: "qwen2.5:4b", k: 15, retr: "MMR λ=0.3", glyph: "R" },
      { name: "Skeptic", role: "Adversarial · contradiction-seeking", model: "llama3:latest", k: 8, retr: "neg-augmented query", glyph: "S" },
      { name: "Counterfactual", role: "Alternative explanations", model: "deepseek-r1:7b", k: 6, retr: "hyp. perturb + retrieve", glyph: "C" }
    ];
    return (
      <div className="page">
        <h1>Agents</h1>
        <div className="lede">Four deliberative agents run in parallel, each with its own retrieval objective and verification policy. Tap one to inspect its prompt, model, and last-call traces.</div>
        <div className="grid-2">
          {agents.map((a, i) => (
            <button key={i} className="tile" onClick={() => onOpen(a)} style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
              <div className="agent" style={{ padding: 0, border: 0 }}>
                <div className="glyph" style={{ width: 40, height: 40, fontSize: 20 }}>{a.glyph}</div>
              </div>
              <div style={{ flex: 1 }}>
                <h4 style={{ marginBottom: 2 }}>{a.name}</h4>
                <p>{a.role}</p>
                <div className="meta">{a.model} · {a.retr} · k={a.k}</div>
              </div>
            </button>
          ))}
        </div>
      </div>
    );
  },
  Calibration: () => (
    <div className="page">
      <h1>Calibration</h1>
      <div className="lede">Tune the weights that combine into final confidence. Defaults from EVIRAG paper §4.3.</div>
      <div className="card" style={{ maxWidth: 560 }}>
        <h3>Confidence weights</h3>
        {[
          ["Claim agreement", 0.30, "support"],
          ["Source diversity", 0.20, "accent"],
          ["Contradiction severity", 0.25, "contra"],
          ["Unverified assumptions", 0.15, "neutral"],
          ["Visual alignment", 0.10, "support"]
        ].map(([k, v, c], i) => (
          <div key={i} className="bar-row" style={{ gridTemplateColumns: "150px 1fr 36px" }}>
            <span className="name">{k}</span>
            <div className="track"><div className={"fill " + c} style={{ width: (v*100)+"%" }}/></div>
            <span className="num">{v.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  ),
  History: ({ threads, onOpen }) => (
    <div className="page">
      <h1>History</h1>
      <div className="lede">All inquiries are stored locally with their full claim graph and agent traces.</div>
      <div className="grid-2">
        {threads.map(t => (
          <button key={t.id} className="tile" onClick={() => onOpen(t)}>
            <h4>{t.title}</h4>
            <p>{t.time} · {t.level === "high" ? "high disagreement" : t.level === "med" ? "contested" : "narrowing"}</p>
            <div className="meta">re-open inquiry →</div>
          </button>
        ))}
      </div>
    </div>
  ),
  Discover: ({ onOpen }) => (
    <div className="page">
      <h1>Discover</h1>
      <div className="lede">Curated controversies from the field, ranked by claim entropy.</div>
      <div className="grid-2">
        {[
          { t: "Is overparameterization necessary for generalization?", d: "ML · 47 claims · density 18%", level: "high" },
          { t: "Statins for primary prevention — mortality endpoint", d: "Cardiology · 41 claims · density 22%", level: "high" },
          { t: "Equilibrium climate sensitivity to 2× CO₂", d: "Climate · 29 claims · narrowing", level: "med" },
          { t: "Replication rates in social priming", d: "Meta-sci · 27 claims · disputed", level: "med" }
        ].map((it, i) => (
          <button key={i} className="tile" onClick={() => onOpen(it)}>
            <h4>{it.t}</h4>
            <p>{it.d}</p>
          </button>
        ))}
      </div>
    </div>
  )
};

window.Pages = Pages;
