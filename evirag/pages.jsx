/* global React, Icon */
const { useState, useEffect } = React;

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

  // ── History: persistent per-user sessions from Supabase ──────────────────
  History: ({ user, userSessions, onResumeSession, threads, onOpen }) => {
    const sessions = userSessions || [];
    const hasUser  = !!user;
    return (
      <div className="page">
        <h1>History</h1>
        <div className="lede">
          {hasUser
            ? `${sessions.length} saved session${sessions.length !== 1 ? "s" : ""} across all devices — click any to resume.`
            : "Sign in to save and access history across devices. Showing this browser session only."}
        </div>

        {/* Supabase persistent sessions */}
        {hasUser && sessions.length > 0 && (
          <>
            <h3 style={{ fontSize: 13, color: "var(--muted)", marginBottom: 10, fontFamily: "Inter, sans-serif" }}>
              SAVED SESSIONS
            </h3>
            <div className="grid-2" style={{ marginBottom: 24 }}>
              {sessions.map(s => (
                <button key={s.id} className="tile" onClick={() => onResumeSession && onResumeSession(s)}>
                  <h4 style={{ fontSize: 14 }}>{s.title}</h4>
                  <p style={{ fontSize: 12 }}>
                    {window.timeAgo ? window.timeAgo(s.updated_at) : s.updated_at?.slice(0, 10)}
                  </p>
                  <div className="meta">resume session →</div>
                </button>
              ))}
            </div>
          </>
        )}

        {/* Browser-session threads (always shown) */}
        {threads.length > 0 && (
          <>
            <h3 style={{ fontSize: 13, color: "var(--muted)", marginBottom: 10, fontFamily: "Inter, sans-serif" }}>
              THIS SESSION
            </h3>
            <div className="grid-2">
              {threads.map(t => (
                <button key={t.id} className="tile" onClick={() => onOpen(t)}>
                  <h4>{t.title}</h4>
                  <p>{t.time} · {t.level === "high" ? "high disagreement" : t.level === "med" ? "contested" : "narrowing"}</p>
                  <div className="meta">re-open inquiry →</div>
                </button>
              ))}
            </div>
          </>
        )}

        {!threads.length && (!hasUser || !sessions.length) && (
          <div className="tile">
            <h4>No inquiries yet</h4>
            <p>Submit a corpus query to create the first history entry.</p>
          </div>
        )}
      </div>
    );
  },

  // ── Discover: popular queries from Supabase, grouped by topic ────────────
  Discover: ({ bootstrap, onOpen }) => {
    const [popular, setPopular] = useState([]);
    const [activeTab, setActiveTab] = useState("trending");

    useEffect(() => {
      if (window.EVIRAG_DB) {
        window.EVIRAG_DB.getPopularQueries(40).then(setPopular).catch(() => {});
      }
    }, []);

    const CATEGORY_LABELS = {
      biology: "🧬 Biology", neuroscience: "🧠 Neuroscience",
      physics: "⚛️ Physics", chemistry: "🧪 Chemistry",
      climate: "🌍 Climate", ai: "🤖 AI & ML",
      medicine: "💊 Medicine", general: "🔬 General",
    };

    const fallbackItems = ((bootstrap?.topics?.tabs || [])
      .flatMap(t => (t.items || []).map(i => ({ ...i, category: "general" }))).slice(0, 12))
      .map(i => ({ query: i.title || i.q, category: "general", count: 1 }));

    const items = popular.length ? popular : fallbackItems;

    // Group by category
    const byCategory = {};
    items.forEach(item => {
      const cat = item.category || "general";
      if (!byCategory[cat]) byCategory[cat] = [];
      byCategory[cat].push(item);
    });
    const categories = Object.keys(byCategory).sort((a, b) => byCategory[b].length - byCategory[a].length);

    const recentItems = [...items].sort((a, b) => new Date(b.last_seen || 0) - new Date(a.last_seen || 0)).slice(0, 12);
    const trendingItems = items.slice(0, 12);

    const displayItems = activeTab === "trending" ? trendingItems
      : activeTab === "recent" ? recentItems
      : (byCategory[activeTab] || []);

    return (
      <div className="page">
        <h1>Discover</h1>
        <div className="lede">
          Popular queries from the EVIRAG community — ranked by frequency and grouped by research area.
          {popular.length === 0 && " Start querying to populate this page."}
        </div>

        {/* Tab strip */}
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 18 }}>
          {["trending", "recent", ...categories].map(tab => (
            <button
              key={tab}
              className={"chip" + (activeTab === tab ? " is-active" : "")}
              onClick={() => setActiveTab(tab)}
            >
              {tab === "trending" ? "🔥 Trending" :
               tab === "recent"   ? "🕐 Recent"   :
               (CATEGORY_LABELS[tab] || tab)}
            </button>
          ))}
        </div>

        <div className="grid-2">
          {displayItems.length ? displayItems.map((it, i) => (
            <button key={it.query + i} className="tile" onClick={() => onOpen({ q: it.query, title: it.query })}>
              <h4 style={{ fontSize: 14 }}>{it.query}</h4>
              <p style={{ fontSize: 12 }}>
                {CATEGORY_LABELS[it.category] || it.category}
                {it.count > 1 ? ` · ${it.count} searches` : ""}
              </p>
              <div className="meta">explore →</div>
            </button>
          )) : (
            <div className="tile">
              <h4>No queries yet</h4>
              <p>Be the first to explore this topic.</p>
            </div>
          )}
        </div>
      </div>
    );
  },

  // ── Profile page ──────────────────────────────────────────────────────────
  Profile: ({ user, onSubmit, onGo }) => {
    const [stats, setStats]   = useState(null);
    const [claims, setClaims] = useState([]);

    useEffect(() => {
      if (user && window.EVIRAG_DB) {
        window.EVIRAG_DB.getUserStats(user.id).then(setStats).catch(() => {});
        window.EVIRAG_DB.getUserClaims(user.id, 10).then(setClaims).catch(() => {});
      }
    }, [user]);

    if (!user) return (
      <div className="page">
        <h1>Profile</h1>
        <div className="lede">Sign in to view your profile and research stats.</div>
      </div>
    );

    const displayName = user.fullName || user.firstName || "Researcher";
    const email = user.primaryEmailAddress?.emailAddress || "";
    const joinedDate = user.createdAt
      ? new Date(user.createdAt).toLocaleDateString(undefined, { year: "numeric", month: "long" })
      : null;

    return (
      <div className="page">
        <h1>Profile</h1>

        {/* Identity card */}
        <div className="card" style={{ display: "flex", alignItems: "center", gap: 18, marginBottom: 24, padding: 20 }}>
          {window.Avatar && <window.Avatar user={user} size={56}/>}
          <div>
            <div style={{ fontSize: 20, fontFamily: "Newsreader, serif", fontWeight: 600 }}>{displayName}</div>
            {email && <div style={{ fontSize: 13, color: "var(--muted)", marginTop: 2 }}>{email}</div>}
            {joinedDate && <div style={{ fontSize: 11, color: "var(--muted-2)", marginTop: 4 }}>Member since {joinedDate}</div>}
          </div>
        </div>

        {/* Stats row */}
        {stats && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginBottom: 24 }}>
            {[
              { label: "Sessions",         value: stats.sessions },
              { label: "Queries asked",    value: stats.turns    },
              { label: "Claims discovered", value: stats.claims  },
            ].map(({ label, value }) => (
              <div key={label} className="card" style={{ textAlign: "center", padding: "14px 8px" }}>
                <div style={{ fontSize: 28, fontFamily: "Newsreader, serif", fontWeight: 600 }}>{value}</div>
                <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>{label}</div>
              </div>
            ))}
          </div>
        )}

        {/* Recent claims */}
        {claims.length > 0 && (
          <>
            <h3 style={{ fontSize: 13, color: "var(--muted)", marginBottom: 10, fontFamily: "Inter, sans-serif" }}>
              YOUR RECENT CLAIMS
            </h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 24 }}>
              {claims.map((c, i) => (
                <div key={i} className="card" style={{ padding: "10px 14px" }}>
                  <div style={{ fontSize: 13, fontFamily: "Newsreader, serif", fontStyle: "italic", color: "var(--ink-2)", lineHeight: 1.5 }}>
                    {c.claim_text}
                  </div>
                  <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 4 }}>
                    Confidence {(c.confidence * 100).toFixed(0)}% ·{" "}
                    {window.timeAgo ? window.timeAgo(c.created_at) : c.created_at?.slice(0, 10)}
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {/* Actions */}
        <div style={{ display: "flex", gap: 10 }}>
          <button className="chip" onClick={() => onGo("history")}>View all sessions</button>
          <button className="chip" onClick={() => onGo("discover")}>Explore Discover</button>
        </div>
      </div>
    );
  },
};

window.Pages = Pages;
