/* global React, Icon */
const Sidebar = ({ active, onNew, onGo, onPickThread, threads, isOpen, onClose }) => {
  const items = [
    { id: "home", label: "New inquiry", icon: "plus" },
    { id: "discover", label: "Discover", icon: "compass" },
    { id: "corpus", label: "Corpus", icon: "library" },
    { id: "graph", label: "Disagreement graph", icon: "graph" },
    { id: "agents", label: "Agents", icon: "cpu" },
    { id: "calibration", label: "Calibration", icon: "settings" },
    { id: "history", label: "History", icon: "history" }
  ];
  return (
    <aside className={"sidebar" + (isOpen ? " is-open" : "")}>
      <div className="sb-top">
        <div className="sb-mark" aria-label="EVIRAG">
          <svg width="26" height="26" viewBox="0 0 32 32" fill="none">
            <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="1.5"/>
            <path d="M9 11h12M9 16h8M9 21h12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
            <circle cx="22" cy="16" r="2.2" fill="currentColor"/>
          </svg>
        </div>
        <button className="sb-collapse" title="Close" onClick={onClose}><Icon name="panel" size={16}/></button>
      </div>

      <nav className="sb-nav">
        {items.map(it => (
          <button
            key={it.id}
            className={"sb-item" + ((active === it.id || (it.id === "home" && active === "results")) ? " is-active" : "")}
            onClick={() => it.id === "home" ? onNew() : onGo(it.id)}
          >
            <span className="ico"><Icon name={it.icon} size={15}/></span>
            {it.label}
          </button>
        ))}
      </nav>

      <div className="sb-section">Recent threads</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 2, marginTop: 6, overflow: "auto", maxHeight: 320 }}>
        {threads.length === 0 ? (
          <div className="sb-thread-empty">No recent threads</div>
        ) : threads.map(t => (
          <button key={t.id} className="sb-thread" onClick={() => onPickThread(t)}>
            <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{t.title}</div>
            <div className="meta">
              <span className={"dot " + t.level}></span>
              <span>{t.time}</span>
              <span>·</span>
              <span>{t.level === "high" ? "high" : t.level === "med" ? "contested" : "narrowing"}</span>
            </div>
          </button>
        ))}
      </div>

      <div className="sb-bottom">
        <button className="sb-user" onClick={() => onGo("calibration")}>
          <div className="sb-avatar">EV</div>
          <div style={{ flex: 1, textAlign: "left" }}>
            <div className="name">Researcher</div>
            <div className="sub">Lab corpus · 1,284 papers</div>
          </div>
          <Icon name="chevron-right" size={14}/>
        </button>
      </div>
    </aside>
  );
};

window.Sidebar = Sidebar;
