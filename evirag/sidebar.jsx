/* global React, Icon */
const Sidebar = ({ active, onNew, onGo, onPickThread, threads, stats, isOpen, onClose, onLogout }) => {
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
      <div className="sb-top" style={{ justifyContent: "flex-end", height: 40, alignItems: "center", display: "flex", paddingRight: 12 }}>
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
        <button className="sb-item" onClick={onLogout} style={{ color: "var(--muted)" }}>
          <Icon name="arrow-left" size={16}/>
          Log out
        </button>
      </div>

    </aside>
  );
};

window.Sidebar = Sidebar;
