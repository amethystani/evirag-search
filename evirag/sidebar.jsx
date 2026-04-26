/* global React, Icon */
// ── EVIRAG · sidebar.jsx ───────────────────────────────────────────────────────
// Persistent sidebar: nav, per-user session history from Supabase, user profile.
// ──────────────────────────────────────────────────────────────────────────────
const { useState, useEffect, useCallback } = React;

// ── Time formatting ────────────────────────────────────────────────────────────
function timeAgo(iso) {
  if (!iso) return "";
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1)  return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  if (d < 7)  return `${d}d ago`;
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

// ── User Avatar ────────────────────────────────────────────────────────────────
const Avatar = ({ user, size = 28 }) => {
  const initials = user
    ? ((user.firstName || "")[0] || "") + ((user.lastName || "")[0] || (user.firstName || "")[1] || "")
    : "?";
  if (user && user.imageUrl) {
    return (
      <img
        src={user.imageUrl} alt={user.fullName || "User"}
        style={{ width: size, height: size, borderRadius: "50%", objectFit: "cover", flexShrink: 0 }}
      />
    );
  }
  return (
    <div style={{
      width: size, height: size, borderRadius: "50%",
      background: "var(--ink)", color: "var(--bg)",
      display: "flex", alignItems: "center", justifyContent: "center",
      fontSize: size * 0.38, fontWeight: 600, flexShrink: 0,
      fontFamily: "Inter, sans-serif", letterSpacing: 0,
    }}>
      {initials.toUpperCase() || "?"}
    </div>
  );
};

// ── Sidebar ────────────────────────────────────────────────────────────────────
const Sidebar = ({
  active, onNew, onGo, onPickThread, onPickSession,
  threads, stats, isOpen, onClose, onLogout,
  user, userSessions, onDeleteSession, onRefreshSessions,
}) => {
  const items = [
    { id: "home",        label: "New inquiry",        icon: "plus"    },
    { id: "discover",    label: "Discover",            icon: "compass" },
    { id: "corpus",      label: "Corpus",              icon: "library" },
    { id: "graph",       label: "Disagreement graph",  icon: "graph"   },
    { id: "agents",      label: "Agents",              icon: "cpu"     },
    { id: "calibration", label: "Calibration",         icon: "settings"},
    { id: "history",     label: "History",             icon: "history" },
    { id: "profile",     label: "Profile",             icon: "user"    },
  ];

  const sessions = userSessions || [];
  const displayName = user
    ? (user.fullName || user.firstName || user.primaryEmailAddress?.emailAddress || "Researcher")
    : null;
  const emailDisplay = user?.primaryEmailAddress?.emailAddress;

  return (
    <aside className={"sidebar" + (isOpen ? " is-open" : "")}>
      {/* Top close button */}
      <div className="sb-top" style={{ justifyContent: "flex-end", height: 40, alignItems: "center", display: "flex", paddingRight: 12 }}>
        <button className="sb-collapse" title="Close" onClick={onClose}>
          <Icon name="panel" size={16}/>
        </button>
      </div>

      {/* Navigation */}
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

      {/* Persistent sessions from Supabase */}
      <div className="sb-section" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", paddingRight: 8 }}>
        <span>Recent sessions</span>
        {user && (
          <button
            title="Refresh" onClick={onRefreshSessions}
            style={{ all: "unset", cursor: "pointer", color: "var(--muted)", fontSize: 11 }}>
            ↻
          </button>
        )}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 2, overflow: "auto", maxHeight: 260 }}>
        {!user ? (
          <div className="sb-thread-empty">Sign in to see saved sessions</div>
        ) : sessions.length === 0 ? (
          <div className="sb-thread-empty">No sessions yet — start a query!</div>
        ) : sessions.map(s => (
          <div key={s.id} className="sb-thread" style={{ position: "relative", paddingRight: 28 }}
            onClick={() => onPickSession && onPickSession(s)}>
            <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontSize: 12, fontWeight: 500 }}>
              {s.title}
            </div>
            <div className="meta">
              <span style={{ fontSize: 10 }}>{timeAgo(s.updated_at)}</span>
            </div>
            {/* Delete button */}
            <button
              title="Delete session"
              onClick={e => { e.stopPropagation(); onDeleteSession && onDeleteSession(s.id); }}
              style={{
                all: "unset", cursor: "pointer",
                position: "absolute", right: 6, top: "50%", transform: "translateY(-50%)",
                color: "var(--muted)", fontSize: 14, lineHeight: 1,
                padding: "2px 4px", borderRadius: 4,
              }}
              onMouseEnter={e => e.currentTarget.style.color = "var(--ink)"}
              onMouseLeave={e => e.currentTarget.style.color = "var(--muted)"}
            >
              ×
            </button>
          </div>
        ))}
      </div>

      {/* User profile bottom */}
      <div className="sb-bottom" style={{ borderTop: "1px solid var(--border)", paddingTop: 10 }}>
        {user ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "4px 10px", cursor: "pointer" }}
              onClick={() => onGo("profile")}>
              <Avatar user={user} size={28}/>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 12, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {displayName}
                </div>
                {emailDisplay && (
                  <div style={{ fontSize: 10, color: "var(--muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {emailDisplay}
                  </div>
                )}
              </div>
            </div>
            <button className="sb-item" onClick={onLogout} style={{ color: "var(--muted)", fontSize: 12 }}>
              <Icon name="arrow-left" size={14}/>
              Sign out
            </button>
          </div>
        ) : (
          <button className="sb-item" onClick={onLogout} style={{ color: "var(--muted)" }}>
            <Icon name="arrow-left" size={16}/>
            Log out
          </button>
        )}
      </div>
    </aside>
  );
};

window.Sidebar = Sidebar;
window.Avatar  = Avatar;
window.timeAgo = timeAgo;
