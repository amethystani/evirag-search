/* global React, Icon */
const { useEffect } = React;

const Modal = ({ open, onClose, title, sub, children, wide }) => {
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal"
        style={wide ? { width: "min(960px, 100%)" } : undefined}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-head">
          <div>
            <h2>{title}</h2>
            {sub && <div className="sub">{sub}</div>}
          </div>
          <button className="modal-close" onClick={onClose} title="Close (Esc)">
            <Icon name="plus" size={18} stroke={1.6} className="rot45"/>
          </button>
        </div>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  );
};

window.Modal = Modal;
