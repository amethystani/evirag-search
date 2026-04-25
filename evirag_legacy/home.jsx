/* global React, Icon, SAMPLE_QUERIES */
const { useState, useRef, useEffect } = React;

const Home = ({ onSubmit }) => {
  const [value, setValue] = useState("");
  const [mode, setMode] = useState("EVIRAG"); // EVIRAG | Vanilla
  const [agents, setAgents] = useState(true);
  const [vlm, setVlm] = useState(true);
  const [tab, setTab] = useState("Frontier");
  const inputRef = useRef(null);

  useEffect(() => { inputRef.current && inputRef.current.focus(); }, []);

  const submit = (q) => {
    const text = (q ?? value).trim();
    if (!text) return;
    onSubmit(text, { mode, agents, vlm });
  };

  const onKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const tabs = Object.keys(SAMPLE_QUERIES);
  const items = SAMPLE_QUERIES[tab] || [];

  return (
    <div className="home">
      <div className="wordmark fade-up">
        evirag<span className="dot"></span>
      </div>
      <div className="tagline fade-up d1">
        Scientific truth lives in the disagreement, not the consensus.
      </div>

      <div className="composer fade-up d2">
        <textarea
          ref={inputRef}
          className="composer-input"
          rows={1}
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            const el = e.target;
            el.style.height = "auto";
            el.style.height = Math.min(el.scrollHeight, 180) + "px";
          }}
          onKeyDown={onKey}
          placeholder="Ask a contested question — or paste a claim to verify…"
        />
        <div className="composer-row">
          <div className="composer-left">
            <button className="btn-plus" title="Attach"><Icon name="plus" size={16}/></button>
            <button
              className={"chip" + (agents ? " is-active" : "")}
              onClick={() => setAgents(a => !a)}
            >
              <span className="ico"><Icon name="cpu" size={13}/></span>
              4 agents
            </button>
            <button
              className={"chip" + (vlm ? " is-active" : "")}
              onClick={() => setVlm(v => !v)}
            >
              <span className="ico"><Icon name="beaker" size={13}/></span>
              Visual grounding
            </button>
            <button className="chip">
              <span className="ico"><Icon name="library" size={13}/></span>
              Corpus: lab/2026
            </button>
          </div>
          <div className="composer-right">
            <button className="model-pill">
              <span>{mode}</span>
              <Icon name="chevron-down" size={12}/>
            </button>
            <button className="icon-btn" title="Voice"><Icon name="mic" size={16}/></button>
            <button
              className="send-btn"
              onClick={() => submit()}
              disabled={!value.trim()}
              title="Submit inquiry"
            >
              <Icon name="arrow-up" size={16} stroke={2}/>
            </button>
          </div>
        </div>
      </div>

      <div className="suggest fade-up d3">
        <div className="suggest-head">
          <Icon name="sparkle" size={14}/>
          <span>Try a contested question</span>
        </div>
        <div className="suggest-tabs">
          {tabs.map(t => (
            <button
              key={t}
              className={"s-tab" + (t === tab ? " is-active" : "")}
              onClick={() => setTab(t)}
            >
              {t === "Frontier" && <Icon name="bolt" size={12}/>}
              {t === "Health" && <Icon name="beaker" size={12}/>}
              {t === "Climate" && <Icon name="compass" size={12}/>}
              {t === "Economics" && <Icon name="scale" size={12}/>}
              {t === "Method" && <Icon name="filter" size={12}/>}
              {t}
            </button>
          ))}
        </div>
        <div className="suggest-list">
          {items.map((it, i) => (
            <button key={i} className="s-item" onClick={() => submit(it.q)}>
              <div>
                <div>{it.q}</div>
                <div className="meta" style={{ marginTop: 4 }}>{it.meta}</div>
              </div>
              <span className="arr"><Icon name="arrow-right" size={14}/></span>
            </button>
          ))}
        </div>
      </div>

      <div className="foot-note fade-up d4">
        EVIRAG · epistemic-fidelity-first retrieval · phi3:mini · llama3 · qwen2.5 · deepseek-r1
      </div>
    </div>
  );
};

window.Home = Home;
