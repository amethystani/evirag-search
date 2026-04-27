/* global React, Icon */
const { useState, useRef, useEffect } = React;

// ── PDF text extraction using PDF.js (loaded from CDN) ───────────────────────
const extractPdfText = async (file) => {
  const lib = window.pdfjsLib;
  if (!lib) {
    console.warn("[pdf] pdfjs-dist not loaded yet");
    return "";
  }
  // Point worker to the same CDN version
  lib.GlobalWorkerOptions.workerSrc =
    "https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js";
  try {
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await lib.getDocument({ data: arrayBuffer }).promise;
    let text = "";
    const maxPages = Math.min(pdf.numPages, 30);
    for (let i = 1; i <= maxPages; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      text += content.items.map((item) => item.str).join(" ") + "\n";
    }
    return text.trim().slice(0, 20000);
  } catch (err) {
    console.error("[pdf] extraction error:", err);
    return "";
  }
};

const Home = ({ onSubmit, bootstrap, agents, setAgents }) => {
  const [value, setValue] = useState("");
  const [mode, setMode] = useState("EVIRAG"); // EVIRAG | Vanilla
  const [tab, setTab] = useState("");
  const [modeOpen, setModeOpen] = useState(false);
  const [suggestCollapsed, setSuggestCollapsed] = useState(false);
  const [attachOpen, setAttachOpen] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  // PDF state
  const [pdfName, setPdfName] = useState("");
  const [pdfText, setPdfText] = useState("");
  const [pdfLoading, setPdfLoading] = useState(false);

  const inputRef    = useRef(null);
  const modeRef     = useRef(null);
  const attachRef   = useRef(null);
  const fileInputRef= useRef(null);
  const recognitionRef = useRef(null);

  const topicTabs = (bootstrap && bootstrap.topics && bootstrap.topics.tabs) || [];
  const activeTab = topicTabs.find((t) => t.id === tab) || topicTabs[0];
  const items = activeTab ? activeTab.items : [];
  const stats = bootstrap && bootstrap.stats;

  useEffect(() => { inputRef.current && inputRef.current.focus(); }, []);
  useEffect(() => {
    if (!tab && topicTabs.length) setTab(topicTabs[0].id);
  }, [tab, topicTabs]);

  useEffect(() => {
    if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.onresult = (event) => {
        let finalTranscript = "";
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          }
        }
        if (finalTranscript) {
          setValue((prev) => (prev ? prev + " " : "") + finalTranscript);
        }
      };
      recognition.onerror = (e) => {
        console.error("Speech recognition error:", e);
        setIsRecording(false);
      };
      recognition.onend = () => setIsRecording(false);
      recognitionRef.current = recognition;
    }

    const onPointerDown = (event) => {
      if (modeRef.current && !modeRef.current.contains(event.target)) setModeOpen(false);
      if (attachRef.current && !attachRef.current.contains(event.target)) setAttachOpen(false);
    };
    window.addEventListener("pointerdown", onPointerDown);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
      if (recognitionRef.current) recognitionRef.current.stop();
    };
  }, []);

  const toggleMic = () => {
    if (!recognitionRef.current) {
      alert("Speech recognition is not supported in this browser.");
      return;
    }
    if (isRecording) {
      recognitionRef.current.stop();
    } else {
      try { recognitionRef.current.start(); setIsRecording(true); }
      catch (e) { console.error(e); }
    }
  };

  // ── PDF upload handler ──────────────────────────────────────────────────────
  const handleFileChange = async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    setAttachOpen(false);
    setPdfLoading(true);
    setPdfName(file.name);
    const text = await extractPdfText(file);
    setPdfText(text);
    setPdfLoading(false);
    // Reset file input so same file can be re-uploaded if needed
    e.target.value = "";
    if (!text) {
      alert("Could not extract text from this file. Make sure it is a text-based PDF.");
      setPdfName("");
    }
  };

  const clearPdf = () => { setPdfName(""); setPdfText(""); };

  // ── Submit — appends PDF context if a doc is loaded ────────────────────────
  const submit = (q) => {
    const text = (q ?? value).trim();
    if (!text) return;
    // Visual grounding is always ON; pass vlm=true unconditionally
    onSubmit(text, { mode, agents, vlm: true, pdfContext: pdfText || undefined });
  };

  const onKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
  };

  const corpusLabel = stats
    ? `${stats.total_documents} docs · ${stats.total_claims} claims`
    : "Loading corpus";

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

            {/* ── Attach / PDF upload ──────────────────────────────────── */}
            <div className="mode-wrap" ref={attachRef}>
              <button
                className="btn-plus"
                title="Upload PDF"
                onClick={() => setAttachOpen(!attachOpen)}
              >
                <Icon name="plus" size={16}/>
              </button>
              {attachOpen && (
                <div className="mode-menu" style={{ left: 0, right: "auto" }}>
                  <button
                    onClick={() => { setAttachOpen(false); fileInputRef.current && fileInputRef.current.click(); }}
                    style={{ display: "flex", alignItems: "center", gap: 8 }}
                  >
                    <Icon name="library" size={13}/>
                    Upload PDF
                  </button>
                </div>
              )}
            </div>

            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.txt"
              style={{ display: "none" }}
              onChange={handleFileChange}
            />

            {/* 4-agents chip */}
            <button
              className={"chip" + (agents ? " is-active" : "")}
              onClick={() => setAgents((a) => !a)}
              title={agents ? "4 agents ON — click to disable" : "Click to enable 4-agent deliberative pipeline"}
            >
              <span className="ico"><Icon name="cpu" size={13}/></span>
              4 agents
              {agents && <span style={{ marginLeft: 4, fontSize: 10, opacity: 0.85 }}>●</span>}
            </button>

            {/* Visual grounding — always ON, shown as a static badge */}
            <span className="chip is-active" title="Visual grounding is always enabled" style={{ cursor: "default", pointerEvents: "none" }}>
              <span className="ico"><Icon name="beaker" size={13}/></span>
              Visual grounding
            </span>

            {/* PDF loaded chip */}
            {pdfLoading && (
              <span className="chip" style={{ color: "var(--accent)" }}>
                <span className="ico"><Icon name="library" size={13}/></span>
                Reading PDF…
              </span>
            )}
            {pdfName && !pdfLoading && (
              <span className="chip is-active" style={{ maxWidth: 200, overflow: "hidden" }}>
                <span className="ico"><Icon name="doc" size={13}/></span>
                <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 110 }}>{pdfName}</span>
                <button
                  onClick={clearPdf}
                  style={{ marginLeft: 2, color: "var(--surface)", opacity: 0.7, display: "flex", alignItems: "center", padding: "0 2px", borderRadius: 3 }}
                  title="Remove PDF"
                >
                  <Icon name="close" size={11}/>
                </button>
              </span>
            )}

            {/* Corpus stats */}
            <span className="chip is-static">
              <span className="ico"><Icon name="library" size={13}/></span>
              {corpusLabel}
            </span>
          </div>

          <div className="composer-right">
            {/* Mode picker */}
            <div className="mode-wrap" ref={modeRef}>
              <button
                className="model-pill"
                onClick={() => setModeOpen((open) => !open)}
                title="Switch retrieval mode"
                aria-expanded={modeOpen}
              >
                <span>{mode}</span>
                <Icon name="chevron-down" size={12}/>
              </button>
              {modeOpen && (
                <div className="mode-menu">
                  {["EVIRAG", "Vanilla"].map((item) => (
                    <button
                      key={item}
                      className={item === mode ? "is-active" : ""}
                      onClick={() => { setMode(item); setModeOpen(false); }}
                    >
                      {item}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Mic */}
            <button
              className="icon-btn"
              title={isRecording ? "Stop listening" : "Dictate (Voice typing)"}
              onClick={toggleMic}
              style={{
                color: isRecording ? "var(--contra)" : "var(--muted)",
                background: isRecording ? "var(--bg)" : "transparent",
              }}
            >
              <Icon name="mic" size={16}/>
            </button>

            {/* Send */}
            <button
              className="send-btn"
              onClick={() => submit()}
              disabled={!value.trim() && !pdfText}
              title="Submit inquiry"
            >
              <Icon name="arrow-up" size={16} stroke={2}/>
            </button>
          </div>
        </div>
      </div>

      <div className="suggest fade-up d3">
        <div
          className="suggest-head"
          style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: suggestCollapsed ? 0 : 12 }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Icon name="sparkle" size={14}/>
            <span>Try a contested question</span>
          </div>
          <button className="icon-btn" style={{ width: 24, height: 24 }} onClick={() => setSuggestCollapsed(!suggestCollapsed)}>
            <Icon name={suggestCollapsed ? "chevron-down" : "arrow-up"} size={14}/>
          </button>
        </div>
        {!suggestCollapsed && (
          <>
            <div className="suggest-tabs">
              {topicTabs.map((t) => (
                <button
                  key={t.id}
                  className={"s-tab" + (activeTab && t.id === activeTab.id ? " is-active" : "")}
                  onClick={() => setTab(t.id)}
                >
                  {t.id === "documents"   && <span style={{ display: "flex", alignItems: "center" }}><Icon name="library" size={12}/></span>}
                  {t.id === "disagreement"&& <span style={{ display: "flex", alignItems: "center" }}><Icon name="graph" size={12}/></span>}
                  {t.id === "claims"      && <span style={{ display: "flex", alignItems: "center" }}><Icon name="doc" size={12}/></span>}
                  {t.label}
                </button>
              ))}
            </div>
            <div className="suggest-list">
              {items.length ? items.map((it, i) => (
                <button key={it.claim_id || it.doc_id || i} className="s-item" onClick={() => submit(it.q)}>
                  <div>
                    <div>{it.q}</div>
                    <div className="meta" style={{ marginTop: 4 }}>{it.meta}</div>
                  </div>
                  <span className="arr"><Icon name="arrow-right" size={14}/></span>
                </button>
              )) : (
                <div className="s-empty">Waiting for corpus topics from the API.</div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

window.Home = Home;
