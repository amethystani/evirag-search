/* global window */
// ── EVIRAG · db.jsx ────────────────────────────────────────────────────────────
// Supabase data layer: sessions, turns, analytics, user claims.
// Exposed as window.EVIRAG_DB — called from app.jsx, sidebar.jsx, pages.jsx.
// ──────────────────────────────────────────────────────────────────────────────

(function () {
  const SUPABASE_URL  = "https://xlzwfkgurrrspcdyqele.supabase.co";
  const SUPABASE_ANON = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhsendma2d1cnJyc3BjZHlxZWxlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMzNjEwMDcsImV4cCI6MjA3ODkzNzAwN30.6uq0o5Vidu21UBEtjdOJGZEeH8-2unsyPpvnA49CxnM";

  let _sb = null;
  function getSB() {
    if (!_sb && window.supabase) {
      _sb = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON);
    }
    return _sb;
  }

  // ── Sessions ─────────────────────────────────────────────────────────────────
  async function saveSession(sessionId, userId, title) {
    const sb = getSB(); if (!sb || !userId) return;
    const { error } = await sb.from("chat_sessions").upsert({
      id: sessionId, user_id: userId,
      title: (title || "Untitled").slice(0, 120),
      updated_at: new Date().toISOString(),
    }, { onConflict: "id" });
    if (error) console.warn("[db] saveSession:", error.message);
  }

  async function getUserSessions(userId) {
    const sb = getSB(); if (!sb || !userId) return [];
    const { data, error } = await sb
      .from("chat_sessions")
      .select("id, title, created_at, updated_at")
      .eq("user_id", userId)
      .order("updated_at", { ascending: false })
      .limit(30);
    if (error) console.warn("[db] getUserSessions:", error.message);
    return data || [];
  }

  async function deleteSession(sessionId) {
    const sb = getSB(); if (!sb) return;
    const { error } = await sb.from("chat_sessions").delete().eq("id", sessionId);
    if (error) console.warn("[db] deleteSession:", error.message);
  }

  // ── Turns ─────────────────────────────────────────────────────────────────────
  async function saveTurn({ sessionId, userId, turnNum, userMessage, answer, claim, sources }) {
    const sb = getSB(); if (!sb || !userId) return;
    const { error } = await sb.from("session_turns").insert({
      session_id: sessionId, user_id: userId, turn_num: turnNum,
      user_message: (userMessage || "").slice(0, 500),
      answer: (answer || "").slice(0, 8000),
      claim: (claim || "").slice(0, 500),
      sources: (sources || []).slice(0, 8).map(s => ({
        title: s.title, year: s.year, doi: s.doi, source: s.source
      })),
    });
    if (error) console.warn("[db] saveTurn:", error.message);
  }

  async function getSessionTurns(sessionId) {
    const sb = getSB(); if (!sb) return [];
    const { data, error } = await sb
      .from("session_turns")
      .select("turn_num, user_message, answer, claim, sources, created_at")
      .eq("session_id", sessionId)
      .order("turn_num", { ascending: true });
    if (error) console.warn("[db] getSessionTurns:", error.message);
    return data || [];
  }

  // ── Query analytics (Discover) ────────────────────────────────────────────────
  const CATEGORIES = {
    biology:   ["gene", "cell", "protein", "crispr", "dna", "rna", "evolution",
                "species", "bacteria", "virus", "immune", "cancer", "tumor", "organism"],
    neuroscience: ["brain", "neural", "neuron", "synapse", "cognition", "memory",
                   "alzheimer", "dopamine", "cortex", "consciousness", "sleep"],
    physics:   ["quantum", "relativity", "particle", "dark matter", "black hole",
                "gravity", "thermodynamics", "photon", "wave", "string theory"],
    chemistry: ["molecule", "reaction", "catalyst", "polymer", "bond", "acid",
                "carbon", "compound", "synthesis", "element", "periodic"],
    climate:   ["climate", "temperature", "co2", "greenhouse", "warming", "emission",
                "atmosphere", "ocean", "arctic", "carbon", "fossil fuel"],
    ai:        ["neural network", "machine learning", "deep learning", "transformer",
                "llm", "gpt", "artificial intelligence", "reinforcement", "diffusion"],
    medicine:  ["drug", "vaccine", "treatment", "disease", "clinical", "patient",
                "therapy", "diagnosis", "covid", "antibiotic", "surgery", "trial"],
  };

  function categorize(query) {
    const q = query.toLowerCase();
    for (const [cat, keywords] of Object.entries(CATEGORIES)) {
      if (keywords.some(k => q.includes(k))) return cat;
    }
    return "general";
  }

  async function trackQuery(query) {
    const sb = getSB();
    if (!sb || !query || query.length < 5) return;
    const category = categorize(query);
    const { error } = await sb.rpc("increment_query_count", {
      p_query: query.slice(0, 200),
      p_category: category,
    });
    if (error) console.warn("[db] trackQuery:", error.message);
  }

  async function getPopularQueries(limit = 30) {
    const sb = getSB(); if (!sb) return [];
    const { data, error } = await sb
      .from("query_analytics")
      .select("query, category, count, last_seen")
      .order("count", { ascending: false })
      .limit(limit);
    if (error) console.warn("[db] getPopularQueries:", error.message);
    return data || [];
  }

  // ── User claims (cross-session graph) ────────────────────────────────────────
  async function saveUserClaim({ userId, sessionId, claimText, confidence }) {
    const sb = getSB(); if (!sb || !userId || !claimText) return;
    const { error } = await sb.from("user_claims").insert({
      user_id: userId, session_id: sessionId,
      claim_text: claimText.slice(0, 500),
      confidence: confidence || 0.8,
    });
    if (error) console.warn("[db] saveUserClaim:", error.message);
  }

  async function getUserClaims(userId, limit = 50) {
    const sb = getSB(); if (!sb || !userId) return [];
    const { data, error } = await sb
      .from("user_claims")
      .select("claim_text, confidence, session_id, created_at")
      .eq("user_id", userId)
      .order("created_at", { ascending: false })
      .limit(limit);
    if (error) console.warn("[db] getUserClaims:", error.message);
    return data || [];
  }

  // ── User stats (for profile page) ─────────────────────────────────────────────
  async function getUserStats(userId) {
    const sb = getSB(); if (!sb || !userId) return { sessions: 0, turns: 0, claims: 0 };
    const [sessRes, turnsRes, claimsRes] = await Promise.all([
      sb.from("chat_sessions").select("id", { count: "exact", head: true }).eq("user_id", userId),
      sb.from("session_turns").select("id", { count: "exact", head: true }).eq("user_id", userId),
      sb.from("user_claims").select("id",   { count: "exact", head: true }).eq("user_id", userId),
    ]);
    return {
      sessions: sessRes.count  || 0,
      turns:    turnsRes.count || 0,
      claims:   claimsRes.count || 0,
    };
  }

  // ── Expose ────────────────────────────────────────────────────────────────────
  window.EVIRAG_DB = {
    saveSession, getUserSessions, deleteSession,
    saveTurn, getSessionTurns,
    trackQuery, getPopularQueries,
    saveUserClaim, getUserClaims,
    getUserStats,
    categorize,
  };
})();
