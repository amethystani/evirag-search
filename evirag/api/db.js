const { webcrypto } = require("node:crypto");

const SUPABASE_URL = process.env.SUPABASE_URL || "https://xlzwfkgurrrspcdyqele.supabase.co";
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const CLERK_ISSUER = process.env.CLERK_ISSUER || "https://superb-whippet-24.clerk.accounts.dev";

let jwksCache = null;
let jwksCacheAt = 0;

function json(res, status, body) {
  res.setHeader("Cache-Control", "no-store");
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.status(status).send(JSON.stringify(body));
}

function b64urlToBuffer(value) {
  const padded = value.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - value.length % 4) % 4);
  return Buffer.from(padded, "base64");
}

function decodePart(value) {
  return JSON.parse(b64urlToBuffer(value).toString("utf8"));
}

async function getJwks() {
  if (jwksCache && Date.now() - jwksCacheAt < 10 * 60 * 1000) return jwksCache;
  const resp = await fetch(`${CLERK_ISSUER}/.well-known/jwks.json`);
  if (!resp.ok) throw new Error("Unable to load Clerk JWKS.");
  jwksCache = await resp.json();
  jwksCacheAt = Date.now();
  return jwksCache;
}

async function verifyClerkToken(token) {
  const parts = String(token || "").split(".");
  if (parts.length !== 3) throw new Error("Malformed token.");
  const [headerPart, payloadPart, signaturePart] = parts;
  const header = decodePart(headerPart);
  const payload = decodePart(payloadPart);

  const expectedIssuer = CLERK_ISSUER.replace(/\/+$/, "");
  const actualIssuer = String(payload.iss || "").replace(/\/+$/, "");
  if (actualIssuer !== expectedIssuer) throw new Error("Unexpected issuer.");
  const now = Math.floor(Date.now() / 1000);
  if (payload.exp && payload.exp < now) throw new Error("Expired token.");
  if (payload.nbf && payload.nbf > now + 30) throw new Error("Token not active.");
  if (!payload.sub) throw new Error("Missing subject.");

  const jwks = await getJwks();
  const jwk = (jwks.keys || []).find((key) => key.kid === header.kid);
  if (!jwk) throw new Error("Unknown signing key.");

  const key = await (globalThis.crypto?.subtle || webcrypto.subtle).importKey(
    "jwk",
    jwk,
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false,
    ["verify"]
  );
  const ok = await (globalThis.crypto?.subtle || webcrypto.subtle).verify(
    "RSASSA-PKCS1-v1_5",
    key,
    b64urlToBuffer(signaturePart),
    Buffer.from(`${headerPart}.${payloadPart}`)
  );
  if (!ok) throw new Error("Invalid token signature.");
  return payload;
}

function parseBody(req) {
  if (!req.body) return {};
  if (typeof req.body === "object") return req.body;
  try { return JSON.parse(req.body); } catch { return {}; }
}

function enc(value) {
  return encodeURIComponent(String(value || ""));
}

function text(value, max = 500) {
  return String(value || "").slice(0, max);
}

function normalizeSources(sources) {
  return Array.isArray(sources)
    ? sources.slice(0, 8).map((s) => ({
        title: s.title,
        year: s.year,
        doi: s.doi,
        source: s.source,
      }))
    : [];
}

async function sb(path, options = {}) {
  if (!SUPABASE_SERVICE_ROLE_KEY) throw new Error("Server database key is not configured.");
  const resp = await fetch(`${SUPABASE_URL}/rest/v1/${path}`, {
    ...options,
    headers: {
      apikey: SUPABASE_SERVICE_ROLE_KEY,
      Authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  if (!resp.ok) {
    const msg = await resp.text().catch(() => "");
    const err = new Error(`Supabase ${resp.status}: ${msg.slice(0, 240)}`);
    err.status = resp.status;
    throw err;
  }

  if (options.method === "HEAD" || resp.status === 204) return null;
  const raw = await resp.text();
  return raw ? JSON.parse(raw) : null;
}

async function countRows(table, userId) {
  const resp = await fetch(`${SUPABASE_URL}/rest/v1/${table}?user_id=eq.${enc(userId)}&select=id`, {
    method: "HEAD",
    headers: {
      apikey: SUPABASE_SERVICE_ROLE_KEY,
      Authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
      Prefer: "count=exact",
    },
  });
  if (!resp.ok) return 0;
  const range = resp.headers.get("content-range") || "";
  const total = Number(range.split("/").pop());
  return Number.isFinite(total) ? total : 0;
}

async function ensureOwnedSession(sessionId, userId) {
  const rows = await sb(`chat_sessions?id=eq.${enc(sessionId)}&user_id=eq.${enc(userId)}&select=id&limit=1`);
  return Array.isArray(rows) && rows.length > 0;
}

module.exports = async function handler(req, res) {
  if (req.method !== "POST") return json(res, 405, { error: "Method not allowed" });

  let payload;
  try {
    const token = String(req.headers.authorization || "").replace(/^Bearer\s+/i, "");
    payload = await verifyClerkToken(token);
  } catch {
    return json(res, 401, { error: "Unauthorized" });
  }

  const userId = payload.sub;
  const body = parseBody(req);
  const action = body.action;
  const data = body.data || {};

  try {
    if (action === "saveSession") {
      const sessionId = text(data.sessionId, 160);
      if (!sessionId) return json(res, 400, { error: "Missing sessionId" });
      const existing = await sb(`chat_sessions?id=eq.${enc(sessionId)}&select=id,user_id&limit=1`);
      if (existing.length && existing[0].user_id !== userId) return json(res, 403, { error: "Forbidden" });
      if (existing.length) {
        await sb(`chat_sessions?id=eq.${enc(sessionId)}&user_id=eq.${enc(userId)}`, {
          method: "PATCH",
          headers: { Prefer: "return=minimal" },
          body: JSON.stringify({ title: text(data.title || "Untitled", 120), updated_at: new Date().toISOString() }),
        });
      } else {
        await sb("chat_sessions", {
          method: "POST",
          headers: { Prefer: "return=minimal" },
          body: JSON.stringify({
            id: sessionId,
            user_id: userId,
            title: text(data.title || "Untitled", 120),
            updated_at: new Date().toISOString(),
          }),
        });
      }
      return json(res, 200, { ok: true });
    }

    if (action === "getUserSessions") {
      const rows = await sb(`chat_sessions?user_id=eq.${enc(userId)}&select=id,title,created_at,updated_at&order=updated_at.desc&limit=30`);
      return json(res, 200, { data: rows || [] });
    }

    if (action === "deleteSession") {
      await sb(`chat_sessions?id=eq.${enc(data.sessionId)}&user_id=eq.${enc(userId)}`, {
        method: "DELETE",
        headers: { Prefer: "return=minimal" },
      });
      return json(res, 200, { ok: true });
    }

    if (action === "saveTurn") {
      if (!(await ensureOwnedSession(data.sessionId, userId))) return json(res, 403, { error: "Forbidden" });
      await sb("session_turns", {
        method: "POST",
        headers: { Prefer: "return=minimal" },
        body: JSON.stringify({
          session_id: text(data.sessionId, 160),
          user_id: userId,
          turn_num: Number(data.turnNum || 0),
          user_message: text(data.userMessage, 500),
          answer: text(data.answer, 8000),
          claim: text(data.claim, 500),
          sources: normalizeSources(data.sources),
        }),
      });
      return json(res, 200, { ok: true });
    }

    if (action === "getSessionTurns") {
      if (!(await ensureOwnedSession(data.sessionId, userId))) return json(res, 403, { error: "Forbidden" });
      const rows = await sb(`session_turns?session_id=eq.${enc(data.sessionId)}&user_id=eq.${enc(userId)}&select=turn_num,user_message,answer,claim,sources,created_at&order=turn_num.asc`);
      return json(res, 200, { data: rows || [] });
    }

    if (action === "saveUserClaim") {
      if (data.sessionId && !(await ensureOwnedSession(data.sessionId, userId))) {
        return json(res, 403, { error: "Forbidden" });
      }
      await sb("user_claims", {
        method: "POST",
        headers: { Prefer: "return=minimal" },
        body: JSON.stringify({
          user_id: userId,
          session_id: text(data.sessionId, 160),
          claim_text: text(data.claimText, 500),
          confidence: Number(data.confidence || 0.8),
        }),
      });
      return json(res, 200, { ok: true });
    }

    if (action === "getUserClaims") {
      const limit = Math.min(Math.max(Number(data.limit || 50), 1), 100);
      const rows = await sb(`user_claims?user_id=eq.${enc(userId)}&select=claim_text,confidence,session_id,created_at&order=created_at.desc&limit=${limit}`);
      return json(res, 200, { data: rows || [] });
    }

    if (action === "getUserStats") {
      const [sessions, turns, claims] = await Promise.all([
        countRows("chat_sessions", userId),
        countRows("session_turns", userId),
        countRows("user_claims", userId),
      ]);
      return json(res, 200, { data: { sessions, turns, claims } });
    }

    return json(res, 400, { error: "Unknown action" });
  } catch (err) {
    const status = err.status && err.status < 500 ? err.status : 500;
    return json(res, status, { error: status === 500 ? "Database request failed" : err.message });
  }
};
