/* global React, Icon */
// ── EVIRAG · auth.jsx ──────────────────────────────────────────────────────────
// Splash screen + Clerk-backed sign-in / sign-up with email verification.
// window._clerk must be set by app.jsx before Login is shown.
// ──────────────────────────────────────────────────────────────────────────────
const { useState, useEffect } = React;

const Wordmark = ({ size = 28 }) => (
  <span className="splash-mark" style={{ fontSize: size }}>
    evirag<span className="dot" style={{ width: size * 0.22, height: size * 0.22, transform: `translateY(-${size * 0.32}px)` }}/>
  </span>
);

// ── Splash ────────────────────────────────────────────────────────────────────
const Splash = ({ onGetStarted }) => (
  <div className="splash">
    <div className="splash-top">
      <Wordmark size={28}/>
      <div className="splash-meta">v0.5</div>
    </div>
    <div className="splash-center">
      <p className="splash-tag">
        Scientific truth lives in the disagreement,<br/>not the consensus.
      </p>
      <button className="splash-cta" onClick={onGetStarted}>
        Get started
        <span className="arr"><Icon name="arrow-right" size={14} stroke={2}/></span>
      </button>
    </div>
    <div className="splash-foot" style={{ justifyContent: "flex-end" }}>
      <span>made for researchers</span>
    </div>
  </div>
);

// ── Login (Clerk-backed) ──────────────────────────────────────────────────────
const Login = ({ onBack }) => {
  const [mode,    setMode]    = useState("signin");  // signin | signup
  const [step,    setStep]    = useState("form");    // form | verify
  const [email,   setEmail]   = useState("");
  const [pw,      setPw]      = useState("");
  const [name,    setName]    = useState("");
  const [code,    setCode]    = useState("");
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState("");

  const clerkAvailable = !!(window._clerk);

  // ── Helpers ────────────────────────────────────────────────────────────────
  const clerkErr = (err) =>
    err?.errors?.[0]?.longMessage || err?.errors?.[0]?.message || String(err);

  const demoLogin = () => {
    // No Clerk key configured — create ephemeral demo session
    const demoUser = {
      id:        "demo_" + Math.random().toString(36).slice(2, 10),
      firstName: name.split(" ")[0] || "Researcher",
      lastName:  name.split(" ").slice(1).join(" ") || "",
      get fullName() { return [this.firstName, this.lastName].filter(Boolean).join(" "); },
      primaryEmailAddress: { emailAddress: email || "demo@evirag.app" },
      imageUrl: null,
    };
    window._demoUser = demoUser;
    window.dispatchEvent(new CustomEvent("evirag:auth", { detail: { user: demoUser } }));
  };

  // ── Sign in with email ─────────────────────────────────────────────────────
  const handleSignIn = async (e) => {
    e.preventDefault();
    setError(""); setLoading(true);
    if (!clerkAvailable) { demoLogin(); return; }
    try {
      const result = await window._clerk.client.signIn.create({
        identifier: email, password: pw,
      });
      if (result.status === "complete") {
        await window._clerk.setActive({ session: result.createdSessionId });
        // Clerk listener in app.jsx picks up the new user
      } else {
        setError("Additional verification required — please check your email.");
      }
    } catch (err) { setError(clerkErr(err)); }
    finally { setLoading(false); }
  };

  // ── Sign up with email ─────────────────────────────────────────────────────
  const handleSignUp = async (e) => {
    e.preventDefault();
    setError(""); setLoading(true);
    if (!clerkAvailable) { demoLogin(); return; }
    try {
      const [firstName, ...rest] = name.trim().split(" ");
      await window._clerk.client.signUp.create({
        emailAddress: email, password: pw,
        firstName, lastName: rest.join(" ") || undefined,
      });
      await window._clerk.client.signUp.prepareEmailAddressVerification({
        strategy: "email_code",
      });
      setStep("verify");
    } catch (err) { setError(clerkErr(err)); }
    finally { setLoading(false); }
  };

  // ── Verify email code ──────────────────────────────────────────────────────
  const handleVerify = async (e) => {
    e.preventDefault();
    setError(""); setLoading(true);
    try {
      const result = await window._clerk.client.signUp.attemptEmailAddressVerification({ code });
      if (result.status === "complete") {
        await window._clerk.setActive({ session: result.createdSessionId });
      } else {
        setError("Verification incomplete — please try again.");
      }
    } catch (err) { setError(clerkErr(err)); }
    finally { setLoading(false); }
  };

  // ── Google OAuth ───────────────────────────────────────────────────────────
  const handleGoogle = async () => {
    if (!clerkAvailable) { demoLogin(); return; }
    try {
      await window._clerk.client.signIn.authenticateWithRedirect({
        strategy: "oauth_google",
        redirectUrl:         window.location.origin + "/",
        redirectUrlComplete: window.location.origin + "/",
      });
    } catch (err) {
      // Fallback: open Clerk's hosted modal
      window._clerk.openSignIn();
    }
  };

  // ── Verify step ────────────────────────────────────────────────────────────
  if (step === "verify") {
    return (
      <div className="login">
        <div className="login-top">
          <button className="login-back" onClick={() => { setStep("form"); setError(""); }}>
            <Icon name="arrow-left" size={14}/> Back
          </button>
          <Wordmark size={24}/>
        </div>
        <form className="login-card" onSubmit={handleVerify}>
          <h1>Check your email</h1>
          <div className="sub">We sent a 6-digit code to <strong>{email}</strong></div>
          {error && <div className="login-error">{error}</div>}
          <div className="login-field">
            <label>Verification code</label>
            <input
              value={code} onChange={e => setCode(e.target.value)}
              placeholder="123456" maxLength={6} autoFocus
              style={{ letterSpacing: "0.25em", fontSize: 20, textAlign: "center" }}
            />
          </div>
          <button className="login-submit" type="submit" disabled={loading}>
            {loading ? "Verifying…" : "Confirm email →"}
          </button>
          <div className="login-fine">
            Didn't receive it? Check spam or{" "}
            <button type="button" style={{ all: "unset", cursor: "pointer", color: "var(--ink)", textDecoration: "underline" }}
              onClick={async () => {
                await window._clerk?.client.signUp.prepareEmailAddressVerification({ strategy: "email_code" });
                setError("New code sent.");
              }}>
              resend
            </button>.
          </div>
        </form>
      </div>
    );
  }

  // ── Main form ──────────────────────────────────────────────────────────────
  return (
    <div className="login">
      <div className="login-top">
        <button className="login-back" onClick={onBack}>
          <Icon name="arrow-left" size={14}/> Back
        </button>
        <Wordmark size={24}/>
      </div>

      <form className="login-card" onSubmit={mode === "signin" ? handleSignIn : handleSignUp}>
        <h1>{mode === "signin" ? "Welcome back" : "Create account"}</h1>
        <div className="sub">
          {mode === "signin" ? "Sign in to your EVIRAG workspace" : "Spin up a new researcher account"}
        </div>

        <div className="login-tabs">
          <button type="button" className={mode === "signin" ? "is-active" : ""} onClick={() => { setMode("signin"); setError(""); }}>Sign in</button>
          <button type="button" className={mode === "signup" ? "is-active" : ""} onClick={() => { setMode("signup"); setError(""); }}>Sign up</button>
        </div>

        {error && <div className="login-error">{error}</div>}

        {mode === "signup" && (
          <div className="login-field">
            <label>Full name</label>
            <input value={name} onChange={e => setName(e.target.value)} placeholder="Ada Lovelace" autoFocus/>
          </div>
        )}
        <div className="login-field">
          <label>Email</label>
          <input type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="you@lab.edu"/>
        </div>
        <div className="login-field">
          <label>Password</label>
          <input type="password" value={pw} onChange={e => setPw(e.target.value)} placeholder="••••••••"/>
        </div>

        <button className="login-submit" type="submit" disabled={loading}>
          {loading ? "Please wait…" : mode === "signin" ? "Sign in →" : "Create account →"}
        </button>

        <div className="login-divider">or continue with</div>

        <button type="button" className="sso" onClick={handleGoogle} disabled={loading}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <path d="M21.6 12.2c0-.7-.1-1.4-.2-2H12v3.8h5.4c-.2 1.2-.9 2.3-2 3v2.5h3.2c1.9-1.7 3-4.3 3-7.3z"/>
            <path d="M12 22c2.7 0 5-.9 6.6-2.5l-3.2-2.5c-.9.6-2 1-3.4 1-2.6 0-4.8-1.7-5.6-4.1H3.1v2.6C4.7 19.7 8.1 22 12 22z"/>
            <path d="M6.4 13.9c-.2-.6-.3-1.3-.3-1.9s.1-1.3.3-1.9V7.5H3.1C2.4 8.9 2 10.4 2 12s.4 3.1 1.1 4.5l3.3-2.6z"/>
            <path d="M12 5.9c1.5 0 2.8.5 3.8 1.5l2.8-2.8C16.9 3 14.7 2 12 2 8.1 2 4.7 4.3 3.1 7.5l3.3 2.6c.8-2.3 3-4.2 5.6-4.2z"/>
          </svg>
          Continue with Google
        </button>

        {!clerkAvailable && (
          <div className="login-fine" style={{ color: "var(--accent)", marginTop: 8 }}>
            ⚠ Demo mode: add a Clerk publishable key in EVIRAG_CONFIG.clerkKey for real auth.
          </div>
        )}

        <div className="login-fine">
          By continuing you agree to the <a href="#">terms</a> and acknowledge that EVIRAG surfaces disagreement rather than resolving it.
        </div>
      </form>
    </div>
  );
};

window.Splash = Splash;
window.Login  = Login;
window.Wordmark = Wordmark;
