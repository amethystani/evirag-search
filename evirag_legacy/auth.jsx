/* global React, Icon */
const { useState } = React;

const Wordmark = ({ size = 28 }) => (
  <span className="splash-mark" style={{ fontSize: size }}>
    evirag<span className="dot" style={{ width: size * 0.22, height: size * 0.22, transform: `translateY(-${size * 0.32}px)` }}/>
  </span>
);

const Splash = ({ onGetStarted }) => (
  <div className="splash">
    <div className="splash-top">
      <Wordmark size={28}/>
      <div className="splash-meta">epistemic-fidelity-first · v0.4</div>
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
    <div className="splash-foot">
      <span>retrieval-augmented · disagreement-aware</span>
      <span>made for researchers</span>
    </div>
  </div>
);

const Login = ({ onLogin, onBack }) => {
  const [mode, setMode] = useState("signin");
  const [email, setEmail] = useState("");
  const [pw, setPw] = useState("");
  const [name, setName] = useState("");

  const submit = (e) => {
    if (e) e.preventDefault();
    onLogin();
  };

  return (
    <div className="login">
      <div className="login-top">
        <button className="login-back" onClick={onBack}>
          <Icon name="arrow-left" size={14}/> Back
        </button>
        <Wordmark size={24}/>
      </div>

      <form className="login-card" onSubmit={submit}>
        <h1>{mode === "signin" ? "Welcome back" : "Create account"}</h1>
        <div className="sub">{mode === "signin" ? "Sign in to your EVIRAG workspace" : "Spin up a new lab corpus"}</div>

        <div className="login-tabs">
          <button type="button" className={mode === "signin" ? "is-active" : ""} onClick={() => setMode("signin")}>Sign in</button>
          <button type="button" className={mode === "signup" ? "is-active" : ""} onClick={() => setMode("signup")}>Sign up</button>
        </div>

        {mode === "signup" && (
          <div className="login-field">
            <label>Name</label>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Ada Lovelace"/>
          </div>
        )}
        <div className="login-field">
          <label>Email</label>
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@lab.edu"/>
        </div>
        <div className="login-field">
          <label>Password</label>
          <input type="password" value={pw} onChange={(e) => setPw(e.target.value)} placeholder="••••••••"/>
        </div>

        <button className="login-submit" type="submit">
          {mode === "signin" ? "Sign in" : "Create account"} →
        </button>

        <div className="login-divider">or continue with</div>

        <button type="button" className="sso" onClick={submit}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M21.6 12.2c0-.7-.1-1.4-.2-2H12v3.8h5.4c-.2 1.2-.9 2.3-2 3v2.5h3.2c1.9-1.7 3-4.3 3-7.3z"/><path d="M12 22c2.7 0 5-.9 6.6-2.5l-3.2-2.5c-.9.6-2 1-3.4 1-2.6 0-4.8-1.7-5.6-4.1H3.1v2.6C4.7 19.7 8.1 22 12 22z"/><path d="M6.4 13.9c-.2-.6-.3-1.3-.3-1.9s.1-1.3.3-1.9V7.5H3.1C2.4 8.9 2 10.4 2 12s.4 3.1 1.1 4.5l3.3-2.6z"/><path d="M12 5.9c1.5 0 2.8.5 3.8 1.5l2.8-2.8C16.9 3 14.7 2 12 2 8.1 2 4.7 4.3 3.1 7.5l3.3 2.6c.8-2.3 3-4.2 5.6-4.2z"/></svg>
          Continue with Google
        </button>
        <button type="button" className="sso" onClick={submit}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M3 3h8v8H3zM13 3h8v8h-8zM3 13h8v8H3zM13 13h8v8h-8z"/></svg>
          Continue with SSO
        </button>

        <div className="login-fine">
          By continuing you agree to the <a href="#">terms</a> and acknowledge that EVIRAG surfaces disagreement rather than resolving it.
        </div>
      </form>
    </div>
  );
};

window.Splash = Splash;
window.Login = Login;
