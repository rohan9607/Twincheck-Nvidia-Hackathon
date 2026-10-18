import React, { useEffect, useMemo, useState } from "react";

const store = {
  get: (k) => { try { return localStorage.getItem(k) || ""; } catch { return ""; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* storage blocked */ } },
};

function useApi(token) {
  return useMemo(() => async (path) => {
    const r = await fetch(path, { headers: { Authorization: `Bearer ${token}` } });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
  }, [token]);
}

const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);
const img = (path, token) => `/api/image?path=${encodeURIComponent(path)}&token=${encodeURIComponent(token)}`;

function Rate({ label, r, good }) {
  if (!r || !r.n) return <div className="rate"><span>{label}</span><b>–</b><small>no trials yet</small></div>;
  return (
    <div className="rate">
      <span>{label}</span>
      <b className={good ? "ok" : ""}>{pct(r.rate)}</b>
      <div className="ci"><i style={{ left: `${r.lo * 100}%`, width: `${(r.hi - r.lo) * 100}%` }} /><em style={{ left: `${r.rate * 100}%` }} /></div>
      <small>{r.k}/{r.n} · 95% CI {pct(r.lo)}–{pct(r.hi)}</small>
    </div>
  );
}

function Live({ token }) {
  const [events, setEvents] = useState([]);
  const [status, setStatus] = useState("connecting");
  useEffect(() => {
    const es = new EventSource(`/api/stream?token=${encodeURIComponent(token)}`);
    es.onopen = () => setStatus("live");
    es.onerror = () => setStatus("reconnecting");
    es.onmessage = (m) => setEvents((ev) => [JSON.parse(m.data), ...ev].slice(0, 60));
    return () => es.close();
  }, [token]);
  const fails = events.filter((e) => !e.passed);
  return (
    <section>
      <p className="muted">Stream: <b>{status}</b> · {events.length} checks this session · {fails.length} failed</p>
      {fails.length === 0 && <p className="empty">No failed checks yet. Every pick shows up here the moment it is checked.</p>}
      <div className="alerts">
        {fails.map((e) => (
          <article key={e.check_id} className={`alert ${e.urgent ? "urgent" : ""}`}>
            <header>
              <b>Trial {e.trial_id}</b> · {e.check_point.replace("_", "-")} · {e.object_class}
              <span className={`tag ${e.action}`}>{e.action}</span>
            </header>
            <p>{e.reason}</p>
            <div className="pair">
              <figure>{e.expected_img ? <img src={img(e.expected_img, token)} alt="expected" /> : <div className="noimg">expected: from config</div>}<figcaption>Expected</figcaption></figure>
              <figure>{e.observed_img ? <img src={img(e.observed_img, token)} alt="observed" /> : <div className="noimg">no image</div>}<figcaption>Observed</figcaption></figure>
            </div>
          </article>
        ))}
      </div>
      <h3>All recent checks</h3>
      <table><thead><tr><th>trial</th><th>check</th><th>result</th><th>error</th><th>reason</th></tr></thead>
        <tbody>{events.map((e) => (
          <tr key={e.check_id}><td>{e.trial_id}</td><td>{e.check_point}</td>
            <td className={e.passed ? "ok" : "bad"}>{e.passed ? "pass" : "FAIL"}</td>
            <td>{e.position_err_mm != null ? `${Number(e.position_err_mm).toFixed(1)} mm` : ""}</td><td>{e.reason}</td></tr>))}
        </tbody></table>
    </section>
  );
}

function Results({ api }) {
  const [env, setEnv] = useState("");
  const [data, setData] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => { api(`/api/summary${env ? `?env=${env}` : ""}`).then(setData).catch((e) => setErr(String(e))); }, [api, env]);
  return (
    <section>
      <div className="seg">{["", "twin", "real"].map((e) => <button key={e} className={env === e ? "on" : ""} onClick={() => setEnv(e)}>{e || "all"}</button>)}</div>
      {err && <p className="bad">{err}</p>}
      <div className="cards">
        {(data || []).map((g) => (
          <div className="card" key={`${g.env}-${g.mode}`}>
            <h3>{g.env} · {g.mode}</h3><p className="muted">{g.trials} trials</p>
            <Rate label="Pick success" r={g.pick_success} good />
            {g.mode !== "baseline" && <Rate label="Failures caught" r={g.failures_caught} good />}
            {g.mode !== "baseline" && <Rate label="False alarms" r={g.false_alarms} />}
          </div>
        ))}
      </div>
    </section>
  );
}

function Trials({ api }) {
  const [t, setT] = useState([]);
  useEffect(() => { api("/api/trials?limit=200").then(setT).catch(() => {}); }, [api]);
  return (
    <table><thead><tr><th>#</th><th>env</th><th>mode</th><th>rung</th><th>object</th><th>caused</th><th>worked?</th><th>flagged?</th><th>resync</th></tr></thead>
      <tbody>{t.map((r) => (
        <tr key={r.trial_id}><td>{r.trial_id}</td><td>{r.env}</td><td>{r.mode}</td><td>{r.ladder_rung}</td><td>{r.object_class}</td>
          <td>{r.induced_failure || ""}</td><td>{r.pick_success == null ? "?" : r.pick_success ? "yes" : "no"}</td>
          <td className={r.flagged ? "bad" : ""}>{r.flagged ? "yes" : "no"}</td>
          <td>{r.resync_max_shift_mm != null ? `${r.resync_max_shift_mm.toFixed(1)} mm` : ""}</td></tr>))}
      </tbody></table>
  );
}

export default function App() {
  const [token, setToken] = useState(store.get("twincheck_token"));
  const [draft, setDraft] = useState("");
  const [tab, setTab] = useState("live");
  const api = useApi(token);
  if (!token) {
    return (
      <main className="login">
        <h1>TwinCheck</h1>
        <p className="muted">Enter the dashboard token (TWINCHECK_DASH_TOKEN on the Spark).</p>
        <form onSubmit={(e) => { e.preventDefault(); store.set("twincheck_token", draft); setToken(draft); }}>
          <input type="password" value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="token" />
          <button>Open</button>
        </form>
      </main>
    );
  }
  return (
    <main>
      <nav>
        <h1>TwinCheck</h1>
        {["live", "results", "trials"].map((t) => <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{t}</button>)}
        <button className="ghost" onClick={() => { store.set("twincheck_token", ""); setToken(""); }}>sign out</button>
      </nav>
      {tab === "live" && <Live token={token} />}
      {tab === "results" && <Results api={api} />}
      {tab === "trials" && <Trials api={api} />}
    </main>
  );
}
