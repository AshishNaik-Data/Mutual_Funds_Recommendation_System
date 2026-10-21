import React, { useEffect, useState } from "react";
import "./App.css";
import FundCard from "./FundCard";

// Set REACT_APP_API_URL in frontend/.env (e.g. your ngrok URL) instead of editing code.
const API_URL = (process.env.REACT_APP_API_URL || "http://localhost:8000").replace(/\/+$/, "");
// ngrok's free tier shows an HTML warning page to browsers unless this header is sent.
const HEADERS = { "Content-Type": "application/json", "ngrok-skip-browser-warning": "true" };

const RISK_LEVELS = ["Low", "Medium", "High"];
const MIN_GOAL = 15;
const MAX_GOAL = 300;

// Used only if GET /categories is unreachable.
const FALLBACK_CATEGORIES = [
  "Aggressive Hybrid Mutual Funds", "Arbitrage Mutual Funds", "Banking and PSU Mutual Funds",
  "Childrens Funds", "Conservative Hybrid Mutual Funds", "Contra Funds", "Corporate Bond Mutual Funds",
  "Credit Risk Funds", "Dividend Yield Funds", "Dynamic Asset Allocation or Balanced Advantage",
  "Dynamic Bond", "ELSS Mutual Funds", "Equity Savings Mutual Funds", "Flexi Cap Funds",
  "Floater Mutual Funds", "FoFs Domestic", "Focused Funds", "Gilt Mutual Funds", "Index Funds",
  "Large & Mid Cap Funds", "Large Cap Mutual Funds", "Liquid Mutual Funds", "Low Duration Funds",
  "Medium Duration Funds", "Medium to Long Duration Funds", "Mid Cap Mutual Funds", "Money Market Funds",
  "Multi Asset Allocation Mutual Funds", "Multi Cap Funds", "Overnight Mutual Funds", "Retirement Funds",
  "Sectoral / Thematic Mutual Funds", "Short Duration Funds", "Small Cap Mutual Funds",
  "Ultra Short Duration Funds", "Value Funds",
];

// What writes the explanations, from GET /health.
function modelText(health) {
  if (!health || !health.llm_backend) return "";
  if (health.llm_backend === "none") return "Explanations: from fund data (no language model)";
  const name = String(health.base_model || health.llm_backend).split("/").pop();
  if (health.llm_backend === "hf") {
    return health.fine_tuned ? `Model: ${name} + LoRA fine-tune` : `Model: ${name} (not fine-tuned)`;
  }
  return `Model: ${name}`;
}

async function errorMessage(res) {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d) => d.msg).join("; ");
  } catch (_) {
    /* not JSON (e.g. ngrok offline page) */
  }
  return `Server error (${res.status})`;
}

function App() {
  const [goal, setGoal] = useState("");
  const [category, setCategory] = useState(""); // "" = any category
  const [risk, setRisk] = useState("");
  const [categories, setCategories] = useState(FALLBACK_CATEGORIES);
  const [results, setResults] = useState([]);
  const [notice, setNotice] = useState("");
  const [query, setQuery] = useState(null);
  const [error, setError] = useState("");
  const [fieldErrors, setFieldErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_URL}/health`, { headers: HEADERS })
      .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
      .then((data) => {
        if (!cancelled) setHealth(data);
      })
      .catch(() => {
        /* footer just omits the model */
      });
    fetch(`${API_URL}/categories`, { headers: HEADERS })
      .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
      .then((data) => {
        if (!cancelled && Array.isArray(data.categories) && data.categories.length) {
          setCategories(data.categories);
        }
      })
      .catch(() => {
        /* keep the fallback list; the submit will surface connection errors */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const validateInputs = () => {
    const errors = {};
    const trimmed = goal.trim();
    if (!trimmed) errors.goal = "Goal cannot be empty.";
    else if (trimmed.length < MIN_GOAL) errors.goal = `Goal must be at least ${MIN_GOAL} characters.`;
    if (!risk) errors.risk = "Please select a risk level.";
    setFieldErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setNotice("");
    setQuery(null);
    setResults([]);
    if (!validateInputs()) return;

    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/recommend`, {
        method: "POST",
        headers: HEADERS,
        body: JSON.stringify({ goal: goal.trim(), category: category || null, risk }),
      });
      if (!res.ok) throw new Error(await errorMessage(res));
      const data = await res.json();
      setResults(data.funds || []);
      setNotice(data.notice || "");
      setQuery(data.query || null);
      if (!data.funds || data.funds.length === 0) setError("No matching funds found. Try a broader goal.");
    } catch (err) {
      const offline = err instanceof TypeError; // fetch() network failure
      setError(offline ? `Could not reach the backend at ${API_URL}. Is it running?` : err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>📊 Mutual Fund Recommendation System</h1>

      <form onSubmit={handleSubmit} className="form" noValidate>
        <label>
          💬 Investment Goal:
          <textarea
            rows="4"
            maxLength={MAX_GOAL}
            placeholder="e.g., I want long-term growth over 10 years with low costs"
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
          />
          <div className="char-count">{goal.length} / {MAX_GOAL}</div>
          {fieldErrors.goal && <p className="error">{fieldErrors.goal}</p>}
        </label>

        <label>
          📌 Category:
          <select value={category} onChange={(e) => setCategory(e.target.value)}>
            <option value="">Any category (let my goal decide)</option>
            {categories.map((cat) => (
              <option key={cat} value={cat}>{cat}</option>
            ))}
          </select>
        </label>

        <label>
          ⚠️ Risk Level:
          <select value={risk} onChange={(e) => setRisk(e.target.value)}>
            <option value="">-- Select Risk Level --</option>
            {RISK_LEVELS.map((r) => (
              <option key={r} value={r}>{r}</option>
            ))}
          </select>
          {fieldErrors.risk && <p className="error">{fieldErrors.risk}</p>}
        </label>

        <button type="submit" disabled={loading}>
          {loading ? "Finding funds…" : "🔍 Get Recommendations"}
        </button>
      </form>

      {error && <p className="error" role="alert">{error}</p>}
      {loading && <p className="loading">Loading recommendations...</p>}

      {results.length > 0 && (
        <div className="results">
          <h2 className="results-title">🔹 Recommended Mutual Funds</h2>
          {notice && <p className="notice">{notice}</p>}
          {query && query.rewritten && (
            <p className="query-info">Also searched for: “{query.rewritten}”</p>
          )}
          {results.map((fund) => (
            <FundCard key={fund.scheme_code || fund.scheme_name} fund={fund} />
          ))}
          <p className="disclaimer">
            Based on past performance, which does not guarantee future returns. This is not investment advice.
          </p>
        </div>
      )}

      <p className="backend-url">
        Backend: {API_URL}
        {modelText(health) && <span className="model-info"> · {modelText(health)}</span>}
      </p>
    </div>
  );
}

export default App;
