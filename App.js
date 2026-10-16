import React, { useState } from "react";
import "./App.css";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

function App() {
  const [goal, setGoal] = useState("");
  const [category, setCategory] = useState("");
  const [risk, setRisk] = useState("");
  const [results, setResults] = useState([]);
  const [error, setError] = useState("");
  const [fieldErrors, setFieldErrors] = useState({});
  const [loading, setLoading] = useState(false);

  const backendURL = "https://4a3b-34-16-105-228.ngrok-free.app"; // ✅  ngrok URL
  const riskLevels = ["Low", "Medium", "High"];

  const categories = [
    "FoFs Domestic", "Arbitrage Mutual Funds", "Dynamic Asset Allocation or Balanced Advantage",
    "Sectoral / Thematic Mutual Funds", "Banking and PSU Mutual Funds", "Corporate Bond Mutual Funds",
    "Credit Risk Funds", "Dividend Yield Funds", "Dynamic Bond", "Large & Mid Cap Funds",
    "Aggressive Hybrid Mutual Funds", "Equity Savings Mutual Funds", "Flexi Cap Funds", "Floater Mutual Funds",
    "Focused Funds", "Large Cap Mutual Funds", "Gilt Mutual Funds", "Medium to Long Duration Funds",
    "Liquid Mutual Funds", "Low Duration Funds", "Medium Duration Funds", "Mid Cap Mutual Funds",
    "Money Market Funds", "Index Funds", "Value Funds", "Conservative Hybrid Mutual Funds",
    "Ultra Short Duration Funds", "Short Duration Funds", "Small Cap Mutual Funds", "ELSS Mutual Funds",
    "Childrens Funds", "Multi Asset Allocation Mutual Funds", "Multi Cap Funds", "Overnight Mutual Funds",
    "Retirement Funds", "Contra Funds"
  ];

  const validateInputs = () => {
    const errors = {};
    if (!goal.trim()) {
      errors.goal = "Goal cannot be empty.";
    } else if (goal.trim().length < 15) {
      errors.goal = "Goal must be at least 15 characters.";
    }
    if (!category) {
      errors.category = "Please select a mutual fund category.";
    }
    if (!risk) {
      errors.risk = "Please select a risk level.";
    }

    setFieldErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResults([]);

    if (!validateInputs()) return;

    setLoading(true);
    const apiUrl = `${backendURL}/recommend`;
    console.log("🚀 Calling backend:", apiUrl);

    try {
      const res = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal, category, risk }),
      });

      if (!res.ok) throw new Error("Server error");

      const data = await res.json();
      console.log("🔎 Recommendations received:", data.funds);
      setResults(data.funds || []);
    } catch (err) {
      console.error("❌ Error fetching recommendations:", err.message);
      setError("Error fetching recommendations.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>📊 Mutual Fund Recommendation System</h1>

      <form onSubmit={handleSubmit} className="form">
        <label>
          💬 Investment Goal:
          <textarea
            rows="4"
            placeholder="e.g., I want long-term capital growth with low risk"
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
          />
          <div className="char-count">{goal.length} / 300</div>
          {fieldErrors.goal && <p className="error">{fieldErrors.goal}</p>}
        </label>

        <label>
          📌 Category:
          <select value={category} onChange={(e) => setCategory(e.target.value)}>
            <option value="">-- Select Category --</option>
            {categories.map((cat, idx) => (
              <option key={idx} value={cat}>{cat}</option>
            ))}
          </select>
          {fieldErrors.category && <p className="error">{fieldErrors.category}</p>}
        </label>

        <label>
          ⚠️ Risk Level:
          <select value={risk} onChange={(e) => setRisk(e.target.value)}>
            <option value="">-- Select Risk Level --</option>
            {riskLevels.map((r, idx) => (
              <option key={idx} value={r}>{r}</option>
            ))}
          </select>
          {fieldErrors.risk && <p className="error">{fieldErrors.risk}</p>}
        </label>

        <button type="submit" disabled={loading}>
          🔍 Get Recommendations
        </button>
      </form>

      {error && <p className="error">{error}</p>}
      {loading && <p className="loading">Loading recommendations...</p>}

      {results.length > 0 && (
        <div className="results">
          <h2 style={{ color: "#0d3b66", marginBottom: "1rem" }}>
            🔹 Recommended Mutual Funds
          </h2>
          {results.map((fund, idx) => (
            <div className="fund-card" key={idx}>
              <h3>{fund.scheme_name}</h3>
              <p><strong>NAV:</strong> ₹{fund.NAV}</p>
              <p><strong>Returns:</strong> 1Y: {fund.returns_1yr}%, 3Y: {fund.returns_3yr}%, 5Y: {fund.returns_5yr}%</p>
              <p><strong>Expense Ratio:</strong> {fund.expense_ratio}%</p>
              <p><strong>Risk:</strong> {fund.risk_level}</p>
              <p><strong>Explanation:</strong> {fund.explanation}</p>

              {/* NAV Bar Chart */}
              <Bar
                data={{
                  labels: [fund.scheme_name],
                  datasets: [
                    {
                      label: "NAV (₹)",
                      data: [fund.NAV],
                      backgroundColor: "#0d3b66",
                    },
                  ],
                }}
                options={{
                  indexAxis: "y",
                  plugins: { legend: { display: false } },
                  scales: { x: { beginAtZero: true } },
                }}
                height={50}
              />

              {/* Returns Bar Chart */}
              <Bar
                data={{
                  labels: ["1Y", "3Y", "5Y"],
                  datasets: [
                    {
                      label: "Returns (%)",
                      data: [fund.returns_1yr, fund.returns_3yr, fund.returns_5yr],
                      backgroundColor: ["#3e95cd", "#8e5ea2", "#3cba9f"],
                    },
                  ],
                }}
                options={{
                  plugins: { legend: { display: false } },
                  scales: { y: { beginAtZero: true } },
                }}
                height={150}
              />
            </div>
          ))}
        </div>
      )}

      <p className="backend-url">Backend: {backendURL}</p>
    </div>
  );
}

export default App;
