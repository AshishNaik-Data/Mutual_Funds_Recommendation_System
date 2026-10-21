import React from "react";
import { Bar } from "react-chartjs-2";
import { BarElement, CategoryScale, Chart as ChartJS, Legend, LinearScale, Tooltip } from "chart.js";
import "./FundCard.css";

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

const pct = (v) => (v === null || v === undefined ? "n/a" : `${v}%`);

function navText(fund) {
  if (fund.NAV === null || fund.NAV === undefined) return "n/a";
  const value = `₹${Number(fund.NAV).toFixed(2)}`;
  if (fund.nav_source === "live" && fund.nav_date) return `${value} (live, ${fund.nav_date})`;
  if (fund.nav_source === "stale" && fund.nav_date) {
    // No NAV published for weeks: the dataset's scheme code is probably a closed or merged plan.
    return `${value} (last published ${fund.nav_date}; this plan may be closed or merged)`;
  }
  return fund.nav_source === "dataset" ? `${value} (from dataset)` : value;
}

const SOURCE_TAGS = {
  llm: ["AI-written", "Written by the language model and checked against the fund's figures"],
  template: ["From fund data", "Built directly from the fund's figures"],
};

const SCORE_LABELS = [
  ["relevance", "Goal match"],
  ["returns", "Returns vs peers"],
  ["cost", "Cost vs peers"],
];

function FundCard({ fund }) {
  const returns = [fund.returns_1yr, fund.returns_3yr, fund.returns_5yr];
  const breakdown = fund.score_breakdown || {};

  return (
    <div className="fund-card">
      <h3>{fund.scheme_name}</h3>
      {fund.category && <p className="fund-meta">{fund.category}</p>}
      <p>
        <strong>NAV:</strong>{" "}
        <span className={fund.nav_source === "stale" ? "nav-stale" : undefined}>{navText(fund)}</span>
      </p>
      <p>
        <strong>Returns:</strong> 1Y: {pct(fund.returns_1yr)}, 3Y: {pct(fund.returns_3yr)}, 5Y: {pct(fund.returns_5yr)}
      </p>
      <p><strong>Expense Ratio:</strong> {pct(fund.expense_ratio)}</p>
      <p><strong>Risk:</strong> {fund.risk_level}</p>

      {fund.explanation && (
        <div className="explanation">
          <strong>Why this fund?</strong>
          {SOURCE_TAGS[fund.explanation_source] && (
            <span
              className={`source-tag ${fund.explanation_source}`}
              title={SOURCE_TAGS[fund.explanation_source][1]}
            >
              {SOURCE_TAGS[fund.explanation_source][0]}
            </span>
          )}
          <p>{fund.explanation}</p>
        </div>
      )}

      {fund.score_breakdown && (
        <div className="chips" aria-label="Score breakdown">
          {SCORE_LABELS.map(([key, label]) => (
            <span className="chip" key={key}>
              {label}: {Math.round((breakdown[key] || 0) * 100)}%
            </span>
          ))}
        </div>
      )}

      <div className="chart">
        <Bar
          data={{
            labels: ["1Y", "3Y", "5Y"],
            datasets: [
              {
                label: "Returns (%)",
                data: returns,
                backgroundColor: ["#3e95cd", "#8e5ea2", "#3cba9f"],
              },
            ],
          }}
          options={{
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true, title: { display: true, text: "Return (%)" } } },
          }}
        />
      </div>
    </div>
  );
}

export default FundCard;
