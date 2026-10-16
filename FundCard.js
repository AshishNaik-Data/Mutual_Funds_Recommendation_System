import React from "react";
import "./FundCard.css";

function FundCard({ fund }) {
  return (
    <div className="card">
      <h2>{fund.scheme_name}</h2>
      <p><strong>NAV:</strong> ₹{fund.NAV}</p>
      <p><strong>Returns:</strong> {fund.returns_1yr}% (1Y), {fund.returns_3yr}% (3Y), {fund.returns_5yr}% (5Y)</p>
      <p><strong>Expense Ratio:</strong> {fund.expense_ratio}%</p>
      <p><strong>Risk Level:</strong> {fund.risk_level}</p>
      <div className="explanation">
        <strong>Why this fund?</strong>
        <p>{fund.explanation}</p>
      </div>
    </div>
  );
}

export default FundCard;
