import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";

jest.mock("react-chartjs-2", () => ({ Bar: () => null }));

const fund = {
  scheme_code: 1,
  scheme_name: "Test Large Cap Fund",
  category: "Large Cap Mutual Funds",
  NAV: 101.5,
  nav_source: "live",
  nav_date: "2026-10-16",
  returns_1yr: 5,
  returns_3yr: 15,
  returns_5yr: 12,
  expense_ratio: 0.5,
  risk_level: "Very High",
  score: 0.7,
  score_breakdown: { relevance: 0.9, returns: 0.8, cost: 0.6, horizon_fit: 1, risk_fit: 1 },
  explanation: "A low-cost large cap fund.",
  explanation_source: "llm",
};

const health = {
  status: "ok",
  llm_backend: "hf",
  base_model: "microsoft/Phi-3-mini-4k-instruct",
  adapter: "lora-phi-3-mini-4k-instruct",
  fine_tuned: true,
};

const responses = (funds) => (url) => {
  if (url.endsWith("/categories")) {
    return { categories: ["Large Cap Mutual Funds", "Loaded From Backend"], risk_levels: ["Low", "Medium", "High"] };
  }
  if (url.endsWith("/health")) return health;
  return { funds, notice: null, query: { original: "x", rewritten: null } };
};

const mockBackend = (funds) => {
  const body = responses(funds);
  global.fetch = jest.fn((url) => Promise.resolve({ ok: true, json: () => Promise.resolve(body(url)) }));
};

beforeEach(() => mockBackend([fund]));

// Wait for the startup requests so their state updates happen inside act().
const renderApp = async () => {
  render(<App />);
  await screen.findByRole("option", { name: "Loaded From Backend" });
  await screen.findByText(/Phi-3-mini-4k-instruct \+ LoRA fine-tune/);
};

const submitGoal = async () => {
  fireEvent.change(screen.getByPlaceholderText(/long-term growth/i), {
    target: { value: "Long-term growth over ten years" },
  });
  fireEvent.change(screen.getAllByRole("combobox")[1], { target: { value: "High" } });
  fireEvent.click(screen.getByRole("button", { name: /get recommendations/i }));
};

test("renders the form and loads categories from the backend", async () => {
  await renderApp();
  expect(screen.getByText(/Mutual Fund Recommendation System/i)).toBeInTheDocument();
  expect(screen.getByText(/Any category/i)).toBeInTheDocument();
});

test("validates the goal before calling the backend", async () => {
  await renderApp();
  fireEvent.click(screen.getByRole("button", { name: /get recommendations/i }));
  expect(screen.getByText(/Goal cannot be empty/i)).toBeInTheDocument();
  // only the startup requests (categories, health), no /recommend
  expect(global.fetch.mock.calls.map(([url]) => url).some((u) => u.endsWith("/recommend"))).toBe(false);
});

test("shows recommendations from the backend", async () => {
  await renderApp();
  await submitGoal();
  await waitFor(() => expect(screen.getByText("Test Large Cap Fund")).toBeInTheDocument());
  expect(screen.getByText(/A low-cost large cap fund/)).toBeInTheDocument();
  expect(screen.getByText(/live, 2026-10-16/)).toBeInTheDocument();
  expect(screen.getByText("AI-written")).toBeInTheDocument();
});

test("labels an old NAV as last published, not live", async () => {
  mockBackend([{ ...fund, nav_source: "stale", nav_date: "2013-08-13", explanation_source: "template" }]);
  await renderApp();
  await submitGoal();
  await waitFor(() => expect(screen.getByText(/last published 2013-08-13/)).toBeInTheDocument());
  expect(screen.queryByText(/live, 2013/)).not.toBeInTheDocument();
  expect(screen.getByText("From fund data")).toBeInTheDocument();
});
