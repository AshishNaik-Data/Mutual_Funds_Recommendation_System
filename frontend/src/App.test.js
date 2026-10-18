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
};

beforeEach(() => {
  global.fetch = jest.fn((url) =>
    Promise.resolve({
      ok: true,
      json: () =>
        Promise.resolve(
          url.endsWith("/categories")
            ? { categories: ["Large Cap Mutual Funds", "Loaded From Backend"], risk_levels: ["Low", "Medium", "High"] }
            : { funds: [fund], notice: null, query: { original: "x", rewritten: null } }
        ),
    })
  );
});

// Wait for the categories request so its state update happens inside act().
const renderApp = async () => {
  render(<App />);
  await screen.findByRole("option", { name: "Loaded From Backend" });
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
  expect(global.fetch).toHaveBeenCalledTimes(1); // only the categories request
});

test("shows recommendations from the backend", async () => {
  await renderApp();
  fireEvent.change(screen.getByPlaceholderText(/long-term growth/i), {
    target: { value: "Long-term growth over ten years" },
  });
  fireEvent.change(screen.getAllByRole("combobox")[1], { target: { value: "High" } });
  fireEvent.click(screen.getByRole("button", { name: /get recommendations/i }));
  await waitFor(() => expect(screen.getByText("Test Large Cap Fund")).toBeInTheDocument());
  expect(screen.getByText(/A low-cost large cap fund/)).toBeInTheDocument();
  expect(screen.getByText(/live, 2026-10-16/)).toBeInTheDocument();
});
