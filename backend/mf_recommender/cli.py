"""Command-line recommender (replaces the notebook's input() loop).

    python -m mf_recommender.cli "Grow my savings over 7 years" --risk High
    python -m mf_recommender.cli "Park cash for 3 months" --category "Liquid Mutual Funds" --llm none
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging

from .config import Settings
from .data import RISK_BANDS
from .service import RecommenderService


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("goal", help="Investment goal in plain English")
    parser.add_argument("--category", default=None, help="Sub-category (default: search all)")
    parser.add_argument("--risk", choices=list(RISK_BANDS), default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-explain", action="store_true", help="Skip explanations")
    parser.add_argument("--llm", choices=["hf", "chat", "none"], help="Override MF_LLM_BACKEND")
    parser.add_argument("--offline", action="store_true", help="No live NAV lookups")
    parser.add_argument("--json", action="store_true", help="Print raw JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    settings = Settings.from_env()
    if args.llm:
        settings = dataclasses.replace(settings, llm_backend=args.llm)
    if args.offline:
        settings = dataclasses.replace(settings, live_nav=False)

    service = RecommenderService.from_settings(settings)
    result = service.recommend(args.goal, args.category, args.risk, top_k=args.top_k, explain=not args.no_explain)

    if args.json:
        print(json.dumps(dataclasses.asdict(result), indent=2, ensure_ascii=False))
        return
    if result.query.get("rewritten"):
        print(f"Search query (LLM rewrite): {result.query['rewritten']}")
    if result.notice:
        print(f"Note: {result.notice}")
    for i, fund in enumerate(result.funds, 1):
        nav = f"₹{fund['NAV']:.2f} ({fund['nav_source']})" if fund["NAV"] else "n/a"
        print(f"\n{i}. {fund['scheme_name']}  [{fund['category']}, {fund['risk_level']} risk]")
        print(
            f"   NAV {nav} | Returns 1Y {fund['returns_1yr']}%  3Y {fund['returns_3yr']}%  "
            f"5Y {fund['returns_5yr']}% | Expense {fund['expense_ratio']}%"
        )
        b = fund["score_breakdown"]
        print(
            f"   Score {fund['score']:.3f} (relevance {b['relevance']}, returns {b['returns']}, "
            f"cost {b['cost']}, horizon fit {b['horizon_fit']}, risk fit {b['risk_fit']})"
        )
        if fund["explanation"]:
            print(f"   Why: {fund['explanation']}  [{fund['explanation_source']}]")


if __name__ == "__main__":
    main()
