"""Command-line interface for the portfolio pipeline."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.portfolio_pipeline import (
    fetch_returns,
    optimize_frontier,
    plot_allocations,
    plot_frontier,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Efficient frontier + allocation plots")
    parser.add_argument("--tickers", nargs="+", default=["KO", "GE", "NVDA"], help="Ticker symbols")
    parser.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--interval", default="1d", help="Price interval for yfinance")
    parser.add_argument("--monthly", action="store_true", help="Resample to month-end prices")
    parser.add_argument("--allow-short", action="store_true", help="Allow short positions")
    parser.add_argument("--max-weight", type=float, default=None, help="Per-asset weight cap (e.g., 0.4)")
    parser.add_argument("--num-caps", type=int, default=400, help="Number of variance caps to evaluate")
    parser.add_argument("--x-axis", choices=["cap", "vol"], default="cap", help="Plot x-axis variable")
    parser.add_argument("--output-dir", default="/output", help="Directory to write plots")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    end_date = args.end or date.today().isoformat()
    returns = fetch_returns(
        args.tickers,
        start=args.start,
        end=end_date,
        interval=args.interval,
        auto_adjust=True,
        use_monthly=args.monthly,
    )

    result = optimize_frontier(
        returns,
        allow_short=args.allow_short,
        max_weight=args.max_weight,
        num_caps=args.num_caps,
    )

    output_dir = Path(args.output_dir)
    frontier_path = output_dir / "efficient_frontier.png"
    allocations_path = output_dir / "allocation_by_risk.png"

    plot_frontier(result.frontier, frontier_path)
    plot_allocations(result.allocations, x_axis=args.x_axis, output_path=allocations_path)

    print(f"Saved frontier plot to {frontier_path.resolve()}")
    print(f"Saved allocation plot to {allocations_path.resolve()}")


if __name__ == "__main__":
    main()
