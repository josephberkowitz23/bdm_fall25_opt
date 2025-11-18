"""Portfolio optimization helpers using Pyomo and IPOPT.

This module mirrors the professor's example structure: it downloads
monthly returns with yfinance, builds a Markowitz-style Pyomo model,
solves a sweep of risk caps with IPOPT, and plots the efficient frontier
and allocation paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from pyomo.environ import (  # type: ignore
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Param,
    Set,
    Var,
    maximize,
)
from pyomo.opt import SolverFactory, TerminationCondition  # type: ignore

# Path to Ipopt installed by IDAES (can be overridden by callers)
IPOPT_PATH = "/content/bin/ipopt"


def download_monthly_returns(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download daily prices from Yahoo Finance and convert to monthly returns.

    Args:
        tickers: List of ticker symbols.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).

    Returns:
        DataFrame of monthly returns (rows = months, columns = tickers).
    """

    price_dict = {}

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=False,
            )
            if df.empty:
                print(f"[warn] no data for {ticker}, skipping")
                continue
            if "Close" not in df.columns:
                print(f"[warn] 'Close' column missing for {ticker}, skipping")
                continue

            close_series = df["Close"]
            if not close_series.empty and isinstance(close_series.index, pd.DatetimeIndex):
                price_dict[ticker] = close_series
            else:
                print(
                    f"[warn] Skipping {ticker} due to empty or malformed 'Close' series/index",
                )

        except Exception as exc:  # pragma: no cover - external I/O
            print(f"[error] downloading {ticker}: {exc}")

    if not price_dict:
        raise RuntimeError("No valid price data downloaded for any ticker. Check tickers/date range.")

    daily_prices = pd.concat(price_dict.values(), axis=1, keys=price_dict.keys())
    daily_prices = daily_prices.dropna(how="all")

    daily_returns = daily_prices.pct_change().dropna(how="all")

    # monthly compounded returns: (1+r1)(1+r2)... - 1
    monthly_returns = (1 + daily_returns).resample("ME").prod() - 1
    monthly_returns = monthly_returns.dropna(how="any")

    if isinstance(monthly_returns, pd.Series):
        monthly_returns = monthly_returns.to_frame()

    print("Monthly returns shape:", monthly_returns.shape)
    return monthly_returns


def build_markowitz_model(returns_df: pd.DataFrame) -> Tuple[ConcreteModel, list, pd.Series, pd.DataFrame]:
    """Build a Pyomo Markowitz model with long-only weights and a unit budget."""

    assets = list(returns_df.columns)
    mu = returns_df.mean()  # expected returns
    sigma = returns_df.cov()  # covariance matrix

    model = ConcreteModel()
    model.Assets = Set(initialize=assets)

    model.x = Var(model.Assets, within=NonNegativeReals, bounds=(0, 1))

    model.mu = Param(model.Assets, initialize=mu.to_dict())
    sigma_dict = {(i, j): float(sigma.loc[i, j]) for i in assets for j in assets}
    model.Sigma = Param(model.Assets, model.Assets, initialize=sigma_dict)

    def total_return(mod):
        return sum(mod.mu[a] * mod.x[a] for a in mod.Assets)

    model.obj = Objective(rule=total_return, sense=maximize)

    def budget(mod):
        return sum(mod.x[a] for a in mod.Assets) == 1

    model.budget = Constraint(rule=budget)

    return model, assets, mu, sigma


def portfolio_variance(weights: Iterable[float], sigma: np.ndarray) -> float:
    """Compute portfolio variance given weights and covariance."""

    w = np.array(weights, dtype=float)
    return float(w @ sigma @ w)


def sweep_efficient_frontier(
    returns_df: pd.DataFrame,
    *,
    ipopt_path: str = IPOPT_PATH,
    n_points: int = 200,
):
    """Solve a grid of risk caps and return the efficient frontier and allocations."""

    model, assets, mu, sigma = build_markowitz_model(returns_df)
    sigma_np = sigma.values
    n_assets = len(assets)

    equal_weights = np.ones(n_assets) / n_assets
    min_var = portfolio_variance(equal_weights, sigma_np)
    max_var_single = float(np.max(np.diag(sigma_np)))

    min_cap = max(min_var * 0.5, 1e-8)
    max_cap = max(max_var_single * 1.5, min_cap * 5)

    caps = np.linspace(min_cap, max_cap, n_points)

    solver = SolverFactory("ipopt", executable=ipopt_path)

    frontier_data = {"Risk": [], "Return": []}
    alloc_data = {asset: [] for asset in assets}
    alloc_data["Risk"] = []

    print(f"Solving {len(caps)} portfolio problems from cap={min_cap:.3e} to {max_cap:.3e}...")

    for cap in caps:
        if hasattr(model, "risk_constraint"):
            model.del_component(model.risk_constraint)

        def risk_con(mod):
            return sum(mod.Sigma[i, j] * mod.x[i] * mod.x[j] for i in mod.Assets for j in mod.Assets) <= cap

        model.risk_constraint = Constraint(rule=risk_con)

        result = solver.solve(model, tee=False)
        term = result.solver.termination_condition

        if term not in (TerminationCondition.optimal, TerminationCondition.locallyOptimal):
            continue

        weights = [model.x[a]() for a in assets]
        realized_var = portfolio_variance(weights, sigma_np)
        realized_ret = float(np.dot(mu.values, np.array(weights)))

        frontier_data["Risk"].append(realized_var)
        frontier_data["Return"].append(realized_ret)

        alloc_data["Risk"].append(realized_var)
        for asset, weight in zip(assets, weights):
            alloc_data[asset].append(weight)

    if not frontier_data["Risk"]:
        raise RuntimeError("No feasible portfolios found. Try different tickers/dates.")

    frontier_df = pd.DataFrame(frontier_data).sort_values("Risk").reset_index(drop=True)
    alloc_df = pd.DataFrame(alloc_data).sort_values("Risk").set_index("Risk")

    return frontier_df, alloc_df


def plot_frontier(frontier_df: pd.DataFrame, output_path: Path | None = None) -> Path | None:
    """Plot the efficient frontier; optionally save to disk."""

    plt.figure(figsize=(8, 5))
    plt.plot(frontier_df["Risk"], frontier_df["Return"], marker="o", linestyle="-", markersize=3)
    plt.title("Efficient Frontier")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Expected Monthly Return")
    plt.grid(True)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
        plt.close()
        return output_path

    plt.show()
    return None


def plot_allocations(alloc_df: pd.DataFrame, output_path: Path | None = None) -> Path | None:
    """Plot allocation paths vs portfolio risk; optionally save to disk."""

    plt.figure(figsize=(10, 6))
    for col in alloc_df.columns:
        plt.plot(alloc_df.index, alloc_df[col], marker="o", markersize=3, linewidth=0.7, label=str(col))
    plt.title("Optimal Allocation vs Portfolio Risk")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Proportion Invested")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
        plt.close()
        return output_path

    plt.show()
    return None


def run_portfolio_example(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    *,
    ipopt_path: str = IPOPT_PATH,
    n_points: int = 200,
    output_dir: Path | None = None,
):
    """Convenience wrapper to run the full pipeline and plot results."""

    monthly_returns = download_monthly_returns(tickers, start_date, end_date)
    print("Monthly returns head:")
    print(monthly_returns.head())

    frontier_df, alloc_df = sweep_efficient_frontier(
        monthly_returns,
        ipopt_path=ipopt_path,
        n_points=n_points,
    )

    frontier_path = None
    alloc_path = None
    if output_dir:
        output_dir = Path(output_dir)
        frontier_path = output_dir / "efficient_frontier.png"
        alloc_path = output_dir / "allocation_by_risk.png"

    plot_frontier(frontier_df, output_path=frontier_path)
    plot_allocations(alloc_df, output_path=alloc_path)

    return monthly_returns, frontier_df, alloc_df


if __name__ == "__main__":
    TICKERS = ["GE", "KO", "NVDA"]
    START = "2020-01-01"
    END = "2024-01-01"

    run_portfolio_example(
        tickers=TICKERS,
        start_date=START,
        end_date=END,
        n_points=250,
    )
