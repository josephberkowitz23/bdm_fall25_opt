import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf

from pyomo.environ import (
    ConcreteModel, Set, Var, NonNegativeReals, Param,
    Objective, maximize, Constraint
)
from pyomo.opt import SolverFactory, TerminationCondition

# Path to Ipopt installed by IDAES
IPOPT_PATH = "/content/bin/ipopt"

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)


# Data: monthly returns

def download_monthly_returns(tickers, start_date, end_date):
    """
    Download daily prices from Yahoo Finance and convert to monthly returns.

    Args:
        tickers (list[str]): List of ticker symbols.
        start_date (str): YYYY-MM-DD.
        end_date (str): YYYY-MM-DD.

    Returns:
        pd.DataFrame: monthly returns (rows = months, columns = tickers).
    """
    price_dict = {}

    for t in tickers:
        try:
            df = yf.download(
                t,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=False
            )
            if df.empty:
                print(f"[warn] no data for {t}, skipping")
                continue
            if "Close" not in df.columns:
                print(f"[warn] 'Close' column missing for {t}, skipping")
                continue

            close_series = df["Close"]
            # Ensure the series is not empty and has a proper DatetimeIndex
            if not close_series.empty and isinstance(close_series.index, pd.DatetimeIndex):
                price_dict[t] = close_series
            else:
                print(f"[warn] Skipping {t} due to empty or malformed 'Close' series/index for the given date range.")

        except Exception as e:
            print(f"[error] downloading {t}: {e}")

    if not price_dict:
        raise RuntimeError("No valid price data downloaded for any ticker. Check tickers/date range.")

    # combine into one DataFrame: columns = tickers, index = dates
    # Using pd.concat for more explicit alignment of Series by index
    daily_prices = pd.concat(price_dict.values(), axis=1, keys=price_dict.keys())
    daily_prices = daily_prices.dropna(how="all")

    # daily simple returns
    daily_returns = daily_prices.pct_change().dropna(how="all")

    # monthly compounded returns: (1+r1)(1+r2)... - 1
    monthly_returns = (1 + daily_returns).resample("ME").prod() - 1
    monthly_returns = monthly_returns.dropna(how="any")

    # ensure DataFrame (even if single ticker)
    if isinstance(monthly_returns, pd.Series):
        monthly_returns = monthly_returns.to_frame()

    print("Monthly returns shape:", monthly_returns.shape)
    return monthly_returns


# Optimization model

def build_markowitz_model(returns_df):
    """
    Build a Pyomo Markowitz model:
    - decision vars: x_i (weights) >= 0
    - objective: maximize expected return
    - constraint: sum x_i = 1
    - risk constraint x' Σ x <= cap will be added per-run.

    Args:
        returns_df (pd.DataFrame): monthly returns, columns = assets.

    Returns:
        (model, assets, mu, Sigma)
    """
    assets = list(returns_df.columns)
    mu = returns_df.mean()      # expected returns
    Sigma = returns_df.cov()    # covariance matrix

    m = ConcreteModel()
    m.Assets = Set(initialize=assets)

    # weights: 0 <= x_i <= 1
    m.x = Var(m.Assets, within=NonNegativeReals, bounds=(0, 1))

    # parameters
    m.mu = Param(m.Assets, initialize=mu.to_dict())
    Sigma_dict = {(i, j): float(Sigma.loc[i, j]) for i in assets for j in assets}
    m.Sigma = Param(m.Assets, m.Assets, initialize=Sigma_dict)

    # objective: maximize sum(m.mu[a] * m.x[a] for a in m.Assets)
    def total_return(m):
        return sum(m.mu[a] * m.x[a] for a in m.Assets)
    m.obj = Objective(rule=total_return, sense=maximize)

    # budget constraint: sum x_i = 1
    def budget(m):
        return sum(m.x[a] for a in m.Assets) == 1
    m.budget = Constraint(rule=budget)

    return m, assets, mu, Sigma


def portfolio_variance(weights, Sigma):
    w = np.array(weights, dtype=float)
    return float(w @ Sigma @ w)

# Efficient frontier sweep

def sweep_efficient_frontier(returns_df, ipopt_path=IPOPT_PATH, n_points=200):
    """
    For a range of variance caps, solve:
        max mu'x
        s.t. x' Σ x <= cap, sum x = 1, x >= 0

    Returns:
        frontier_df: columns ['Risk', 'Return']
        alloc_df: index = Risk, columns = assets (weights at that risk)
    """
    model, assets, mu, Sigma = build_markowitz_model(returns_df)
    Sigma_np = Sigma.values
    n = len(assets)

    # choose a range of risk caps
    eq_w = np.ones(n) / n
    min_var = portfolio_variance(eq_w, Sigma_np)          # equal-weight variance
    max_var_single = float(np.max(np.diag(Sigma_np)))     # max single-asset var

    # make sure we have a sensible range
    min_cap = max(min_var * 0.5, 1e-8)
    max_cap = max(max_var_single * 1.5, min_cap * 5)

    caps = np.linspace(min_cap, max_cap, n_points)

    solver = SolverFactory("ipopt", executable=ipopt_path)

    # store results as dict-of-lists to avoid scalar DataFrame issues
    frontier_data = {"Risk": [], "Return": []}
    alloc_data = {asset: [] for asset in assets}
    alloc_data["Risk"] = []

    print(f"Solving {len(caps)} portfolio problems from cap={min_cap:.3e} to {max_cap:.3e}...")

    for cap in caps:
        # remove old variance constraint if it exists
        if hasattr(model, "risk_constraint"):
            model.del_component(model.risk_constraint)

        # x' Σ x <= cap
        def risk_con(m):
            return sum(
                m.Sigma[i, j] * m.x[i] * m.x[j]
                for i in m.Assets for j in m.Assets
            ) <= cap

        model.risk_constraint = Constraint(rule=risk_con)

        result = solver.solve(model, tee=False)
        term = result.solver.termination_condition

        if term not in (TerminationCondition.optimal, TerminationCondition.locallyOptimal):
            # skip infeasible or failed solves
            continue

        # extract weights
        w = [model.x[a]() for a in assets]
        realized_var = portfolio_variance(w, Sigma_np)
        realized_ret = float(np.dot(mu.values, np.array(w)))

        frontier_data["Risk"].append(realized_var)
        frontier_data["Return"].append(realized_ret)

        alloc_data["Risk"].append(realized_var)
        for asset, weight in zip(assets, w):
            alloc_data[asset].append(weight)

    if len(frontier_data["Risk"]) == 0:
        raise RuntimeError("No feasible portfolios found. Try different tickers/dates.")

    frontier_df = pd.DataFrame(frontier_data).sort_values("Risk").reset_index(drop=True)

    alloc_df = pd.DataFrame(alloc_data).sort_values("Risk").set_index("Risk")

    return frontier_df, alloc_df


# Plotting helpers

def _save_and_report(fig_name):
    filepath = os.path.join(output_dir, fig_name)
    plt.savefig(filepath, bbox_inches="tight")
    print(f"Saved figure to: {filepath}")
    return filepath


def plot_frontier(frontier_df, *, ticker_label="portfolio"):
    plt.figure(figsize=(8, 5))
    plt.plot(frontier_df["Risk"], frontier_df["Return"],
             marker="o", linestyle="-", markersize=3)
    plt.title("Efficient Frontier")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Expected Monthly Return")
    plt.grid(True)
    plt.tight_layout()
    path = _save_and_report(f"efficient_frontier_{ticker_label}.png")
    plt.show()
    return path


def plot_allocations(alloc_df, *, ticker_label="portfolio"):
    plt.figure(figsize=(10, 6))
    for col in alloc_df.columns:
        plt.plot(
            alloc_df.index,
            alloc_df[col],
            marker="o",
            markersize=3,
            linewidth=0.7,
            label=str(col)
        )
    plt.title("Optimal Allocation vs Portfolio Risk")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Proportion Invested")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    path = _save_and_report(f"allocation_by_risk_{ticker_label}.png")
    plt.show()
    return path


# Convenience wrapper

def run_portfolio_example(
    tickers,
    start_date,
    end_date,
    ipopt_path=IPOPT_PATH,
    n_points=200
):
    monthly_returns = download_monthly_returns(tickers, start_date, end_date)
    print("Monthly returns head:")
    print(monthly_returns.head())

    frontier_df, alloc_df = sweep_efficient_frontier(
        monthly_returns,
        ipopt_path=ipopt_path,
        n_points=n_points
    )

    ticker_label = "_".join(tickers)
    plot_frontier(frontier_df, ticker_label=ticker_label)
    plot_allocations(alloc_df, ticker_label=ticker_label)

    return monthly_returns, frontier_df, alloc_df


def _cumulative_return(returns_df):
    compounded = (1 + returns_df).prod() - 1
    return float(compounded.mean()) if isinstance(compounded, pd.Series) else float(compounded)


def run_paper_test(tickers):
    """Run a simple walk-forward paper test using the existing pipeline."""
    train_start = "2024-01-01"
    train_end = "2025-07-31"
    test_start = "2025-08-01"
    test_end = "2025-10-31"

    train_returns = download_monthly_returns(tickers, train_start, train_end)
    test_returns = download_monthly_returns(tickers, test_start, test_end)

    frontier_df, alloc_df = sweep_efficient_frontier(train_returns)
    best_alloc = alloc_df.iloc[-1]
    strategy_rets = (test_returns * best_alloc.values).sum(axis=1)

    equal_weight = np.ones(len(tickers)) / len(tickers)
    eq_rets = (test_returns * equal_weight).sum(axis=1)

    benchmark_ticker = "SPY"
    benchmark_returns = download_monthly_returns([benchmark_ticker], test_start, test_end)
    benchmark_series = benchmark_returns.iloc[:, 0]

    strategy_total = _cumulative_return(strategy_rets)
    eq_total = _cumulative_return(eq_rets)
    bench_total = _cumulative_return(benchmark_series)

    print("Strategy vs Benchmarks (Aug–Oct 2025):")
    print(f"  Model strategy:   {strategy_total*100:.2f}% total return")
    print(f"  Equal weight:     {eq_total*100:.2f}% total return")
    print(f"  {benchmark_ticker}:      {bench_total*100:.2f}% total return")

    ticker_label = "_".join(tickers) + "_paper_test"
    plot_frontier(frontier_df, ticker_label=ticker_label)
    plot_allocations(alloc_df, ticker_label=ticker_label)


if __name__ == "__main__":
    TICKERS = ["GE", "KO", "NVDA"]
    # Example training run over a custom date range. Adjust START and END
    # to explore different periods or swap in your own tickers above.
    START = "2024-01-01"
    END = "2025-07-31"

    run_portfolio_example(
        tickers=TICKERS,
        start_date=START,
        end_date=END,
        n_points=250   # more grid points = more dots
    )

    # Run a simple paper test: train on Jan 2024–Jul 2025, test on Aug–Oct 2025
    run_paper_test(TICKERS)
