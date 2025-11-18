"""Portfolio optimization helpers using Pyomo and IPOPT.

This module mirrors the professor's example structure: it downloads
monthly returns with yfinance, builds a Markowitz-style Pyomo model,
solves a sweep of risk caps with IPOPT, and plots the efficient frontier
and allocation paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

"""Portfolio optimization pipeline utilities.

This module fetches prices with yfinance, prepares returns, solves a
mean-variance optimization over a grid of variance caps, and generates
plots that can be saved by the CLI.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cvxpy as cp
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

SolverName = str


@dataclass
class OptimizationResult:
    frontier: pd.DataFrame
    allocations: pd.DataFrame
    assets: List[str]


def _pick_price_frame(raw: pd.DataFrame, prefer: Sequence[str] = ("Adj Close", "Close")) -> pd.DataFrame:
    """Return a single-field price frame from the yfinance response."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        top = set(raw.columns.get_level_values(0))
        for field in prefer:
            if field in top:
                return raw[field].copy()
        first = list(top)[0]
        return raw[first].copy()
    for field in prefer:
        if field in raw.columns:
            return raw[[field]].rename(columns={field: "Price"})
    numeric_cols = [c for c in raw.columns if np.issubdtype(raw[c].dtype, np.number)]
    if not numeric_cols:
        return pd.DataFrame()
    return raw[[numeric_cols[0]]].rename(columns={numeric_cols[0]: "Price"})


def fetch_returns(
    tickers: Iterable[str],
    *,
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = True,
    use_monthly: bool = True,
    min_data_fraction: float = 0.85,
) -> pd.DataFrame:
    """Download prices and return a cleaned return matrix.

    Args:
        tickers: Iterable of ticker symbols.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD). Defaults to today.
        interval: Sampling interval understood by yfinance.
        auto_adjust: Whether to use dividend/split adjusted prices.
        use_monthly: If True, resample to monthly prices before returns.
        min_data_fraction: Minimum fraction of non-null rows required.

    Returns:
        DataFrame of percentage returns with tickers as columns.
    """

    end_date = end or date.today().isoformat()
    tickers_use = [t.strip() for t in tickers if t.strip()]
    if len(tickers_use) < 2:
        raise ValueError("Provide at least two tickers for diversification.")

    close_px = pd.DataFrame()
    for ticker in tickers_use:
        raw = yf.download(
            ticker,
            start=start,
            end=end_date,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
            group_by="column",
        )
        px = _pick_price_frame(raw)
        if px.empty:
            continue
        series = px.iloc[:, 0].rename(ticker)
        close_px = series.to_frame() if close_px.empty else close_px.join(series, how="outer")

    if close_px.empty:
        raise RuntimeError("No usable price data returned; check tickers and dates.")

    if not isinstance(close_px.index, pd.DatetimeIndex):
        close_px.index = pd.to_datetime(close_px.index)
    close_px = close_px.sort_index().dropna(how="all")

    prices_use = close_px.resample("M").last() if use_monthly else close_px
    rets = prices_use.pct_change().dropna(how="all")

    min_rows = int(min_data_fraction * len(rets))
    keep = [c for c in rets.columns if rets[c].count() >= min_rows]
    rets = rets[keep].dropna()

    if len(keep) < 2:
        raise RuntimeError("Need at least two assets with sufficient data.")

    return rets


def _fix_psd(cov: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals[eigvals < 0] = 0.0
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _solve_with_solvers(problem: cp.Problem, solver_order: Sequence[SolverName]) -> Optional[np.ndarray]:
    for solver in solver_order:
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=getattr(cp, solver), verbose=False)
        except Exception:
            continue
        if problem.status and problem.status.startswith("optimal"):
            decision = problem.variables()[0]
            if decision.value is not None:
                return np.array(decision.value).ravel()
    return None


def optimize_frontier(
    returns: pd.DataFrame,
    *,
    allow_short: bool = False,
    max_weight: Optional[float] = None,
    num_caps: int = 400,
) -> OptimizationResult:
    """Solve the efficient frontier and allocation paths.

    Args:
        returns: DataFrame of asset returns.
        allow_short: Whether to allow negative weights.
        max_weight: Optional per-asset cap.
        num_caps: Number of variance caps to evaluate.

    Returns:
        OptimizationResult containing frontier points and allocations.
    """

    assets = list(returns.columns)
    n = len(assets)
    mu = returns.mean().values
    Sigma = returns.cov().values
    Sigma_psd = _fix_psd(Sigma)

    def pvar(weights: np.ndarray) -> float:
        return float(weights @ Sigma_psd @ weights)

    x_min = cp.Variable(n)
    cons = [cp.sum(x_min) == 1]
    if not allow_short:
        cons.append(x_min >= 0)
    if max_weight is not None:
        cons.append(x_min <= max_weight)
    cp.Problem(cp.Minimize(cp.quad_form(x_min, Sigma_psd)), cons).solve(solver=cp.SCS, verbose=False)
    w_min = np.array(x_min.value).ravel()
    min_var = pvar(w_min)

    single_asset_vars = np.diag(Sigma_psd)
    max_single_var = float(single_asset_vars.max())
    caps = np.linspace(min_var * 0.95, max_single_var * 1.30, num_caps)

    solver_order: Tuple[SolverName, ...] = ("IPOPT", "ECOS", "OSQP", "SCS")
    frontier_rows = []
    allocation_rows = []

    for cap in caps:
        x = cp.Variable(n)
        cap_cons = [cp.sum(x) == 1, cp.quad_form(x, Sigma_psd) <= cap]
        if not allow_short:
            cap_cons.append(x >= 0)
        if max_weight is not None:
            cap_cons.append(x <= max_weight)
        problem = cp.Problem(cp.Maximize(mu @ x), cap_cons)
        weights = _solve_with_solvers(problem, solver_order)
        if weights is None:
            continue
        frontier_rows.append({"cap": cap, "vol": float(np.sqrt(pvar(weights))), "ret": float(mu @ weights)})
        allocation_rows.append(weights)

    if not frontier_rows:
        raise RuntimeError("No feasible frontier points; try relaxing constraints.")

    frontier = pd.DataFrame(frontier_rows)
    frontier.sort_values("cap", inplace=True, ignore_index=True)

    allocations = pd.DataFrame(np.vstack(allocation_rows), columns=assets)
    allocations["cap"] = frontier["cap"]
    allocations["vol"] = frontier["vol"]

    return OptimizationResult(frontier=frontier, allocations=allocations, assets=assets)


def plot_frontier(frontier: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.6, 5.6))
    plt.plot(frontier["vol"], frontier["ret"], marker="o", label="Efficient frontier")
    plt.xlabel("Portfolio Volatility (σ)")
    plt.ylabel("Expected Return (period)")
    plt.title("Efficient Frontier (sanity check)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_allocations(allocations: pd.DataFrame, *, x_axis: str, output_path: Path) -> Path:
    if x_axis not in {"cap", "vol"}:
        raise ValueError("x_axis must be 'cap' or 'vol'")
    xkey = x_axis
    xlabel = "Risk Cap (variance, x'Σx)" if xkey == "cap" else "Portfolio Volatility (σ)"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.6, 5.6))
    for asset in [c for c in allocations.columns if c not in {"cap", "vol"}]:
        plt.plot(
            allocations[xkey],
            allocations[asset],
            linestyle="-",
            marker="o",
            markersize=2,
            linewidth=0.7,
            alpha=0.9,
            label=asset,
        )
        plt.scatter(allocations[xkey], allocations[asset], s=8, alpha=0.7)
    plt.ylim(0.0, 1.0)
    plt.xlabel(xlabel)
    plt.ylabel("Proportion invested")
    plt.title("Optimal Allocation vs Risk (long-only)")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
