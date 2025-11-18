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
