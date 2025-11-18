# Portfolio Pipeline

A lightweight pipeline for downloading equity prices, fitting a mean-variance optimization model, and visualizing the efficient frontier plus optimal allocations. The layout mirrors the structure requested by the professor so anyone can clone the repo, install dependencies, and run the CLI.

## Features
- Pulls price history with [yfinance](https://pypi.org/project/yfinance/).
- Cleans and resamples prices (daily or monthly) before computing returns.
- Solves a long-only mean-variance optimization across a grid of risk caps using CVXPY and multiple solvers (IPOPT if available, otherwise ECOS/OSQP/SCS).
- Saves efficient frontier and allocation plots to disk so they render reliably outside notebooks.

## Installation
Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The requirements include the common open-source solvers that ship with CVXPY and `cyipopt` for IPOPT support on Linux. If IPOPT wheels are unavailable for your platform, CVXPY will automatically fall back to ECOS/OSQP/SCS.

## Usage
Run the CLI from the repository root:

```bash
python main.py \
  --tickers KO GE NVDA \
  --start 2020-01-01 \
  --end 2025-11-03 \
  --interval 1d \
  --monthly
```

Key options:
- `--tickers`: One or more ticker symbols (space separated).
- `--start` / `--end`: Date range (YYYY-MM-DD). Default end date is today.
- `--interval`: Data interval accepted by yfinance (default: `1d`).
- `--monthly`: Resample to month-end prices before computing returns (cleaner signals).
- `--allow-short`: Enable shorting (disabled by default).
- `--output-dir`: Where to write plots (`artifacts/` by default).

On completion the script writes two PNG files to the output directory:
- `efficient_frontier.png`
- `allocation_by_risk.png`

You can adjust plot behavior with `--x-axis` (`cap` to match the professor's variance axis, or `vol` for achieved volatility) and `--num-caps` to control the resolution of the risk grid.

## Project Structure
```
README.md
requirements.txt
.gitignore
LICENSE
src/
  portfolio_pipeline.py
main.py
```

## License
This project is released under the MIT License. See `LICENSE` for details.
