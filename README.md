# Portfolio Pipeline

A lightweight pipeline for downloading equity prices, fitting a mean-variance optimization model with Pyomo + IPOPT, and visualizing the efficient frontier plus optimal allocations. The layout mirrors the structure requested by the professor so anyone can clone the repo, install dependencies, and run the CLI.

## Features
- Pulls price history with [yfinance](https://pypi.org/project/yfinance/).
- Cleans and resamples prices before computing monthly returns.
- Solves a long-only mean-variance optimization across a grid of risk caps using Pyomo and IPOPT.
- Saves efficient frontier and allocation plots to disk (or shows them interactively with `--show`).

## Installation
Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The requirements include Pyomo plus `cyipopt` for IPOPT support on Linux. Set `--ipopt-path` if IPOPT is installed elsewhere on your system.

## Usage
Run the CLI from the repository root:

```bash
python main.py \
  --tickers KO GE NVDA \
  --start 2020-01-01 \
  --end 2024-01-01 \
  --n-points 250 \
  --output-dir /output
```

Key options:
- `--tickers`: One or more ticker symbols (space separated).
- `--start` / `--end`: Date range (YYYY-MM-DD).
- `--ipopt-path`: Path to your IPOPT executable (defaults to the IDAES install path).
- `--n-points`: Number of variance caps to sweep for the frontier.
- `--output-dir`: Where to write plots (`/output` by default).
- `--show`: Display plots interactively instead of saving.

The CLI saves two PNG files when `--show` is not provided:
- `efficient_frontier.png`
- `allocation_by_risk.png`

## Quick copy/paste for a "Python tab" (REPL or notebook)
If you just want to open a fresh Python session elsewhere (e.g., another tab/notebook) and run the core functions without wiring up the CLI, follow these steps:

1. Make sure dependencies are installed in the environment backing that tab:
   ```bash
   pip install -r /path/to/clone/requirements.txt
   ```
2. Start Python in the directory where you can import the cloned repo (or add it to `PYTHONPATH`).
3. Copy/paste the snippet below to fetch returns, solve the frontier, and write the plots:
   ```python
   from pathlib import Path

   from src.portfolio_pipeline import fetch_returns, optimize_frontier, plot_allocations, plot_frontier

   returns = fetch_returns(
       ["KO", "GE", "NVDA"],
       start="2020-01-01",
       end="2024-01-01",
       use_monthly=True,
   )

   result = optimize_frontier(returns, allow_short=False, max_weight=None, num_caps=150)

   out_dir = Path("/output")
   plot_frontier(result.frontier, out_dir / "efficient_frontier.png")
   plot_allocations(result.allocations, x_axis="vol", output_path=out_dir / "allocation_by_risk.png")
   ```

This requires no command-line arguments and produces the same PNGs as the CLI. Adjust the ticker list, dates, or `x_axis` argument to explore other scenarios.

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
