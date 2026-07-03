# Macro ETF Dashboard

Generates a single-page HTML dashboard that ranks ~120 ETFs by exponential-regression
momentum, classifies the macro regime (Bonds vs Stocks) using a 9-component Trend
Score, evaluates 17 ETF-pair Matrix signals, and (optionally) runs a Claude analysis
over the full dataset.

## Quick start

```bash
# 1. Install dependencies (pinned)
python3 -m pip install -r requirements.txt

# 2. Configure your Anthropic key (only needed for the AI section)
cp .env.example .env
# then edit .env and paste your real key

# 3. Run
python3 generate_dashboard.py
# or double-click "Run Dashboard.command" on macOS
```

Run without flags in a terminal and the script asks `¿Ejecutar análisis AI? (s/n)`.
For unattended runs (cron / launchd) use the CLI flags:

| Flag        | Effect                                                        |
|-------------|---------------------------------------------------------------|
| `--ai`      | Run the Claude analysis without prompting                     |
| `--no-ai`   | Skip the Claude analysis without prompting                    |
| `--no-push` | Skip the `index.html` copy and the git commit/push            |

With no flag and no interactive terminal (e.g. cron), the AI analysis is skipped.

Data note: if the script runs while the US market is open, today's
still-forming daily bar is dropped so metrics always use the last *completed*
session (a partial bar skews volume-based accumulation counts, ATR and 1-day
returns).

Output: `macro_etf_dashboard.html` (and a copy as `index.html` for GitHub Pages).

## Configuration

All tunables are constants at the top of `generate_dashboard.py`:

| Parameter        | Default | Meaning                                                          |
|------------------|---------|------------------------------------------------------------------|
| `ACC_THRESHOLD`  | `1.5`   | % up move that qualifies a day as accumulation                   |
| `DIST_THRESHOLD` | `-1.5`  | % down move that qualifies a day as distribution                 |
| `VOL_MULT`       | `1.10`  | Volume must exceed prior day × this to count for accum/dist      |
| `VOL_MA_LEN`     | `50`    | Volume SMA window used as the second volume filter               |
| `SLOPE_WINDOW`   | `91`    | Bars for the log-linear regression that produces the "cx" score  |
| `MATRIX_BAND`    | `0.015` | ±1.5% Neutral band around the Matrix MA100                       |
| `TREND_MA`       | `100`   | Moving-average period used in Trend Score and Matrix             |

Environment overrides:

| Variable                  | Default | Effect                                  |
|---------------------------|---------|-----------------------------------------|
| `ANTHROPIC_API_KEY`       | —       | Required for the AI analysis section    |
| `DASHBOARD_FETCH_WORKERS` | `8`     | Parallel ticker downloads               |
| `DASHBOARD_LOG_LEVEL`     | `INFO`  | `DEBUG` / `INFO` / `WARNING` / `ERROR`  |

A `.env` file in this folder is auto-loaded.

## What the dashboard shows

- **Trend Score (0-100%)** — 9 macro sub-scores. ≥70% = Stocks regime.
- **Regime (Bonds / Stocks)** — symmetric hysteresis, 20 consecutive days to flip.
  Persisted in `regime_history.json` so the regime survives across runs.
- **Matrix** — 17 ETF-pair ratios vs MA100 ±1.5% band → Bullish / Neutral / Bearish.
- **ETF table** — per-category ranking by exponential-regression momentum, with
  accumulation/distribution counts (50/20/5 day), ATR, distance from EMA20/SMA50,
  and 1d/1m/3m/annual returns.
- **AI Analysis** (optional) — Claude reads the full payload and writes a 5-section
  commentary (Regime & Trend, Top Opportunities, Risk Signals, Cross-Category
  Patterns, Conclusion). Cached by payload hash in `.ai_analysis_cache.json` so
  the same dataset never costs you twice.

## Performance

Ticker downloads run in parallel with `ThreadPoolExecutor` (default 8 workers).
On a normal connection the full run finishes in ~45-90 seconds (vs ~3 minutes
when run sequentially).

## Backtesting

```bash
python3 backtest.py            # both studies (prices cached after first run)
python3 backtest.py --part a   # regime strategy sweep only
python3 backtest.py --part b   # momentum-decile study only
python3 backtest.py --refresh  # force price re-download
```

`backtest.py` reconstructs the Trend Score and the momentum ranking over the
maximum Yahoo history (the signals are deterministic functions of prices) and
evaluates: (a) an IVV/TLT regime-switching strategy under the production
hysteresis rule plus alternative rules and parameters, with per-switch costs;
(b) forward 21/63/126-day returns by momentum-score decile across the
universe. Mind the caveats in the module docstring (survivorship of the
current universe, Yahoo adjusted data, in-sample parameter sweeps).

## Testing

```bash
python3 -m pip install pytest
python3 -m pytest tests/ -v
```

The test suite covers the deterministic core metrics (`adj_slope_cx`,
`compute_acc_dist`, `compute_net_close`, `compute_atr`, `compute_regime`).
The data-fetch and HTML-generation paths are not unit-tested — they are
exercised by running the full script.

## Files

| File                            | Purpose                                            |
|---------------------------------|----------------------------------------------------|
| `generate_dashboard.py`         | Main script (data fetch + metrics + HTML + AI)     |
| `Run Dashboard.command`         | macOS double-click launcher                        |
| `regime_history.json`           | Persisted Trend Score / regime history             |
| `snapshots/YYYY-MM-DD.json`     | Daily point-in-time dataset (trend, matrix, full ETF table) for research/backtesting. One file per market session; committed along with the daily push. |
| `.ai_analysis_cache.json`       | Local cache of past Claude responses (git-ignored) |
| `index.html` / `macro_etf_dashboard.html` | Generated output                         |
| `tests/`                        | Pytest suite for metric functions                  |

## Troubleshooting

- **"No ANTHROPIC_API_KEY found"** — create `.env` from `.env.example`, or
  `export ANTHROPIC_API_KEY=...` in your shell.
- **A ticker keeps showing "failed"** — Yahoo Finance occasionally rate-limits
  or delists symbols. Retries are built in; if a ticker is permanently dead,
  remove it from `ETF_UNIVERSE` / `EXTRA_TICKERS`.
- **Regime looks wrong after a data gap** — the regime is recomputed every run
  from `regime_history.json`. Delete that file to rebuild from scratch.
