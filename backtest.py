"""backtest.py — historical validation of the dashboard's signals.

The Trend Score and the momentum ranking are deterministic functions of
prices, so they can be reconstructed far beyond the 2-year window the daily
dashboard uses. Two studies:

  Part A — regime strategy. Rebuild the daily Trend Score over the longest
  common history of its 12 component tickers, derive the Bonds/Stocks regime
  under several hysteresis rules (the production one and alternatives), and
  evaluate an IVV-when-Stocks / TLT-when-Bonds switching strategy against
  buy-and-hold baselines.

  Part B — momentum deciles. Every 21 trading days, rank the full ETF
  universe by the exponential-regression momentum score (same `adj_slope_cx`
  the dashboard uses) and measure forward 21/63/126-day returns by decile.
  A sane momentum signal shows monotonically increasing forward returns from
  decile 1 (weakest) to 10 (strongest).

Caveats — read before trusting the numbers:
  * Prices are Yahoo back-adjusted closes (total-return-ish, survivorship of
    the *current* universe: ETFs that died are not in the list).
  * Part A's window is capped by the youngest Trend-Score component.
  * Costs: COST_BPS per regime switch; momentum deciles are frictionless.

Usage:
    python3 backtest.py              # both parts, cached prices if available
    python3 backtest.py --refresh    # force re-download
    python3 backtest.py --part a     # regime study only
    python3 backtest.py --part b     # momentum-decile study only
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import generate_dashboard as gd

CACHE_FILE = ".backtest_prices.pkl"
COST_BPS   = 10          # round-trip cost charged on each regime switch
TREND_MA   = gd.TREND_MA # 100

# The 12 series the Trend Score needs, plus TLT for the Bonds leg.
TREND_TICKERS = ['IVV', 'HYG', 'IEI', 'ISCB', 'ILCB', 'IMCG', 'IMCV',
                 'RSPD', 'RSPS', 'IDU', 'GLD', 'GSG']
RATIOS = [('HYG', 'IEI'), ('ISCB', 'ILCB'), ('IMCG', 'IMCV'),
          ('RSPD', 'RSPS'), ('RSPD', 'IDU'), ('IVV', 'GLD'), ('IVV', 'GSG')]


# ── Data ───────────────────────────────────────────────────────────────────────
def load_prices(refresh: bool = False) -> dict:
    """{ticker: ohlcv df} with max available history, cached on disk."""
    p = Path(CACHE_FILE)
    if p.exists() and not refresh:
        with p.open("rb") as f:
            data = pickle.load(f)
        print(f"Loaded {len(data)} tickers from {CACHE_FILE} "
              f"(use --refresh to re-download).")
        return data

    tickers = sorted(set([t for t, _, _ in gd.ETFS] + TREND_TICKERS + ['TLT']))
    print(f"Downloading max history for {len(tickers)} tickers...")
    raw = yf.download(tickers=tickers, period="max", auto_adjust=True,
                      progress=False, threads=True, group_by="column",
                      repair=True)
    data = gd._split_batch(raw)
    print(f"Got {len(data)}/{len(tickers)} tickers.")
    with p.open("wb") as f:
        pickle.dump(data, f)
    return data


# ── Part A: regime strategy ────────────────────────────────────────────────────
def trend_score_series(close_wide: pd.DataFrame) -> pd.Series:
    """Daily Trend Score, vectorized — identical math to compute_trend_score().

    `close_wide` must contain the 12 TREND_TICKERS aligned on common dates
    (same intersection rule the production script uses)."""
    ivv = close_wide['IVV']
    ma  = ivv.rolling(TREND_MA).mean()
    pts = (ivv > ma).astype(int) * 2
    pts += ((ma - ma.shift(10)) > 0).astype(int)
    for num, den in RATIOS:
        r = close_wide[num] / close_wide[den] * 100
        pts += (r > r.rolling(TREND_MA).mean()).astype(int)
    return (pts / 10).round(1)


def regime_consecutive(scores: np.ndarray, threshold=0.7, confirm=20) -> np.ndarray:
    """Production rule: N *consecutive* days across the threshold to flip."""
    regime, above, below = 'Bonds', 0, 0
    out = np.empty(len(scores), dtype=object)
    for i, s in enumerate(scores):
        if s >= threshold:
            above += 1; below = 0
            if regime == 'Bonds' and above >= confirm:
                regime = 'Stocks'
        else:
            below += 1; above = 0
            if regime == 'Stocks' and below >= confirm:
                regime = 'Bonds'
        out[i] = regime
    return out


def regime_dual(scores: np.ndarray, enter=0.7, exit_=0.6) -> np.ndarray:
    """Dual-threshold hysteresis: flip immediately, no streak counter."""
    regime = 'Bonds'
    out = np.empty(len(scores), dtype=object)
    for i, s in enumerate(scores):
        if regime == 'Bonds' and s >= enter:
            regime = 'Stocks'
        elif regime == 'Stocks' and s < exit_:
            regime = 'Bonds'
        out[i] = regime
    return out


def regime_x_of_n(scores: np.ndarray, threshold=0.7, x=15, n=20) -> np.ndarray:
    """Flip when >= x of the last n days are across the threshold
    (tolerates noisy single days, unlike the consecutive-streak rule)."""
    above = (scores >= threshold).astype(int)
    roll = pd.Series(above).rolling(n, min_periods=1).sum().values
    regime = 'Bonds'
    out = np.empty(len(scores), dtype=object)
    for i in range(len(scores)):
        window = min(i + 1, n)
        if regime == 'Bonds' and roll[i] >= x:
            regime = 'Stocks'
        elif regime == 'Stocks' and (window - roll[i]) >= x:
            regime = 'Bonds'
        out[i] = regime
    return out


def evaluate(regimes: np.ndarray, rets: pd.DataFrame) -> dict:
    """Daily IVV-or-TLT switching P&L. Signal from close t → position held
    from close t to close t+1 (one-day execution lag, no lookahead)."""
    pos = pd.Series(regimes == 'Stocks', index=rets.index).shift(1, fill_value=False)
    r = np.where(pos, rets['IVV'], rets['TLT'])
    switches = pos != pos.shift(1, fill_value=False)
    switches.iloc[0] = False
    r = r - switches.values * (COST_BPS / 1e4)
    eq = np.cumprod(1 + r)
    yrs = len(r) / 252
    cagr = eq[-1] ** (1 / yrs) - 1
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    sharpe = float(np.mean(r) / np.std(r) * np.sqrt(252)) if np.std(r) > 0 else 0.0
    return {"cagr": cagr, "sharpe": sharpe, "maxdd": dd,
            "switches": int(switches.sum()), "pct_in_stocks": float(pos.mean())}


def run_part_a(data: dict):
    print("\n" + "=" * 78)
    print("PART A — Regime strategy (IVV when Stocks / TLT when Bonds)")
    print("=" * 78)

    missing = [t for t in TREND_TICKERS + ['TLT'] if t not in data]
    if missing:
        print(f"Missing tickers, aborting Part A: {missing}")
        return
    wide = pd.DataFrame({t: data[t]['close'] for t in TREND_TICKERS + ['TLT']}).dropna()
    scores = trend_score_series(wide[TREND_TICKERS])

    # Drop the MA100 burn-in, keep the evaluable window
    start = wide.index[TREND_MA + 10]
    scores = scores.loc[start:]
    rets = wide[['IVV', 'TLT']].pct_change().loc[start:]
    print(f"Usable window: {scores.index[0].date()} → {scores.index[-1].date()} "
          f"({len(scores)} sessions, {len(scores)/252:.1f} years)"
          f" — capped by the youngest Trend-Score component.")
    print(f"Cost charged per switch: {COST_BPS} bps. Sharpe uses rf=0.\n")

    rows = []
    sc = scores.values
    # Production rule + sweep
    for thr in (0.5, 0.6, 0.7, 0.8):
        for cf in (5, 10, 15, 20):
            tag = " ← production" if (thr == 0.7 and cf == 20) else ""
            rows.append((f"consecutive  thr={thr:.1f} confirm={cf}{tag}",
                         evaluate(regime_consecutive(sc, thr, cf), rets)))
    # Dual threshold
    for enter, ex in ((0.7, 0.6), (0.7, 0.5), (0.8, 0.6), (0.6, 0.5)):
        rows.append((f"dual         enter={enter:.1f} exit={ex:.1f}",
                     evaluate(regime_dual(sc, enter, ex), rets)))
    # X of N
    for thr, x, n in ((0.7, 15, 20), (0.7, 10, 15), (0.6, 15, 20)):
        rows.append((f"x-of-n       thr={thr:.1f} {x}/{n}",
                     evaluate(regime_x_of_n(sc, thr, x, n), rets)))
    # Baselines
    for name, col in (("Buy&Hold IVV", 'IVV'), ("Buy&Hold TLT", 'TLT')):
        r = rets[col].values
        eq = np.cumprod(1 + r)
        yrs = len(r) / 252
        rows.append((name, {
            "cagr": eq[-1] ** (1 / yrs) - 1,
            "sharpe": float(np.mean(r) / np.std(r) * np.sqrt(252)),
            "maxdd": float((eq / np.maximum.accumulate(eq) - 1).min()),
            "switches": 0, "pct_in_stocks": 1.0 if col == 'IVV' else 0.0}))

    rows.sort(key=lambda kv: kv[1]["sharpe"], reverse=True)
    hdr = f"{'rule':<42}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'flips':>7}{'%stk':>7}"
    print(hdr); print("-" * len(hdr))
    for name, m in rows:
        print(f"{name:<42}{m['cagr']:>7.1%}{m['sharpe']:>8.2f}"
              f"{m['maxdd']:>8.1%}{m['switches']:>7d}{m['pct_in_stocks']:>7.0%}")


# ── Part B: momentum deciles ───────────────────────────────────────────────────
def run_part_b(data: dict, horizons=(21, 63, 126), step=21, min_history=300):
    print("\n" + "=" * 78)
    print("PART B — Forward returns by momentum-score decile (full ETF universe)")
    print("=" * 78)

    universe = [t for t, _, _ in gd.ETFS if t in data]
    calendar = data['IVV']['close'].index
    max_h = max(horizons)
    reb_dates = calendar[min_history:len(calendar) - max_h:step]
    print(f"{len(universe)} tickers · rebalance every {step} sessions · "
          f"{len(reb_dates)} dates from {reb_dates[0].date()} to {reb_dates[-1].date()}\n")

    obs = {h: {} for h in horizons}   # horizon -> decile -> [fwd returns]
    for dt in reb_dates:
        cx, fwd = {}, {}
        for t in universe:
            c = data[t]['close']
            pos = c.index.searchsorted(dt, side='right') - 1
            if pos < min_history or pos + max_h >= len(c):
                continue
            v = gd.adj_slope_cx(c.iloc[:pos + 1], 0)
            if np.isnan(v):
                continue
            cx[t] = v
            fwd[t] = {h: c.iloc[pos + h] / c.iloc[pos] - 1 for h in horizons}
        if len(cx) < 30:
            continue
        deciles = pd.qcut(pd.Series(cx), 10, labels=False, duplicates='drop')
        for t, d in deciles.items():
            for h in horizons:
                obs[h].setdefault(int(d) + 1, []).append(fwd[t][h])

    hdr = f"{'decile':<10}" + "".join(f"{f'fwd {h}d':>10}" for h in horizons) + f"{'n':>8}"
    print(hdr); print("-" * len(hdr))
    for d in range(1, 11):
        cells = "".join(f"{np.mean(obs[h].get(d, [np.nan])):>10.2%}" for h in horizons)
        n = len(obs[horizons[0]].get(d, []))
        label = {1: "1 (worst)", 10: "10 (best)"}.get(d, str(d))
        print(f"{label:<10}{cells}{n:>8d}")
    spread = {h: np.mean(obs[h].get(10, [np.nan])) - np.mean(obs[h].get(1, [np.nan]))
              for h in horizons}
    print("-" * len(hdr))
    print(f"{'10 − 1':<10}" + "".join(f"{spread[h]:>10.2%}" for h in horizons))


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Backtest the dashboard's signals.")
    ap.add_argument("--refresh", action="store_true", help="re-download prices")
    ap.add_argument("--part", choices=["a", "b", "all"], default="all")
    args = ap.parse_args()

    data = load_prices(refresh=args.refresh)
    if args.part in ("a", "all"):
        run_part_a(data)
    if args.part in ("b", "all"):
        run_part_b(data)
