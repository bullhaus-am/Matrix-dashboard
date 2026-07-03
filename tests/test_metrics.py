"""Unit tests for the deterministic metric functions in generate_dashboard.py.

These avoid any network access — only synthetic OHLCV inputs.
"""
from pathlib import Path
import sys
import json
import math

import numpy as np
import pandas as pd
import pytest

# Make the parent directory importable. The module name has no spaces, even
# though the folder does.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import generate_dashboard as gd  # noqa: E402


def _make_df(closes, highs=None, lows=None, volumes=None):
    n = len(closes)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    closes = np.asarray(closes, dtype=float)
    if highs is None:
        highs = closes * 1.005
    if lows is None:
        lows = closes * 0.995
    if volumes is None:
        volumes = np.full(n, 1_000_000, dtype=float)
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


# ── adj_slope_cx ──────────────────────────────────────────────────────────────

def test_adj_slope_cx_positive_for_uptrend():
    # Smooth exponential uptrend → positive slope
    closes = 100 * np.exp(np.linspace(0, 0.5, 400))
    s = pd.Series(closes)
    cx = gd.adj_slope_cx(s, offset=0)
    assert not math.isnan(cx)
    assert cx > 0


def test_adj_slope_cx_negative_for_downtrend():
    closes = 100 * np.exp(np.linspace(0, -0.5, 400))
    s = pd.Series(closes)
    cx = gd.adj_slope_cx(s, offset=0)
    assert cx < 0


def test_adj_slope_cx_nan_when_too_short():
    s = pd.Series(np.linspace(100, 110, 10))
    assert math.isnan(gd.adj_slope_cx(s, offset=0))


# ── compute_acc_dist ──────────────────────────────────────────────────────────

def test_compute_acc_dist_counts_only_high_volume_big_moves():
    # acc_50 looks at .iloc[-1] of a rolling-50 sum, i.e. the last 50 bars only.
    # Place events near the end of the series so they fall in the window.
    n = 120
    closes = np.full(n, 100.0)
    volumes = np.full(n, 1_000_000.0)

    # Day 100: +2% on 2x volume → accumulation (within last 50 bars)
    closes[100] = closes[99] * 1.02
    volumes[100] = volumes[99] * 2.0
    # Day 110: -2% on 2x volume → distribution (within last 50 bars)
    closes[110] = closes[109] * 0.98
    volumes[110] = volumes[109] * 2.0

    df = _make_df(closes, volumes=volumes)
    out = gd.compute_acc_dist(df)
    assert out["acc_50"] >= 1
    assert out["dist_50"] >= 1
    assert out["net_50"] == out["acc_50"] - out["dist_50"]


def test_compute_acc_dist_ignores_low_volume_moves():
    n = 120
    closes = np.full(n, 100.0)
    closes[60] = closes[59] * 1.02  # +2% but flat volume
    df = _make_df(closes)
    out = gd.compute_acc_dist(df)
    assert out["acc_50"] == 0


# ── compute_net_close ─────────────────────────────────────────────────────────

def test_compute_net_close_positive_when_closing_at_highs():
    n = 30
    df = _make_df(
        closes=np.full(n, 100.0),
        highs=np.full(n, 100.0),  # close == high → close_pos = 1
        lows=np.full(n, 99.0),
    )
    assert gd.compute_net_close(df, lookback=5) == 5


def test_compute_net_close_negative_when_closing_at_lows():
    n = 30
    df = _make_df(
        closes=np.full(n, 99.0),
        highs=np.full(n, 100.0),
        lows=np.full(n, 99.0),  # close == low → close_pos = 0
    )
    assert gd.compute_net_close(df, lookback=5) == -5


# ── compute_atr ───────────────────────────────────────────────────────────────

def test_compute_atr_returns_positive_percentage():
    n = 60
    closes = np.linspace(100, 110, n)
    df = _make_df(closes, highs=closes + 1.0, lows=closes - 1.0)
    atr = gd.compute_atr(df, period=5)
    assert atr > 0
    assert atr < 100  # sanity: as % of price


# ── save_snapshot ─────────────────────────────────────────────────────────────

def test_save_snapshot_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ranked = [{"ticker": "IVV", "rank_today": np.int64(1), "cx": np.float64(0.51)}]
    matrix = [{"n": "Stocks vs Gold", "sig": "Bullish", "f": 1, "pct": 80}]
    p = gd.save_snapshot("Jun 10, 2026", 0.6, [2, 1, 0], "Bonds", 76, matrix, ranked)
    snap = json.loads(p.read_text())
    assert p.name == "2026-06-10.json"  # keyed by session date, not run date
    assert snap["session"] == "2026-06-10"
    assert snap["trend_score"] == 0.6
    assert snap["matrix_score"] == 1
    assert snap["etfs"][0]["rank_today"] == 1  # numpy scalars serialized as JSON numbers


def test_save_snapshot_overwrites_same_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = [{"n": "x", "sig": "Neutral", "f": 0, "pct": 50}]
    gd.save_snapshot("Jun 10, 2026", 0.7, [1], "Bonds", 75, m, [])
    p = gd.save_snapshot("Jun 10, 2026", 0.6, [1], "Bonds", 76, m, [])
    assert json.loads(p.read_text())["trend_score"] == 0.6
    assert len(list(p.parent.glob("*.json"))) == 1


# ── _drop_partial_bar ─────────────────────────────────────────────────────────

def _df_ending_today_et(periods=10):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    idx = pd.bdate_range(end=today_et, periods=periods)
    return _make_df(np.full(len(idx), 100.0)).set_index(idx)


def test_drop_partial_bar_drops_today_while_session_open(monkeypatch):
    monkeypatch.setattr(gd, "_us_session_in_progress", lambda: True)
    df = _df_ending_today_et()
    out = gd._drop_partial_bar(df)
    assert len(out) == len(df) - 1
    assert out.index[-1] == df.index[-2]


def test_drop_partial_bar_keeps_today_after_close(monkeypatch):
    monkeypatch.setattr(gd, "_us_session_in_progress", lambda: False)
    df = _df_ending_today_et()
    assert len(gd._drop_partial_bar(df)) == len(df)


def test_drop_partial_bar_keeps_stale_last_bar(monkeypatch):
    monkeypatch.setattr(gd, "_us_session_in_progress", lambda: True)
    df = _df_ending_today_et(periods=11).iloc[:-1]  # last bar = previous bday
    assert len(gd._drop_partial_bar(df)) == len(df)


# ── compute_regime ────────────────────────────────────────────────────────────

def test_compute_regime_starts_in_bonds():
    # All scores below threshold → still Bonds
    regime, days = gd.compute_regime([0.3] * 50)
    assert regime == "Bonds"
    assert days == 50


def test_compute_regime_flips_to_stocks_after_confirm_days():
    series = [0.3] * 30 + [0.8] * 25  # 25 consecutive ≥0.7 (>20)
    regime, days = gd.compute_regime(series, threshold=0.7, confirm_days=20)
    assert regime == "Stocks"
    assert days == 25


def test_compute_regime_does_not_flip_too_early():
    series = [0.3] * 30 + [0.8] * 10  # only 10 days above — not enough
    regime, _ = gd.compute_regime(series, threshold=0.7, confirm_days=20)
    assert regime == "Bonds"


def test_compute_regime_flips_back_to_bonds():
    series = [0.8] * 30 + [0.3] * 25  # enters Stocks, then 25 days below
    regime, days = gd.compute_regime(series, threshold=0.7, confirm_days=20)
    assert regime == "Bonds"
    assert days == 25
