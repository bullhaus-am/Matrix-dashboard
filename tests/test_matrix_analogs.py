"""Tests de matrix_analogs.py (análogos históricos) — todo sintético, sin pickle ni red."""
from pathlib import Path
import json
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matrix_analogs as ma  # noqa: E402
import macro_insights as mi  # noqa: E402
import generate_dashboard as gd  # noqa: E402

N = len(ma.MATRIX_PAIRS)
SUB = [2, 1, 1, 0, 0, 0, 0, 1, 0]


# ── helpers ────────────────────────────────────────────────────────────────────
def _feat(flags=None, sub=None, streak=0.0):
    return {"flags": list(flags) if flags is not None else [0] * N, "sub": list(sub or SUB), "streak": streak}


def _synthetic_prices(n=260, seed=0, short=None, drop=()):
    """Random walks para todos los tickers de la Matrix; `short` = {ticker: barras}."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    tickers = sorted({t for _, a, b, _ in gd.MATRIX_DEFS for t in (a, b) if t} | set(ma.ANALOG_FWD_TICKERS))
    prices = {}
    for t in tickers:
        if t in drop:
            continue
        s = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))), index=idx)
        if short and t in short:
            s = s.iloc[-short[t]:]
        prices[t] = s
    return prices


def _snap(session, flags, sub=SUB, trend=0.5, regime="Bonds", regime_days=5, closes=None):
    sig = {1: "Bullish", 0: "Neutral", -1: "Bearish"}
    matrix = [{"n": n, "sig": sig[f], "f": f, "pct": 50} for n, f in zip(ma.MATRIX_PAIRS, flags)]
    etfs = [{"ticker": t, "close": c} for t, c in (closes or {"IVV": 100, "TLT": 100, "GLD": 100}).items()]
    return {"session": session, "trend_score": trend, "trend_sub_scores": list(sub), "regime": regime,
            "regime_days": regime_days, "matrix_score": sum(flags), "matrix": matrix, "etfs": etfs}


# ── distancia ──────────────────────────────────────────────────────────────────
def test_distance_zero_for_identical_state():
    assert ma.analog_distance(_feat(), _feat()) == 0


def test_distance_weights_flags_subscores_and_streak():
    one_flag = _feat(flags=[1] + [0] * (N - 1))
    assert ma.analog_distance(_feat(), one_flag) == pytest.approx(ma.ANALOG_W_FLAGS * 1)
    # Bullish vs Bearish es un solo mismatch (Hamming), no dos
    assert ma.analog_distance(_feat(flags=[-1] + [0] * (N - 1)), one_flag) == pytest.approx(ma.ANALOG_W_FLAGS * 1)
    # componente de 2 puntos normalizado a [0, 1]: 2 → 0 vale lo mismo que 1 → 0 en uno de 1 punto
    assert ma.analog_distance(_feat(), _feat(sub=[0, 1, 1, 0, 0, 0, 0, 1, 0])) == pytest.approx(ma.ANALOG_W_TREND * 1)
    assert ma.analog_distance(_feat(), _feat(sub=[2, 0, 1, 0, 0, 0, 0, 1, 0])) == pytest.approx(ma.ANALOG_W_TREND * 1)
    # racha: +20 sesiones vs −20 sesiones = 2 en la escala [−1, 1]
    assert ma.analog_distance(_feat(streak=1.0), _feat(streak=-1.0)) == pytest.approx(ma.ANALOG_W_TREND * 2)


def test_distance_rescales_hamming_when_a_pair_is_not_computable():
    b = _feat(flags=[1] + [0] * (N - 2) + [None])
    p = ma.distance_parts(_feat(), b)
    assert p["pairs_compared"] == N - 1 and p["matrix_mismatches"] == 1
    assert p["distance"] == pytest.approx(N / (N - 1), abs=1e-3)


def test_distance_none_below_min_pairs():
    b = _feat(flags=[0] * (ma.ANALOG_MIN_PAIRS - 1) + [None] * (N - ma.ANALOG_MIN_PAIRS + 1))
    assert ma.analog_distance(_feat(), b) is None
    assert ma.distance_parts(_feat(), b) is None


# ── selección de análogos ──────────────────────────────────────────────────────
def test_select_excludes_current_session_and_last_20():
    d = [5.0] * 60
    d[59] = 0.0                                   # hoy
    d[55] = 0.0                                   # dentro de las últimas 20 sesiones
    d[39] = 0.0                                   # justo en el borde (excluida: 59-1-20 = 38 es la última elegible)
    d[10] = 1.0
    chosen = ma.select_analogs(d, k=5)
    assert [c["pos"] for c in chosen] == [10]
    d[38] = 0.5
    assert ma.select_analogs(d, k=1)[0]["pos"] == 38


def test_select_groups_consecutive_days_into_one_episode_keeping_first():
    d = [5.0] * 80
    d[20:25] = [1.5, 1.2, 0.4, 0.9, 1.3]          # un episodio: mejor día 22, racha 21-24 (20 queda fuera: 1.5 > 0.4 + 1.0)
    d[50] = 2.0                                   # otro episodio
    chosen = ma.select_analogs(d, k=3)
    assert [c["pos"] for c in chosen] == [21, 50]
    assert chosen[0]["best_pos"] == 22 and chosen[0]["best_distance"] == 0.4
    assert chosen[0]["last_pos"] == 24 and chosen[0]["episode_sessions"] == 4
    assert chosen[1]["episode_sessions"] == 1


def test_select_treats_candidates_within_gap_as_one_episode_and_skips_none():
    d = [None] * 10 + [5.0] * 70
    d[30], d[36] = 1.0, 1.1                       # 6 sesiones de distancia → mismo episodio
    d[60] = 1.05
    out = [c["pos"] for c in ma.select_analogs(d, k=5, gap=ma.ANALOG_EPISODE_GAP)]
    assert out == [30, 60] or out == [30]         # 60 queda si no cae en las últimas 20 (80-1-20 = 59 → excluido)
    assert out == [30]


# ── régimen y racha ────────────────────────────────────────────────────────────
def test_regime_path_matches_production_compute_regime():
    rng = np.random.default_rng(1)
    scores = list(np.round(rng.choice([0.5, 0.6, 0.7, 0.8, 0.9], size=400, p=[0.2, 0.2, 0.2, 0.2, 0.2]), 1))
    path = ma.regime_path(scores)
    for i in range(1, 400, 7):
        assert path[i][0] == gd.compute_regime(scores[:i + 1])[0]


def test_regime_path_day_counts_sessions_since_flip():
    scores = [0.8] * 25 + [0.5] * 22
    path = ma.regime_path(scores)
    assert path[18] == ("Bonds", 19)
    assert path[19] == ("Stocks", 1)              # 20ª sesión ≥ 70% → flip, día 1
    assert path[24] == ("Stocks", 6)
    assert path[44] == ("Bonds", 1)               # 20ª sesión < 70%
    assert path[46] == ("Bonds", 3)


def test_streak_features_sign_and_cap():
    f = ma.streak_features([0.8] * 25 + [0.5] * 3)
    assert f[0] == pytest.approx(1 / 20)
    assert f[24] == 1.0                           # tope en 20 sesiones
    assert f[25] == pytest.approx(-1 / 20)
    assert f[-1] == pytest.approx(-3 / 20)


# ── forward returns ────────────────────────────────────────────────────────────
def _rows(n, ivv_step=1.01, gld_step=0.99):
    return [{"date": f"d{i:03d}", "flags": [0] * N, "sub": SUB, "trend": 0.8 if i < 15 else 0.5,
             "matrix_score": i % 3, "close": {"IVV": 100 * ivv_step ** i, "TLT": 100.0, "GLD": 100 * gld_step ** i}}
            for i in range(n)]


def test_forward_stats_on_synthetic_series():
    rows = _rows(30)
    regimes = [("Stocks", i + 1) for i in range(15)] + [("Bonds", i + 1) for i in range(15)]
    f = ma.forward_stats(rows, regimes, 0)
    assert f["fwd_10"]["ivv_pct"] == pytest.approx((1.01 ** 10 - 1) * 100, abs=0.01)
    assert f["fwd_10"]["gld_pct"] == pytest.approx((0.99 ** 10 - 1) * 100, abs=0.01)
    assert f["fwd_10"]["tlt_pct"] == 0
    assert f["fwd_10"]["regime_flipped"] is False and f["fwd_10"]["sessions_to_flip"] is None
    assert f["fwd_20"]["regime_flipped"] is True and f["fwd_20"]["sessions_to_flip"] == 15
    assert f["fwd_20"]["regime_after"] == "Bonds"
    assert f["fwd_20"]["trend_change_pts"] == -30
    assert f["fwd_20"]["matrix_score_change"] == rows[20]["matrix_score"] - rows[0]["matrix_score"]
    assert f["ivv_worst_drawdown_20_pct"] == 0.0        # IVV sube todos los días


def test_forward_stats_drawdown_and_missing_horizon():
    rows = _rows(30, ivv_step=0.99)
    regimes = [("Stocks", i + 1) for i in range(30)]
    f = ma.forward_stats(rows, regimes, 0)
    assert f["ivv_worst_drawdown_20_pct"] == pytest.approx((0.99 ** 20 - 1) * 100, abs=0.01)
    f = ma.forward_stats(rows, regimes, 25)
    assert f["fwd_10"] is None and f["fwd_20"] is None and f["ivv_worst_drawdown_20_pct"] is None


def test_summarize_analogs_base_rates():
    a = {"regime": "Stocks", "fwd_10": {"ivv_pct": 1, "tlt_pct": 0, "gld_pct": 0}, "ivv_worst_drawdown_20_pct": -3.0,
         "fwd_20": {"ivv_pct": 2, "tlt_pct": -1, "gld_pct": 1, "regime_flipped": True, "sessions_to_flip": 8,
                    "trend_change_pts": -10, "matrix_score_change": -2}}
    b = json.loads(json.dumps(a)); b["regime"] = "Bonds"; b["fwd_20"].update(ivv_pct=-4, regime_flipped=False, sessions_to_flip=None)
    b["ivv_worst_drawdown_20_pct"] = -9.5
    s = ma.summarize_analogs([a, b], today_regime="Stocks")
    assert s["pct_with_regime_flip_within_20"] == 50 and s["median_sessions_to_flip_when_flipped"] == 8
    assert s["median_ivv_20_pct"] == -1 and s["worst_ivv_drawdown_20_pct"] == -9.5
    assert s["pct_ivv_positive_20"] == 50 and s["analogs_in_same_regime_as_today"] == 1


# ── reconstrucción con las funciones de producción ─────────────────────────────
def test_reconstruct_states_matches_production_functions():
    prices = _synthetic_prices(short={"QQQJ": 150}, drop=("ARGT",))
    rec = ma.reconstruct_states(prices)
    states = rec["states"]
    idx = prices["IVV"].index
    assert rec["missing_tickers"] == ["ARGT"]
    assert states[0]["date"] == idx[gd.TREND_MA + 10 - 1].strftime("%Y-%m-%d")
    assert len(states) == len(idx) - (gd.TREND_MA + 10 - 1)
    qqqj_from = prices["QQQJ"].index[gd.TREND_MA - 1]
    assert rec["pair_available_from"]["QQQJ vs QQQ"] == qqqj_from.strftime("%Y-%m-%d")
    assert rec["pair_available_from"]["Argentina vs S&P"] is None
    i_arg, i_qqqj = ma.MATRIX_PAIRS.index("Argentina vs S&P"), ma.MATRIX_PAIRS.index("QQQJ vs QQQ")
    for st in (states[0], states[len(states) // 2], states[-1]):
        cutoff = pd.Timestamp(st["date"])
        full = {t: s.loc[:cutoff] for t, s in prices.items()}
        full["ARGT"] = pd.Series(float("nan"), index=full["IVV"].index)   # producción explota sin la clave
        score, sub = gd.compute_trend_score(full)
        assert st["trend"] == score and st["sub"] == sub
        prod = {m["n"]: m["f"] for m in gd.compute_matrix(full)}
        assert st["flags"][i_arg] is None
        if cutoff < qqqj_from:
            assert st["flags"][i_qqqj] is None
        for n, f in zip(ma.MATRIX_PAIRS, st["flags"]):
            if f is not None:
                assert f == prod[n], (st["date"], n)
        assert st["matrix_score"] == sum(f for f in st["flags"] if f is not None)
        assert st["close"]["IVV"] == pytest.approx(float(prices["IVV"].loc[:cutoff].iloc[-1]))


def test_load_reconstruction_cache_keyed_by_last_price_day(tmp_path):
    prices = _synthetic_prices(n=200)
    pkl, cache = tmp_path / "prices.pkl", tmp_path / "cache.json"
    with pkl.open("wb") as f:
        pickle.dump({t: pd.DataFrame({"close": s}) for t, s in prices.items()}, f)
    rec = ma.load_reconstruction(pkl, cache)
    assert cache.exists() and rec["prices_last_day"] == prices["IVV"].index[-1].strftime("%Y-%m-%d")
    n_states = len(rec["states"])
    # cache vigente → se reutiliza tal cual (sin reconstruir)
    cached = json.loads(cache.read_text()); cached["states"] = cached["states"][:3]
    cache.write_text(json.dumps(cached))
    assert len(ma.load_reconstruction(pkl, cache)["states"]) == 3
    # un día más de precios → se reconstruye
    longer = _synthetic_prices(n=201)
    with pkl.open("wb") as f:
        pickle.dump({t: pd.DataFrame({"close": s}) for t, s in longer.items()}, f)
    assert len(ma.load_reconstruction(pkl, cache)["states"]) == n_states + 1
    # sin pickle pero con cache (runner cloud) → se usa el cache tal cual
    assert len(ma.load_reconstruction(tmp_path / "nope.pkl", cache)["states"]) == n_states + 1
    # sin pickle y sin cache → None (solo snapshots)
    assert ma.load_reconstruction(tmp_path / "nope.pkl", tmp_path / "nope.json") is None


# ── unión de fuentes ───────────────────────────────────────────────────────────
def test_merge_states_splices_snapshot_closes_and_ffills():
    rec = [{"date": "2026-06-09", "flags": [0] * N, "sub": SUB, "trend": 0.5, "matrix_score": 0,
            "close": {"IVV": 100.0, "TLT": 50.0, "GLD": 10.0}},
           {"date": "2026-06-10", "flags": [0] * N, "sub": SUB, "trend": 0.5, "matrix_score": 0,
            "close": {"IVV": 102.0, "TLT": 51.0, "GLD": 10.0}}]
    snaps = [ma.snapshot_state(_snap("2026-06-10", [1] * N, closes={"IVV": 204.0, "TLT": 102.0, "GLD": 20.0})),
             ma.snapshot_state(_snap("2026-06-11", [0] * N, closes={"IVV": 206.04, "GLD": 20.0}))]
    rows = ma.merge_states(rec, snaps)
    assert [r["date"] for r in rows] == ["2026-06-09", "2026-06-10", "2026-06-11"]
    assert rows[1]["source"] == "snapshot" and rows[1]["flags"] == [1] * N   # señales del snapshot
    assert rows[1]["close"] == {"IVV": 102.0, "TLT": 51.0, "GLD": 10.0}       # closes del pickle
    assert rows[2]["close"]["IVV"] == pytest.approx(103.02)                   # 206.04 × (102 / 204)
    assert rows[2]["close"]["TLT"] == 51.0                                    # faltaba en el snapshot → ffill
    assert rows[2]["close"]["GLD"] == pytest.approx(10.0)


# ── punta a punta solo con snapshots (sin pickle) ──────────────────────────────
def test_analog_facts_snapshots_only(tmp_path, monkeypatch):
    monkeypatch.setattr(mi, "SNAPSHOT_DIR", tmp_path)
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-01-01", periods=70)]
    today = [1, -1, 0, 1, 0, -1, 1, 0, 0, -1, 1, 0, 1, -1, 0, 1, -1]
    other = [-f if f else 1 for f in today]                        # muy distinto
    near = today[:]; near[0] = 0                                   # un flag distinto
    for i, d in enumerate(dates):
        flags = today if i in (11, 12, 55, 69) else near if i == 41 else other
        closes = {"IVV": 100 * 1.01 ** i, "TLT": 100.0, "GLD": 100.0}
        # Trend alternado: racha de 1 sesión todos los días; los días impares
        # quedan del mismo lado del 70% que hoy (69) → racha idéntica
        (tmp_path / f"{d}.json").write_text(json.dumps(_snap(d, flags, trend=0.8 if i % 2 else 0.5, closes=closes)))
    out = ma.analog_facts(prices_path=tmp_path / "nope.pkl", cache_path=tmp_path / "cache.json")
    assert out["history"]["source"].startswith("daily snapshots only")
    assert out["today"]["session"] == dates[69] and out["today"]["simulated_regime_agrees_with_dashboard"]
    assert [a["date"] for a in out["analogs"][:2]] == [dates[11], dates[41]]
    a0 = out["analogs"][0]
    assert a0["distance"] == 0 and a0["episode_sessions"] == 2 and a0["episode_best_match_day"] == dates[11]
    assert a0["fwd_20"]["ivv_pct"] == pytest.approx((1.01 ** 20 - 1) * 100, abs=0.01)
    assert a0["fwd_20"]["regime_flipped"] is False
    assert out["analogs"][1]["distance"] == pytest.approx(ma.ANALOG_W_FLAGS)
    assert all(a["date"] < dates[49] for a in out["analogs"])      # nada de las últimas 20 sesiones (55 queda fuera)
    assert out["summary"]["n_analogs"] == len(out["analogs"]) <= ma.ANALOG_K
    assert out["summary"]["pct_with_regime_flip_within_20"] == 0
    json.dumps(out)                                                # serializable tal cual va al payload


def test_analog_facts_returns_none_when_history_is_too_short(tmp_path, monkeypatch):
    monkeypatch.setattr(mi, "SNAPSHOT_DIR", tmp_path)
    for d in pd.bdate_range("2026-01-01", periods=ma.ANALOG_EXCLUDE_RECENT + 1):
        d = d.strftime("%Y-%m-%d")
        (tmp_path / f"{d}.json").write_text(json.dumps(_snap(d, [0] * N)))
    assert ma.analog_facts(prices_path=tmp_path / "nope.pkl", cache_path=tmp_path / "cache.json") is None
