"""Tests de la capa determinística de macro_insights.py (sin API)."""
from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import macro_insights as mi  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────────
def _snap(session, matrix_sigs, trend_sub=None, regime="Stocks", regime_days=10, score=0.6, pcts=None):
    names = list(mi.MATRIX_TICKERS)
    pcts = pcts or {}
    matrix = [{"n": n, "sig": s, "f": {"Bullish": 1, "Neutral": 0, "Bearish": -1}[s],
               "pct": pcts.get(n, {"Bullish": 100, "Neutral": 50, "Bearish": 0}[s])}
              for n, s in zip(names, matrix_sigs)]
    etf = {"ticker": "IVV", "name": "S&P", "category": "Country/Region", "close": 1, "cx": 0.1, "cx30": 0.1,
           "yearly": 1, "perf_3m": 1, "perf_1m": 1, "perf_1d": 0, "vs_ema20": 1, "vs_sma50": 1, "atr5_pct": 1,
           "atr_ratio": 1, "net_close": 0, "net_50": 0, "net_20": 0, "net_5": 0, "rank_today": 1, "rank_30ago": 1,
           "rank_climb": 0, "mom_rank": 1}
    return {"session": session, "trend_score": score, "trend_sub_scores": trend_sub or [2, 1, 1, 0, 0, 0, 0, 1, 0],
            "regime": regime, "regime_days": regime_days, "matrix_score": sum(m["f"] for m in matrix),
            "matrix": matrix, "etfs": [etf]}


def _all(sig):
    return [sig] * len(mi.MATRIX_TICKERS)


# ── regime_facts ───────────────────────────────────────────────────────────────
def test_regime_countdown_counts_sessions_below_threshold():
    hist = [{"date": f"2026-01-{d:02d}", "score": 0.8, "regime": "Stocks"} for d in range(1, 11)]
    hist += [{"date": f"2026-02-{d:02d}", "score": 0.6, "regime": "Stocks"} for d in range(1, 7)]
    snap = _snap("2026-02-06", _all("Neutral"), regime="Stocks", regime_days=16, score=0.6)
    r = mi.regime_facts(hist, snap)
    assert r["sessions_below_threshold"] == 6
    assert r["sessions_at_or_above_threshold"] == 0
    assert r["flip_would_be_to"] == "Bonds"
    assert r["sessions_to_flip_if_streak_holds"] == mi.REGIME_CONFIRM_DAYS - 6


def test_regime_countdown_is_none_when_streak_is_on_regime_side():
    hist = [{"date": f"2026-01-{d:02d}", "score": 0.8, "regime": "Stocks"} for d in range(1, 11)]
    snap = _snap("2026-01-10", _all("Neutral"), regime="Stocks", score=0.8)
    r = mi.regime_facts(hist, snap)
    assert r["sessions_to_flip_if_streak_holds"] is None
    assert r["sessions_at_or_above_threshold"] == 10


def test_regime_average_excludes_first_and_current_runs():
    hist = []
    for regime, n in [("Bonds", 5), ("Stocks", 30), ("Bonds", 10), ("Stocks", 3)]:
        for i in range(n):
            hist.append({"date": f"2026-01-01", "score": 0.8, "regime": regime})
    snap = _snap("2026-01-01", _all("Neutral"), regime="Stocks", regime_days=3)
    r = mi.regime_facts(hist, snap)
    assert r["avg_completed_regime_sessions"] == 20            # (30 + 10) / 2
    assert r["avg_completed_stocks_regime_sessions"] == 30
    assert r["pct_of_avg_same_regime"] == 10                   # 3 / 30
    assert [f["to"] for f in r["last_flips"]] == ["Stocks", "Bonds", "Stocks"]


# ── matrix_facts ───────────────────────────────────────────────────────────────
def test_matrix_streak_and_last_flip():
    names = list(mi.MATRIX_TICKERS)
    sigs = [_all("Neutral") for _ in range(6)]
    for k in (4, 5):                                           # las 2 últimas sesiones: par 0 Bullish
        sigs[k] = sigs[k][:]; sigs[k][0] = "Bullish"
    snaps = [_snap(f"2026-01-0{i+1}", s) for i, s in enumerate(sigs)]
    m = mi.matrix_facts(snaps)
    p0 = m["pairs"][0]
    assert p0["signal"] == "Bullish"
    assert p0["sessions_in_current_signal"] == 2
    assert p0["last_flip"] == {"session": "2026-01-05", "from": "Neutral", "to": "Bullish"}
    assert p0["streak_reaches_start_of_history"] is False
    assert names[0] in m["pairs_flipped_last_5_sessions"]
    p1 = m["pairs"][1]
    assert p1["sessions_in_current_signal"] == 6 and p1["last_flip"] is None
    assert p1["streak_reaches_start_of_history"] is True
    assert m["score_last_20_sessions"] == [0, 0, 0, 0, 1, 1]


def test_matrix_edge_flags_only_for_neutral_pairs():
    names = list(mi.MATRIX_TICKERS)
    pcts = {names[0]: 3, names[1]: 97, names[2]: 50}
    sigs = _all("Neutral"); sigs[3] = "Bullish"
    snaps = [_snap("2026-01-01", sigs, pcts={**pcts, names[3]: 100})]
    pairs = mi.matrix_facts(snaps)["pairs"]
    assert pairs[0]["edge_flag"] == "near Bearish edge"
    assert pairs[1]["edge_flag"] == "near Bullish edge"
    assert pairs[2]["edge_flag"] is None
    assert pairs[3]["edge_flag"] is None                       # Bullish: pct clavado en 100, sin flag


# ── trend_component_facts ──────────────────────────────────────────────────────
def test_trend_component_last_change_session():
    subs = [[2, 1, 1, 0, 0, 0, 0, 1, 0]] * 3 + [[2, 1, 0, 0, 0, 0, 0, 1, 0]] * 2
    snaps = [_snap(f"2026-01-0{i+1}", _all("Neutral"), trend_sub=s) for i, s in enumerate(subs)]
    comps = mi.trend_component_facts(snaps)
    hy = next(c for c in comps if c["component"] == "High Yield vs Treasuries")
    assert hy["points_now"] == 0
    assert hy["last_session_with_different_value"] == "2026-01-03"
    ivv = comps[0]
    assert ivv["max_points"] == 2 and ivv["last_session_with_different_value"] is None


# ── point-in-time ──────────────────────────────────────────────────────────────
def test_load_snapshots_respects_upto(tmp_path, monkeypatch):
    monkeypatch.setattr(mi, "SNAPSHOT_DIR", tmp_path)
    for d in ("2026-01-05", "2026-01-06", "2026-01-07"):
        (tmp_path / f"{d}.json").write_text(json.dumps(_snap(d, _all("Neutral"))))
    snaps = mi.load_snapshots("2026-01-06")
    assert [s["session"] for s in snaps] == ["2026-01-05", "2026-01-06"]


def test_load_regime_history_parses_dates_and_filters(tmp_path, monkeypatch):
    f = tmp_path / "regime_history.json"
    f.write_text(json.dumps([{"date": "Jan 05, 2026", "score": 0.8, "regime": "Stocks"},
                             {"date": "Jan 06, 2026", "score": 0.6, "regime": "Stocks"},
                             {"date": "Jan 07, 2026", "score": 0.5, "regime": "Stocks"}]))
    monkeypatch.setattr(mi, "REGIME_HISTORY_FILE", f)
    h = mi.load_regime_history("2026-01-06")
    assert [e["date"] for e in h] == ["2026-01-05", "2026-01-06"]


# ── liquidez (parser de dashboard_data.js) ─────────────────────────────────────
def test_liquidity_facts_parses_js_consts_point_in_time(tmp_path):
    js = tmp_path / "dashboard_data.js"
    oas = [{"date": f"2026-01-{d:02d}", "ig": 80 + d, "hy": 300 + d, "bbb": 100} for d in range(1, 31)]
    raw = [{"date": f"2026-01-{d:02d}", "liq1": 5000 + d, "liq2": 2000, "tga": 900, "rrp": 0, "walcl": 6700} for d in (7, 14, 21, 28)]
    js.write_text(f"const RAW_DATA = {json.dumps(raw)};\nconst OAS_DATA = {json.dumps(oas)};\nconst RISK_DATA = [];\n")
    out = mi.liquidity_facts(js, "2026-01-20")
    assert out["credit_oas_bp"]["as_of"] == "2026-01-20"
    assert out["credit_oas_bp"]["hy"] == 320 and out["credit_oas_bp"]["hy_chg_5d"] == 5
    assert [w["date"] for w in out["fed_liquidity_weekly_usd_bn"]["last_8_weeks"]] == ["2026-01-07", "2026-01-14"]
    assert "bullhaus_6_pilares" not in out


def test_liquidity_facts_none_when_file_missing(tmp_path):
    assert mi.liquidity_facts(tmp_path / "nope.js", "2026-01-01") is None


# ── perfiles ───────────────────────────────────────────────────────────────────
def test_resolve_profile_auto_picks_weekly_on_friday():
    assert mi.resolve_profile("auto", "2026-10-16") == "weekly"     # viernes
    assert mi.resolve_profile("auto", "2026-10-13") == "daily"      # martes
    assert mi.resolve_profile("weekly", "2026-10-13") == "weekly"   # explícito gana
    assert mi.resolve_profile("daily", "2026-10-16") == "daily"


def test_system_prompt_per_profile_shares_model_context():
    d, w = mi.system_prompt("daily"), mi.system_prompt("weekly")
    assert d.startswith(mi.ROLE_AND_MODEL) and w.startswith(mi.ROLE_AND_MODEL)
    assert "daily desk note" in d and "daily desk note" not in w
    assert "weekly deep read" in w and "weekly deep read" not in d
    for p in mi.PROFILES.values():
        assert p["model"] in mi.PRICING


def test_previous_insights_returns_last_n_oldest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(mi, "INSIGHTS_DIR", tmp_path)
    for d in ("2026-10-01", "2026-10-02", "2026-10-03", "2026-10-06"):
        (tmp_path / f"{d}.json").write_text(json.dumps({
            "session": d, "meta": {"profile": "daily"},
            "output": {"headline": f"h{d}", "day_type": "continuity",
                       "breakout_light": {"color": "yellow", "size_multiplier": "0.5x", "tilt_long": [], "tilt_avoid": [], "rationale": ""},
                       "insights": [{"title": "t", "conviction": "low", "status": "new", "instruments": [{"ticker": "IVV"}]}]}}))
    (tmp_path / "2026-10-03.preview.json").write_text("{}")          # los preview no cuentan
    prev = mi.previous_insights("2026-10-06", n=2)
    assert [p["session"] for p in prev] == ["2026-10-02", "2026-10-03"]
    assert prev[-1]["breakout_light"] == {"color": "yellow", "size_multiplier": "0.5x"}
    assert prev[-1]["insights"][0]["instruments"] == ["IVV"]
    assert mi.previous_insights("2026-10-01") == []


def test_render_tolerates_missing_new_fields_and_empty_sections():
    out = {"headline": "h", "regime_read": "r", "insights": [], "watch_triggers": [], "contradictions": [],
           "blind_spots": "b", "changes_vs_previous": "", "dashboard_one_liner": "o"}
    md = mi.to_markdown(out, {"profile": "daily"}, "2026-10-13")
    html = mi.html_fragment(out, {"profile": "daily"}, "2026-10-13")
    assert "## Insights" not in md and "Disparador" not in md and "Contradicc" not in md
    assert 'ai-block-title">Insights<' not in html and "mi-table" not in html
    out["breakout_light"] = {"color": "red", "size_multiplier": "0", "tilt_long": ["XLE"], "tilt_avoid": ["IWC"], "rationale": "x"}
    out["day_type"] = "continuity"
    md = mi.to_markdown(out, {"profile": "daily"}, "2026-10-13")
    html = mi.html_fragment(out, {"profile": "daily"}, "2026-10-13")
    assert "Semáforo breakouts: ROJO · tamaño 0" in md and "long: XLE" in md and "evitar: IWC" in md
    assert "mi-light-red" in html and "día de continuidad" in html


# ── esquema de salida: lo mínimo para que structured outputs no rechace ────────
def test_output_schema_is_closed_everywhere():
    def walk(node):
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False
            assert set(node["required"]) == set(node["properties"])
            for v in node["properties"].values():
                walk(v)
        elif node.get("type") == "array":
            walk(node["items"])
    walk(mi.OUTPUT_SCHEMA)
