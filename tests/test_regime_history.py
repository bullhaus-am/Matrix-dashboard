"""Historial del régimen: una entrada por sesión (merge_regime_history).

La histéresis de compute_regime cuenta entradas consecutivas, así que una
sesión que falta en regime_history.json corre el flip. Sin red: solo listas.
"""
from datetime import date
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import generate_dashboard as gd  # noqa: E402


def _fmt(d: date) -> str:
    return d.strftime(gd.REGIME_DATE_FMT)


def _sessions(start: str, n: int) -> list:
    return [ts.date() for ts in pd.bdate_range(start, periods=n)]


def _saved(days, score=0.3):
    return [{"date": _fmt(d), "score": score, "regime": "Bonds"} for d in days]


def test_appends_today_when_it_is_a_new_session():
    days = _sessions("2026-11-02", 4)
    hist, filled = gd.merge_regime_history(_saved(days[:3]), _fmt(days[3]), 0.8)
    assert [e["date"] for e in hist] == [_fmt(d) for d in days]
    assert hist[-1] == {"date": _fmt(days[3]), "score": 0.8}
    assert filled == []


def test_rerun_on_the_same_session_refreshes_the_score_without_duplicating():
    days = _sessions("2026-11-02", 3)
    hist, filled = gd.merge_regime_history(_saved(days), _fmt(days[-1]), 0.9)
    assert len(hist) == 3
    assert hist[-1]["score"] == 0.9
    assert filled == []


def test_fills_sessions_missing_between_saved_entries():
    days = _sessions("2026-11-02", 8)
    saved = _saved([d for i, d in enumerate(days[:7]) if i not in (2, 5)])   # faltan dos
    recomputed = [(pd.Timestamp(d), 0.7) for d in days]
    hist, filled = gd.merge_regime_history(saved, _fmt(days[7]), 0.5, recomputed)
    assert [e["date"] for e in hist] == [_fmt(d) for d in days]              # una por sesión, en orden
    assert filled == [days[2], days[5]]
    by_date = {e["date"]: e for e in hist}
    for d in (days[2], days[5]):
        assert by_date[_fmt(d)] == {"date": _fmt(d), "score": 0.7, "backfilled": True}


def test_saved_scores_are_never_replaced_by_recomputed_ones():
    days = _sessions("2026-11-02", 5)
    recomputed = [(pd.Timestamp(d), 1.0) for d in days]
    hist, _ = gd.merge_regime_history(_saved(days[:4], score=0.3), _fmt(days[4]), 0.4, recomputed)
    assert [e["score"] for e in hist] == [0.3, 0.3, 0.3, 0.3, 0.4]           # hoy: el score del día, no el 1.0
    assert not any(e.get("backfilled") for e in hist)


def test_fills_the_tail_when_the_local_history_is_stale():
    """Checkout atrasado: la última entrada es de hace 8 sesiones."""
    days = _sessions("2026-10-19", 19)
    recomputed = [(pd.Timestamp(d), 0.3) for d in days]
    hist, filled = gd.merge_regime_history(_saved(days[:10]), _fmt(days[18]), 0.5, recomputed)
    assert len(hist) == 19
    assert filled == days[10:18]


def test_does_not_extend_the_history_before_its_first_entry():
    days = _sessions("2026-10-01", 10)
    recomputed = [(pd.Timestamp(d), 0.7) for d in days]
    hist, filled = gd.merge_regime_history(_saved(days[5:9]), _fmt(days[9]), 0.5, recomputed)
    assert hist[0]["date"] == _fmt(days[5])
    assert filled == []


def test_seeds_an_empty_history_with_the_recomputed_series():
    days = _sessions("2026-10-01", 30)
    recomputed = [(pd.Timestamp(d), 0.8) for d in days]
    hist, filled = gd.merge_regime_history([], _fmt(days[-1]), 0.8, recomputed)
    assert len(hist) == 30
    assert filled == days[:-1]
    assert "backfilled" not in hist[-1]                                      # hoy es point-in-time
    assert gd.compute_regime([e["score"] for e in hist])[0] == "Stocks"      # 30 sesiones ≥ 70%


def test_backfilled_flag_is_cleared_when_the_session_is_recomputed_live():
    days = _sessions("2026-11-02", 3)
    saved = _saved(days[:2]) + [{"date": _fmt(days[2]), "score": 0.6, "backfilled": True}]
    hist, _ = gd.merge_regime_history(saved, _fmt(days[2]), 0.7)
    assert hist[-1] == {"date": _fmt(days[2]), "score": 0.7}


def test_stale_session_is_inserted_in_order_not_appended():
    """Yahoo sirve datos viejos: la sesión calculada es anterior a la última guardada."""
    days = _sessions("2026-11-02", 5)
    saved = _saved([days[0], days[1], days[3], days[4]])
    hist, filled = gd.merge_regime_history(saved, _fmt(days[2]), 0.9)
    assert [e["date"] for e in hist] == [_fmt(d) for d in days]
    assert hist[2]["score"] == 0.9
    assert filled == []


def test_a_missing_session_no_longer_delays_the_flip():
    """30 sesiones bajo el umbral y después 20 sobre: el flip a Stocks cae en la
    sesión 50. Si una de esas 20 no se corrió, sin relleno el flip se atrasa."""
    days = _sessions("2026-06-01", 50)
    scores = [0.3] * 30 + [0.8] * 20
    saved = [{"date": _fmt(d), "score": s} for d, s in zip(days[:49], scores)]
    del saved[40]                                                            # corrida que no se hizo
    recomputed = [(pd.Timestamp(d), s) for d, s in zip(days, scores)]

    without, _ = gd.merge_regime_history([dict(e) for e in saved], _fmt(days[49]), 0.8)
    assert gd.compute_regime([e["score"] for e in without])[0] == "Bonds"   # solo 19 entradas seguidas

    filled_hist, filled = gd.merge_regime_history(saved, _fmt(days[49]), 0.8, recomputed)
    assert filled == [days[40]]
    assert gd.compute_regime([e["score"] for e in filled_hist])[0] == "Stocks"


def test_trend_scores_by_date_matches_the_chart_history():
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    tickers = ['IVV', 'HYG', 'IEI', 'ISCB', 'ILCB', 'IMCG', 'IMCV', 'RSPD', 'RSPS', 'IDU', 'GLD', 'GSG']
    prices = {t: pd.Series(100 + 0.2 * pd.RangeIndex(n).to_numpy(), index=idx) for t in tickers}
    dated = gd.trend_scores_by_date(prices, 60)
    assert len(dated) == 60
    assert dated[-1][0] == idx[-1] and dated[0][0] == idx[-60]               # calendario de IVV, en orden
    assert dated[-1][1] == gd.compute_trend_score(prices)[0]                 # el último es el score de hoy
    assert gd.build_trend_history(prices, 60) == [
        {"d": d.strftime("%b %d"), "s": s} for d, s in dated]
