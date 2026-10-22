"""
macro_insights.py
=================
Insights macro / inter-market sobre el modelo Matrix, escritos por Claude.

A diferencia del análisis AI del dashboard (que mira la tabla de 120 ETFs y
elige "oportunidades"), este script trabaja la capa MACRO del modelo:

  * dónde está el régimen dentro de su ciclo y cuánto falta para un flip
    (histéresis de 20 sesiones contada sesión a sesión),
  * qué pares de la Matrix flipearon, cuáles llevan N sesiones en su señal y
    cuáles están al borde de la banda ±1.5% (a punto de flipear),
  * qué dicen los 9 componentes del Trend Score y cómo cambiaron,
  * lecturas cross-asset deterministas (curva, crédito, inflación, apetito de
    riesgo, cíclicos vs defensivos, geografía) armadas desde el snapshot,
  * opcionalmente liquidez Fed + OAS del Liquidity Dashboard (si existe).

Todo lo numérico lo calcula Python a partir de `snapshots/*.json` y
`regime_history.json` (point-in-time: con --session se usa solo lo que se
sabía ese día, sin lookahead). Claude recibe esos hechos ya digeridos y
devuelve un JSON con esquema fijo (structured outputs), que se renderiza a
Markdown y a un HTML con el estilo del dashboard.

Dos perfiles (--profile, default auto):
    daily   nota corta de mesa: día de transición/continuidad, régimen, semáforo
            de breakouts (color + tamaño + tilts), ≤3 insights, 1 contradicción,
            1 disparador, 1 punto ciego. claude-opus-5-5, effort medium.
    weekly  lectura larga: 3-6 insights, varios disparadores y contradicciones,
            cambios vs las diarias de la semana. claude-fable-5-1, effort high.
    auto    weekly si la sesión es viernes, daily el resto.

Uso:
    python3 macro_insights.py                        # último snapshot, perfil auto
    python3 macro_insights.py --profile weekly       # forzar la lectura larga
    python3 macro_insights.py --session 2026-09-15   # replay histórico
    python3 macro_insights.py --dry-run              # muestra perfil + prompt + payload, no llama
    python3 macro_insights.py --model claude-sonnet-5-5 --effort low   # override del perfil
    python3 macro_insights.py --web                  # habilita web search (máx 5 usos)
    python3 macro_insights.py --render insights/2026-10-13.json   # re-renderiza md/html

Salida: insights/YYYY-MM-DD.json  (estructurado, con usage y costo)
        insights/YYYY-MM-DD.md    (legible)
        insights/YYYY-MM-DD.html  (preview standalone, estilo del dashboard)

Requiere: pip install -U "anthropic>=2.1"   (solo para llamar a la API;
          --dry-run y --render funcionan sin el SDK).
Key:      ANTHROPIC_API_KEY en el entorno o en .env (mismo .env del dashboard).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import sys
from datetime import datetime, date
from pathlib import Path

HERE = Path(__file__).parent
SNAPSHOT_DIR = HERE / "snapshots"
REGIME_HISTORY_FILE = HERE / "regime_history.json"
INSIGHTS_DIR = HERE / "insights"
LIQUIDITY_DATA_JS = Path.home() / "Liquidity Dashboard" / "dashboard_data.js"

DEFAULT_PROFILE = "auto"    # auto: viernes → weekly (Fable, nota larga); resto → daily (Opus, nota corta)
TREND_THRESHOLD = 0.7
REGIME_CONFIRM_DAYS = 20
MATRIX_BAND_PCT = 1.5
NEAR_EDGE_PCT = 15          # Neutral con pct <= 15 o >= 85 → "al borde de flipear"
LIQUIDITY_MAX_AGE_DAYS = 10 # pilares más viejos que esto se marcan como stale

# $/MTok: input, output, cache write (5 min), cache read
PRICING = {
    "claude-fable-5-1":  (10.0, 50.0, 12.5, 0.25),
    "claude-opus-5-5":   (4.0,  20.0, 5.0,  0.20),
    "claude-sonnet-5-5": (2.0,  10.0, 2.5,  0.20),
}

# Mismo orden que compute_trend_score() en generate_dashboard.py
TREND_COMPONENTS = [
    ("IVV vs MA100",                 "IVV > MA100",                    2),
    ("IVV MA100 momentum",           "MA100 hoy > MA100 hace 10 sesiones", 1),
    ("High Yield vs Treasuries",     "HYG/IEI > su MA100",             1),
    ("Small vs Large",               "ISCB/ILCB > su MA100 (proxy IJR/IVV)", 1),
    ("Growth vs Value",              "IMCG/IMCV > su MA100",           1),
    ("Discretionary vs Staples",     "RSPD/RSPS > su MA100",           1),
    ("Services vs Utilities",        "RSPD/IDU > su MA100",            1),
    ("Stocks vs Gold",               "IVV/GLD > su MA100",             1),
    ("Stocks vs Commodities",        "IVV/GSG > su MA100",             1),
]

# Nombre del par (clave del snapshot) → tickers, solo para mostrar
MATRIX_TICKERS = {
    "S&P Momentum":               "IVV vs su MA100",
    "High Yield vs Treasuries":   "HYG/IEI",
    "Small Cap vs Large Cap":     "ISCB/ILCB (proxy IJR/IVV)",
    "Growth vs Value":            "IMCG/IMCV",
    "IBD 50 vs S&P":              "FFTY/IVV",
    "Consumer Discr. vs Staples": "RSPD/RSPS",
    "Consumer Svcs vs Utils":     "RSPD/IDU",
    "Emerging Markets vs S&P":    "EEM/IVV",
    "High Beta Growth vs S&P":    "ARKK/IVV",
    "QQQJ vs QQQ":                "QQQJ/QQQ",
    "Argentina vs S&P":           "ARGT/IVV",
    "S&P vs Treasuries 20YR":     "IVV/TLT",
    "Silver vs Gold":             "SLV/GLD",
    "Copper vs Gold":             "CPER/GLD",
    "Wood vs Gold":               "WOOD/GLD",
    "Stocks vs Gold":             "IVV/GLD",
    "Stocks vs Commodities":      "IVV/GSG",
}

# Grupos para las lecturas cross-asset deterministas (tickers del universo)
MACRO_GROUPS = {
    "duration_ladder":   ["SHV", "SHY", "IEI", "IEF", "TLH", "TLT", "TIP", "MBB"],
    "credit":            ["HYG", "SHYG", "GHYG", "HYGH", "SLQD", "EMB", "EMHY"],
    "metals_commod":     ["GLD", "SLV", "GDX", "CPER", "GSG", "WOOD", "VEGI"],
    "risk_appetite":     ["ARKK", "QQQJ", "IWC", "FFTY", "MTUM", "SPLV", "GBTC", "ETHE"],
    "cyclicals":         ["XLI", "XLY", "XLF", "XLB", "XLE", "ITB", "IYT", "IAT"],
    "defensives":        ["XLU", "XLP", "XLV", "XLRE", "IDU", "IHE"],
    "us_core":           ["IVV", "QQQ", "ILCG", "ILCV", "IMCB", "ISCB", "EUSA"],
    "global":            ["URTH", "EEM", "EEMA", "ILF", "MCHI", "INDA", "EWJ", "EWG", "EWZ", "ARGT", "EWY", "EWT"],
}

log = logging.getLogger("macro_insights")


# ── .env mínimo (misma convención que generate_dashboard.py) ──────────────────
def _load_dotenv():
    p = HERE / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ── Carga de datos ─────────────────────────────────────────────────────────────
def load_snapshots(upto: str | None = None) -> list[dict]:
    """Snapshots ordenados por sesión, hasta `upto` inclusive (point-in-time)."""
    files = sorted(SNAPSHOT_DIR.glob("????-??-??.json"))
    snaps = []
    for f in files:
        if upto and f.stem > upto:
            continue
        try:
            snaps.append(json.loads(f.read_text()))
        except Exception as e:
            log.warning("snapshot ilegible %s: %s", f.name, e)
    if not snaps:
        raise SystemExit(f"No hay snapshots en {SNAPSHOT_DIR}" + (f" hasta {upto}" if upto else ""))
    return snaps


def _parse_regime_date(s: str) -> str:
    return datetime.strptime(s, "%b %d, %Y").date().isoformat()


def load_regime_history(upto: str) -> list[dict]:
    """regime_history.json con fechas ISO, hasta `upto` inclusive."""
    if not REGIME_HISTORY_FILE.exists():
        return []
    out = []
    for e in json.loads(REGIME_HISTORY_FILE.read_text()):
        try:
            d = _parse_regime_date(e["date"])
        except Exception:
            continue
        if d <= upto:
            out.append({"date": d, "score": float(e["score"]), "regime": e.get("regime")})
    return out


# ── Capa determinística: hechos del modelo ─────────────────────────────────────
def regime_facts(history: list[dict], snap: dict) -> dict:
    """Dónde está el régimen en su ciclo + cuánto falta para flipear."""
    scores = [h["score"] for h in history]
    regime = snap["regime"]
    streak_above = streak_below = 0
    for s in reversed(scores):
        if s >= TREND_THRESHOLD:
            if streak_below:
                break
            streak_above += 1
        else:
            if streak_above:
                break
            streak_below += 1

    # Un flip requiere 20 sesiones seguidas del lado contrario al régimen actual
    if regime == "Stocks":
        flip_to, flip_streak = "Bonds", streak_below
    else:
        flip_to, flip_streak = "Stocks", streak_above
    sessions_to_flip = max(0, REGIME_CONFIRM_DAYS - flip_streak) if flip_streak else None

    # Duración de regímenes completos (excluye el primero, que es artefacto del
    # arranque en Bonds, y el actual, que sigue abierto)
    runs, flips = [], []
    if history:
        cur, n, start = history[0]["regime"], 0, history[0]["date"]
        for h in history:
            if h["regime"] == cur:
                n += 1
            else:
                runs.append((cur, n))
                flips.append({"date": h["date"], "to": h["regime"]})
                cur, n, start = h["regime"], 1, h["date"]
        runs.append((cur, n))
    completed = runs[1:-1]
    avg_all = round(statistics.mean(d for _, d in completed)) if completed else None
    avg_same = [d for r, d in completed if r == regime]
    avg_same = round(statistics.mean(avg_same)) if avg_same else None

    last60 = scores[-60:]
    return {
        "regime": regime,
        "regime_days": snap["regime_days"],
        "trend_score_pct": int(round(snap["trend_score"] * 100)),
        "threshold_pct": int(TREND_THRESHOLD * 100),
        "hysteresis_sessions": REGIME_CONFIRM_DAYS,
        "sessions_below_threshold": streak_below,
        "sessions_at_or_above_threshold": streak_above,
        "flip_would_be_to": flip_to,
        "sessions_to_flip_if_streak_holds": sessions_to_flip,
        "avg_completed_regime_sessions": avg_all,
        f"avg_completed_{regime.lower()}_regime_sessions": avg_same,
        "pct_of_avg_same_regime": (round(snap["regime_days"] / avg_same * 100) if avg_same else None),
        "last_flips": flips[-5:],
        "trend_last_20_sessions_pct": [int(round(s * 100)) for s in scores[-20:]],
        "trend_60d_min_pct": int(round(min(last60) * 100)) if last60 else None,
        "trend_60d_max_pct": int(round(max(last60) * 100)) if last60 else None,
    }


def trend_component_facts(snaps: list[dict]) -> list[dict]:
    """Los 9 componentes: valor hoy, hace 5 y 20 sesiones, última vez que cambió."""
    cur = snaps[-1]["trend_sub_scores"]
    out = []
    for i, (name, rule, pts) in enumerate(TREND_COMPONENTS):
        def at(k):
            return snaps[k]["trend_sub_scores"][i] if -k <= len(snaps) and len(snaps[k]["trend_sub_scores"]) > i else None
        last_change = None
        for s in reversed(snaps[:-1]):
            if len(s["trend_sub_scores"]) > i and s["trend_sub_scores"][i] != cur[i]:
                last_change = s["session"]
                break
        out.append({
            "component": name, "rule": rule, "max_points": pts,
            "points_now": cur[i],
            "points_5_sessions_ago": at(-6) if len(snaps) >= 6 else None,
            "points_20_sessions_ago": at(-21) if len(snaps) >= 21 else None,
            "last_session_with_different_value": last_change,
        })
    return out


def matrix_facts(snaps: list[dict]) -> dict:
    """Pares de la Matrix con streak, último flip y proximidad al borde."""
    cur = snaps[-1]["matrix"]
    by_session = {s["session"]: {m["n"]: m for m in s["matrix"]} for s in snaps}
    sessions = [s["session"] for s in snaps]
    pairs = []
    for m in cur:
        name = m["n"]
        streak, last_flip = 0, None
        for sess in reversed(sessions):
            prev = by_session[sess].get(name)
            if prev is None:
                break
            if prev["sig"] == m["sig"]:
                streak += 1
            else:
                last_flip = {"session": sessions[len(sessions) - streak], "from": prev["sig"], "to": m["sig"]}
                break
        pct5 = None
        if len(snaps) >= 6:
            p = by_session[sessions[-6]].get(name)
            pct5 = p["pct"] if p else None
        edge = None
        if m["sig"] == "Neutral":
            if m["pct"] <= NEAR_EDGE_PCT:
                edge = "near Bearish edge"
            elif m["pct"] >= 100 - NEAR_EDGE_PCT:
                edge = "near Bullish edge"
        pairs.append({
            "pair": name, "tickers": MATRIX_TICKERS.get(name, ""),
            "signal": m["sig"], "band_position_pct": m["pct"],
            "band_position_pct_5_sessions_ago": pct5,
            "sessions_in_current_signal": streak,
            # racha que llega al primer snapshot: es "al menos N", no N exacto
            "streak_reaches_start_of_history": last_flip is None and streak == len(snaps),
            "last_flip": last_flip,
            "edge_flag": edge,
        })

    score_hist = [s["matrix_score"] for s in snaps[-20:]]
    flips_5 = [p["pair"] for p in pairs if p["last_flip"] and p["sessions_in_current_signal"] <= 5]
    flips_20 = [p["pair"] for p in pairs if p["last_flip"] and p["sessions_in_current_signal"] <= 20]
    return {
        "score_now": snaps[-1]["matrix_score"],
        "bullish": sum(1 for m in cur if m["f"] > 0),
        "neutral": sum(1 for m in cur if m["f"] == 0),
        "bearish": sum(1 for m in cur if m["f"] < 0),
        "band": f"±{MATRIX_BAND_PCT}% alrededor de la MA100 del ratio; band_position_pct solo informa dentro de la banda (0 = borde Bearish, 100 = borde Bullish; fuera de la banda se clava en 0/100)",
        "score_last_20_sessions": score_hist,
        "pairs_flipped_last_5_sessions": flips_5,
        "pairs_flipped_last_20_sessions": flips_20,
        "pairs": pairs,
    }


def _row(e: dict) -> dict:
    return {
        "t": e["ticker"], "cat": e["category"], "rank": e["rank_today"],
        "climb30": e["rank_climb"], "cx": e["cx"],
        "m1": e["perf_1m"], "m3": e["perf_3m"], "y": e["yearly"],
        "vs_ema20": e["vs_ema20"], "vs_sma50": e["vs_sma50"],
        "net50": e["net_50"], "net20": e["net_20"], "net5": e["net_5"],
        "atr_ratio": e["atr_ratio"], "mom_rank": e.get("mom_rank"),
    }


def cross_asset_facts(snap: dict) -> dict:
    """Breadth por categoría, líderes/rezagados y grupos macro (filas crudas)."""
    etfs = snap["etfs"]
    by_t = {e["ticker"]: e for e in etfs}
    cats = {}
    for e in etfs:
        cats.setdefault(e["category"], []).append(e)

    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.median(xs), 2) if xs else None

    breadth = []
    for cat, rows in cats.items():
        rows_sorted = sorted(rows, key=lambda r: r["rank_today"])
        breadth.append({
            "category": cat, "n": len(rows),
            "pct_above_sma50": round(100 * sum(1 for r in rows if r["vs_sma50"] > 0) / len(rows)),
            "pct_above_ema20": round(100 * sum(1 for r in rows if r["vs_ema20"] > 0) / len(rows)),
            "median_cx": med([r["cx"] for r in rows]),
            "median_perf_1m": med([r["perf_1m"] for r in rows]),
            "median_net20": med([r["net_20"] for r in rows]),
            "leaders": [_row(r) for r in rows_sorted[:3]],
            "laggards": [_row(r) for r in rows_sorted[-3:]],
        })

    groups = {g: [_row(by_t[t]) for t in ts if t in by_t] for g, ts in MACRO_GROUPS.items()}

    def spread(a, b):
        if a in by_t and b in by_t:
            return {"pair": f"{a} - {b}", "m1_diff": round(by_t[a]["perf_1m"] - by_t[b]["perf_1m"], 2),
                    "m3_diff": round(by_t[a]["perf_3m"] - by_t[b]["perf_3m"], 2)}
        return None

    spreads = [s for s in [
        spread("TLT", "SHY"),   # duración: >0 = bull steepening/rally de largos
        spread("HYG", "IEF"),   # crédito vs duración
        spread("TIP", "IEF"),   # breakevens proxy
        spread("EMB", "IEF"),   # EM soberano
        spread("GDX", "GLD"),   # beta de mineras
        spread("SLV", "GLD"),
        spread("CPER", "GLD"),
        spread("EEM", "IVV"),
        spread("ISCB", "ILCB"),
        spread("ILCG", "ILCV"),
        spread("XLY", "XLP"),
        spread("XLI", "XLU"),
        spread("QQQ", "IVV"),
        spread("GBTC", "IVV"),
    ] if s]

    top_climbers = sorted(etfs, key=lambda e: -e["rank_climb"])[:8]
    top_fallers = sorted(etfs, key=lambda e: e["rank_climb"])[:8]
    return {
        "universe_n": len(etfs),
        "breadth_by_category": breadth,
        "macro_groups": groups,
        "spreads_1m_3m": spreads,
        "biggest_rank_climbers_30d": [_row(e) for e in top_climbers],
        "biggest_rank_fallers_30d": [_row(e) for e in top_fallers],
    }


def etf_table_compact(snap: dict) -> list[dict]:
    """Toda la tabla, compacta, ordenada por rank (para cross-check del modelo).

    Menos campos que _row(): la tabla entera son ~130 filas y pesa la mitad
    del input; los grupos macro y líderes/rezagados ya llevan el detalle."""
    return [{"t": e["ticker"], "cat": e["category"], "rank": e["rank_today"], "climb30": e["rank_climb"],
             "cx": e["cx"], "m1": e["perf_1m"], "m3": e["perf_3m"], "vs_sma50": e["vs_sma50"], "net20": e["net_20"]}
            for e in sorted(snap["etfs"], key=lambda e: e["rank_today"])]


# ── Liquidez (opcional, del Liquidity Dashboard) ───────────────────────────────
def _js_const(text: str, name: str):
    """Extrae `const NAME = <json>` de dashboard_data.js con el decoder de json."""
    i = text.find(f"const {name} = ")
    if i < 0:
        return None
    i += len(f"const {name} = ")
    try:
        val, _ = json.JSONDecoder().raw_decode(text[i:])
        return val
    except Exception as e:
        log.warning("no pude parsear %s: %s", name, e)
        return None


def liquidity_facts(path: Path, session: str) -> dict | None:
    if not path.exists():
        return None
    text = path.read_text()
    out = {"source": str(path)}

    oas = _js_const(text, "OAS_DATA") or []
    oas = sorted([o for o in oas if o.get("date", "") <= session], key=lambda o: o["date"])
    if oas:
        last = oas[-1]
        def chg(k, n):
            return round(last[k] - oas[-1 - n][k], 1) if len(oas) > n else None
        out["credit_oas_bp"] = {
            "as_of": last["date"],
            "ig": last["ig"], "ig_chg_5d": chg("ig", 5), "ig_chg_20d": chg("ig", 20),
            "hy": last["hy"], "hy_chg_5d": chg("hy", 5), "hy_chg_20d": chg("hy", 20),
            "bbb": last["bbb"], "bbb_chg_20d": chg("bbb", 20),
        }

    raw = _js_const(text, "RAW_DATA") or []
    raw = sorted([r for r in raw if r.get("date", "") <= session], key=lambda r: r["date"])
    if raw:
        out["fed_liquidity_weekly_usd_bn"] = {
            "formula": "net_liquidity = WALCL - RRP - TGA + SRF + SWPT - treasury_net_issuance; reserves_adj = reserves + RRP - TGA",
            "last_8_weeks": [{"date": r["date"], "net_liquidity": r.get("liq1"), "reserves_adj": r.get("liq2"),
                              "tga": r.get("tga"), "rrp": r.get("rrp"), "walcl": r.get("walcl")} for r in raw[-8:]],
        }

    pil = _js_const(text, "PILARES_DATA") or []
    pil = sorted([p for p in pil if p.get("date", "") <= session], key=lambda p: p["date"])
    if pil:
        last = pil[-1]
        age = (date.fromisoformat(session) - date.fromisoformat(last["date"])).days
        entry = {"as_of": last["date"], "regime": last.get("regime"), "score": last.get("overall_score"),
                 "max_score": last.get("max_score"),
                 "pilares": {k: v.get("score") for k, v in (last.get("pilares") or {}).items()}}
        if age > LIQUIDITY_MAX_AGE_DAYS:
            entry["stale"] = f"{age} días viejo — tratar solo como contexto histórico"
        vol = (last.get("pilares") or {}).get("volatilidad", {}).get("sub", {})
        if vol:
            entry["vix_level"] = vol.get("vix_level")
            entry["vix_delta_20d"] = vol.get("vix_delta_20d")
        out["bullhaus_6_pilares"] = entry

    return out if len(out) > 1 else None


# ── Continuidad: insights previos ──────────────────────────────────────────────
def previous_insights(session: str, n: int = 1) -> list[dict]:
    """Resumen de las últimas `n` notas anteriores a `session` (la más vieja primero).

    La diaria recibe la anterior (diaria o semanal); la semanal recibe las
    cinco de la semana. Solo cuentan los archivos YYYY-MM-DD.json (los
    .preview.json quedan afuera)."""
    files = sorted(INSIGHTS_DIR.glob("????-??-??.json"))
    prev = [f for f in files if f.stem < session][-n:]
    out = []
    for f in prev:
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        o = d.get("output") or d
        light = o.get("breakout_light") or {}
        out.append({
            "session": f.stem,
            "profile": (d.get("meta") or {}).get("profile"),
            "day_type": o.get("day_type"),
            "headline": o.get("headline"),
            "breakout_light": ({k: light.get(k) for k in ("color", "size_multiplier")} if light else None),
            "insights": [{"title": i.get("title"), "conviction": i.get("conviction"), "status": i.get("status"),
                          "instruments": [x.get("ticker") for x in i.get("instruments", [])]}
                         for i in o.get("insights", [])],
        })
    return out


# ── Payload + prompt ───────────────────────────────────────────────────────────
def build_payload(session: str | None, liquidity_path: Path | None,
                  prev_notes: int = 1, analogs: bool = True) -> dict:
    snaps = load_snapshots(session)
    snap = snaps[-1]
    session = snap["session"]
    history = load_regime_history(session)
    payload = {
        "session": session,
        "snapshots_available": len(snaps),
        "regime": regime_facts(history, snap),
        "trend_components": trend_component_facts(snaps),
        "matrix": matrix_facts(snaps),
        "cross_asset": cross_asset_facts(snap),
        "etf_table": etf_table_compact(snap),
    }
    liq = liquidity_facts(liquidity_path, session) if liquidity_path else None
    if liq:
        payload["liquidity_and_credit"] = liq
    prev = previous_insights(session, prev_notes)
    if prev:
        payload["previous_insights"] = prev
    if analogs:
        # Análogos históricos: módulo aparte (matrix_analogs.py); si no está, se omite.
        try:
            from matrix_analogs import analog_facts
            an = analog_facts(session)
            if an:
                payload["historical_analogs"] = an
        except ImportError:
            pass
        except Exception as e:
            log.warning("historical_analogs omitido: %s", e)
    return payload


ROLE_AND_MODEL = """You are a cross-asset macro strategist writing for one discretionary trader who runs an inter-market regime model ("Matrix") and trades US-listed ETFs and stocks, mostly breakouts, on top of it. He reads the dashboard every morning; your note sits next to it. He already sees every number — your job is to say what the numbers MEAN together, what is about to change, and what the model cannot see.

THE MODEL (all computed from daily closes, no forecasts):
- Trend Score (0-100%): 9 sub-scores worth 10 points. IVV above its MA100 (2 pts), MA100 rising vs 10 sessions ago (1), and seven ratios each worth 1 point when the ratio is above its own MA100: HYG/IEI, ISCB/ILCB (small vs large), IMCG/IMCV (growth vs value), RSPD/RSPS, RSPD/IDU, IVV/GLD, IVV/GSG.
- Regime (Bonds / Stocks): symmetric hysteresis. Bonds→Stocks needs 20 consecutive sessions with Trend Score ≥ 70%; Stocks→Bonds needs 20 consecutive sessions < 70%. The payload gives you the live countdown and the historical average regime length computed from the model's own history — use those, not a guess.
- Matrix: 17 pairs (16 ratios indexed to 100 plus IVV itself) compared with their MA100 inside a ±1.5% band. Above the band = Bullish (+1), below = Bearish (−1), inside = Neutral (0). Matrix score = sum. `band_position_pct` is only meaningful for Neutral pairs: 0 means sitting on the Bearish edge, 100 on the Bullish edge — a Neutral pair at ≤15 or ≥85 is one or two sessions from flipping. For Bullish/Bearish pairs the value is clamped and carries no information.
- ETF table: `cx` is an annualised exponential-regression slope over 91 sessions (momentum score; 1 = strongest of the universe), `rank` ranks it across ~128 ETFs, `climb30` is rank positions gained in 30 sessions, `net50/20/5` are net accumulation-minus-distribution days (≥1.5% moves on rising volume), `vs_ema20` / `vs_sma50` are % distance from those averages, `atr_ratio` is 5-day ATR over 50-day ATR (volatility expansion > 1), `mom_rank` is a second, EMA-ladder momentum rank.
- Everything in `regime`, `trend_components`, `matrix`, `cross_asset` is pre-computed by code from the point-in-time snapshots: streaks, last flips, edge flags, breadth by category, spreads between ETFs. Treat it as ground truth. `previous_insights` (if present) is a list of your own most recent notes, oldest first — mark what is new, what continues, what you are changing your mind on; if a thesis still holds, say "sigue en pie" and why in one sentence instead of rewriting it.
- `liquidity_and_credit` (if present) comes from a separate dashboard: FRED-based net liquidity, ICE BofA OAS spreads, and a 6-pillar breadth/volume/volatility score. If marked stale, say so and weigh it accordingly.
- `historical_analogs` (if present) is computed by code: the model's own history (Matrix flags, Trend sub-scores and the streak at/above or below 70%, rebuilt from prices back to 2007 plus the daily snapshots) is searched for the sessions closest to today's state; the last 20 sessions are excluded and look-alike consecutive days collapse into one episode dated at its first day. Each analog gives `distance` (lower = closer; one well above the others is a weak match), `pairs_compared` (< 17 means part of the Matrix did not exist yet), regime and `regime_day`, `pairs_differing_from_today`, and `fwd_10` / `fwd_20` with IVV / TLT / GLD returns, Trend and Matrix score change, whether and when the regime flipped, plus `ivv_worst_drawdown_20_pct`; `summary` aggregates them. These are BASE RATES from a handful of episodes, not predictions: say how often this configuration resolved one way or the other and over what horizon, quote dates and numbers, point out when the analogs disagree or sit in the other regime, never present them as a forecast.

WHAT "MACRO APPROACH" MEANS HERE: read the inter-market structure — credit vs duration, the yield-curve proxy (TLT vs SHY), inflation proxies (TIP vs IEF, metals, commodities), risk appetite (ARKK, QQQJ, micro caps, crypto trusts), cyclicals vs defensives, US vs world vs EM, gold vs everything — and relate it to where the regime is in its cycle. The most valuable things you can tell him are (1) transitions: which signals are flipping or about to flip and what that historically implies for the regime; (2) divergences: where the Matrix, the Trend components, breadth and price leadership disagree, and which side to trust; (3) timing: how far the regime is from a flip and what the countdown means for position sizing and for new breakouts; (4) blind spots: what this price-only model cannot see (policy, earnings, positioning, FX) and which readings are most exposed to that.

BREAKOUT LIGHT: `breakout_light` is the one output that feeds his position sizing, so get it right before anything else. green = 1x: the risk pairs (FFTY/IVV, ARKK/IVV, QQQJ/QQQ, Small vs Large) mostly Bullish or Neutral, credit confirming, breadth broadening, regime countdown not running. yellow = 0.5x: mixed or deteriorating, countdown running but early, leadership narrow. red = 0: countdown advanced, risk pairs Bearish, credit widening or breadth collapsing. `tilt_long` and `tilt_avoid` list the ETFs or groups with trend plus accumulation and the ones to stay away from; `rationale` names the signals that decided the color.

CONSTRAINTS:
- Use only the data in the payload unless a web search tool is available, in which case cite what you looked up and keep it to confirming or refuting the model's reading — never let news override the data.
- Every claim points at specific pairs, components, spreads or tickers with their numbers. Every actionable idea names instruments, a direction (long / short / avoid / watch), a horizon and what would invalidate it; if the data does not support a trade, say "no trade" explicitly — that is a valid, valuable answer.
- Distinguish what is already in price (long-standing Bullish/Bearish streaks) from what is changing (recent flips, edge flags, streak breaks). Flag contradictions between signals instead of smoothing them over.
- No generic macro commentary, no definitions, no restating the dashboard. Concrete and opinionated; a sentence that could be written on any day should not be written.
- Write in Spanish (rioplatense, voseo is fine), keep tickers, pair names and signal labels (Bullish / Neutral / Bearish, Stocks / Bonds) exactly as they appear in the data. Numbers in the text as in the data (percentages with one decimal at most)."""

WEEKLY_TASK = """THIS NOTE: the weekly deep read (Friday session). day_type = "weekly". Rank 3 to 6 insights by conviction, each tied to specific signals with its invalidation. Several watch_triggers: the signals closest to changing, with current level, the level that changes them and the implication. Several contradictions, each with which side you trust and why. blind_spots as a paragraph. changes_vs_previous against the week's daily notes: what they got right, what they missed, what you are changing. Dense prose is fine here."""

DAILY_TASK = """THIS NOTE: the daily desk note. Answer exactly these questions, in this order, and nothing else:
1. day_type: "transition" or "continuity". It is a continuity day when nothing flipped, no Neutral pair crossed an edge, no Trend component changed and the countdown means the same as yesterday. On a continuity day keep the whole note to the headline, a two-sentence regime_read, the breakout_light, one watch_trigger and changes_vs_previous; insights may be empty and contradictions may be empty.
2. regime_read: where we are in the cycle (day N against the historical average), sessions to the flip if the streak holds, and which ratios would have to come back (and how far they are) to reset the count.
3. breakout_light with its rationale and the tilts.
4. insights: at most 3, only what changed or is about to change, with status against previous_insights.
5. contradictions: at most one, the contradiction of the day, with which side you trust and why.
6. watch_triggers: exactly one, the single trigger for tomorrow: signal, current level, the level that changes it, and what to do if it crosses.
7. blind_spots: one sentence, the external variable most likely to invalidate today's read.
Short sentences. Nothing that could be written on any other day."""

# Perfiles: la diaria es corta y barata (Opus 5.5), la semanal es la lectura
# profunda (Fable 5.1). `prev_notes` = cuántas notas anteriores recibe.
PROFILES = {
    "daily":  {"label": "nota diaria",  "model": "claude-opus-5-5",  "effort": "medium",
               "max_tokens": 24000, "prev_notes": 1, "task": DAILY_TASK},
    "weekly": {"label": "nota semanal", "model": "claude-fable-5-1", "effort": "high",
               "max_tokens": 32000, "prev_notes": 5, "task": WEEKLY_TASK},
}


def resolve_profile(name: str, session: str) -> str:
    """'auto' → weekly si la sesión es viernes, daily el resto; un nombre explícito gana."""
    if name in PROFILES:
        return name
    return "weekly" if date.fromisoformat(session).weekday() == 4 else "daily"


def system_prompt(profile: str) -> str:
    return ROLE_AND_MODEL + "\n\n" + PROFILES[profile]["task"] + "\n"

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string", "description": "One sentence, the single most important thing today."},
        "day_type": {"type": "string", "enum": ["transition", "continuity", "weekly"]},
        "breakout_light": {
            "type": "object",
            "description": "Sizing filter for new breakouts today.",
            "properties": {
                "color": {"type": "string", "enum": ["green", "yellow", "red"]},
                "size_multiplier": {"type": "string", "enum": ["1x", "0.5x", "0"]},
                "tilt_long": {"type": "array", "items": {"type": "string"}, "description": "ETFs or groups with trend plus accumulation."},
                "tilt_avoid": {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string", "description": "The signals that decided the color, with numbers."},
            },
            "required": ["color", "size_multiplier", "tilt_long", "tilt_avoid", "rationale"],
            "additionalProperties": False,
        },
        "regime_read": {"type": "string", "description": "Where the regime is in its cycle, the flip countdown, what would reset it."},
        "insights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "thesis": {"type": "string", "description": "Paragraph with the argument, tied to specific signals and numbers."},
                    "evidence": {"type": "array", "items": {"type": "string"}, "description": "Short bullet facts from the payload (pair/component/ticker + number)."},
                    "instruments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "ticker": {"type": "string"},
                                "direction": {"type": "string", "enum": ["long", "short", "avoid", "watch"]},
                                "horizon": {"type": "string", "enum": ["days", "weeks", "months"]},
                                "note": {"type": "string"},
                            },
                            "required": ["ticker", "direction", "horizon", "note"],
                            "additionalProperties": False,
                        },
                    },
                    "conviction": {"type": "string", "enum": ["low", "medium", "high"]},
                    "invalidation": {"type": "string", "description": "What would prove this wrong, in terms of the model's own signals."},
                    "status": {"type": "string", "enum": ["new", "continuing", "changed"], "description": "Relative to previous_insights; 'new' when there is no previous note."},
                },
                "required": ["title", "thesis", "evidence", "instruments", "conviction", "invalidation", "status"],
                "additionalProperties": False,
            },
        },
        "watch_triggers": {
            "type": "array",
            "description": "Signals close to changing: regime countdown, Neutral pairs near an edge, components about to flip.",
            "items": {
                "type": "object",
                "properties": {
                    "signal": {"type": "string"},
                    "current": {"type": "string"},
                    "trigger": {"type": "string"},
                    "implication": {"type": "string"},
                },
                "required": ["signal", "current", "trigger", "implication"],
                "additionalProperties": False,
            },
        },
        "contradictions": {"type": "array", "items": {"type": "string"}, "description": "Where signals disagree and which side you trust, with why."},
        "blind_spots": {"type": "string", "description": "What this price-only model cannot see today and which readings are most exposed."},
        "changes_vs_previous": {"type": "string", "description": "What changed since the previous note; empty string if there was none."},
        "dashboard_one_liner": {"type": "string", "description": "≤ 200 characters for the dashboard header."},
    },
    "required": ["headline", "day_type", "breakout_light", "regime_read", "insights", "watch_triggers",
                 "contradictions", "blind_spots", "changes_vs_previous", "dashboard_one_liner"],
    "additionalProperties": False,
}


def user_message(payload: dict) -> str:
    return ("Session: " + payload["session"] + "\n\n"
            "Model facts and data (JSON):\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


# ── Llamada a Claude ───────────────────────────────────────────────────────────
def _usage_cost(model: str, usage) -> dict:
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cr = getattr(usage, "cache_read_input_tokens", 0) or 0
    p = PRICING.get(model)
    cost = None
    if p:
        cost = round(inp / 1e6 * p[0] + out / 1e6 * p[1] + cw / 1e6 * p[2] + cr / 1e6 * p[3], 4)
    return {"input_tokens": inp, "output_tokens": out, "cache_creation_input_tokens": cw,
            "cache_read_input_tokens": cr, "cost_usd": cost}


def call_claude(payload: dict, profile: str, model: str, effort: str, web: bool,
                max_tokens: int = 32000) -> tuple[dict, dict]:
    """Devuelve (output_json, meta). Streaming + structured output + fallbacks server-side."""
    import anthropic

    client = anthropic.Anthropic()
    system = [{"type": "text", "text": system_prompt(profile), "cache_control": {"type": "ephemeral"}}]
    messages = [{"role": "user", "content": user_message(payload)}]
    kwargs = dict(
        model=model, max_tokens=max_tokens, system=system, messages=messages,
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        # fallbacks="default" es el kwarg tipado en anthropic>=2.1; vía extra_body
        # llega igual al body y funciona también con el 0.84 instalado en el Mac.
        extra_body={"fallbacks": "default"},
    )
    if web:
        kwargs["tools"] = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}]

    log.info("Perfil %s → %s (effort=%s, web=%s)...", profile, model, effort, web)
    t0 = datetime.now()
    usage_total = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    final = None
    for _turn in range(4):                      # pause_turn solo aparece con server tools
        with client.beta.messages.stream(**kwargs) as stream:
            msg = stream.get_final_message()
        c = _usage_cost(model, msg.usage)
        for k in usage_total:
            usage_total[k] += c[k]
        if msg.stop_reason == "pause_turn":
            kwargs["messages"] = kwargs["messages"] + [{"role": "assistant", "content": msg.content}]
            continue
        final = msg
        break
    if final is None:
        raise RuntimeError("La API no terminó el turno (pause_turn repetido).")

    served_by = final.model
    fell_back = any(getattr(b, "type", "") == "fallback" for b in final.content) or served_by != model
    if final.stop_reason == "refusal":
        det = getattr(final, "stop_details", None)
        raise RuntimeError(f"La API rechazó el pedido (refusal, category={getattr(det, 'category', None)}); "
                           f"ningún modelo de la cadena contestó.")
    if final.stop_reason == "max_tokens":
        raise RuntimeError(f"Salida truncada por max_tokens={max_tokens}; subí --max-tokens.")

    texts = [b.text for b in final.content if getattr(b, "type", "") == "text"]
    if not texts:
        raise RuntimeError(f"Respuesta sin bloque de texto (stop_reason={final.stop_reason}).")
    data = json.loads(texts[-1])

    p = PRICING.get(model)
    cost = (round(usage_total["input_tokens"] / 1e6 * p[0] + usage_total["output_tokens"] / 1e6 * p[1]
                  + usage_total["cache_creation_input_tokens"] / 1e6 * p[2]
                  + usage_total["cache_read_input_tokens"] / 1e6 * p[3], 4) if p else None)
    meta = {
        "profile": profile, "label": PROFILES[profile]["label"],
        "model_requested": model, "model_served": served_by, "fell_back": fell_back,
        "effort": effort, "web_search": web, "stop_reason": final.stop_reason,
        "request_id": getattr(final, "_request_id", None),
        "elapsed_s": round((datetime.now() - t0).total_seconds(), 1),
        "usage": {**usage_total, "cost_usd": cost},
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "api",
    }
    log.info("Listo en %ss · %s in / %s out · cache w %s / r %s · $%s%s",
             meta["elapsed_s"], usage_total["input_tokens"], usage_total["output_tokens"],
             usage_total["cache_creation_input_tokens"], usage_total["cache_read_input_tokens"], cost,
             f" · FALLBACK → {served_by}" if fell_back else "")
    return data, meta


def estimate_tokens(payload: dict, profile: str, model: str) -> tuple[int, str]:
    """count_tokens si hay key; si no, estimación por caracteres."""
    sys_text = system_prompt(profile)
    text = sys_text + user_message(payload)
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            r = client.messages.count_tokens(model=model, system=sys_text,
                                             messages=[{"role": "user", "content": user_message(payload)}])
            return r.input_tokens, "count_tokens"
        except Exception as e:
            log.warning("count_tokens falló (%s); estimo por caracteres.", e)
    return int(len(text) / 3.6), "estimado (chars/3.6)"


# ── Render ─────────────────────────────────────────────────────────────────────
LIGHT_ES = {"green": "VERDE", "yellow": "AMARILLO", "red": "ROJO"}
DAY_ES = {"transition": "día de transición", "continuity": "día de continuidad", "weekly": "lectura semanal"}


def _meta_line(out: dict, meta: dict) -> str:
    cost = meta.get("usage", {}).get("cost_usd")
    parts = [meta.get("label") or meta.get("profile") or "",
             DAY_ES.get(out.get("day_type"), out.get("day_type") or ""),
             meta.get("model_served") or meta.get("model_requested") or "",
             f"effort {meta.get('effort')}",
             "web search" if meta.get("web_search") else "sin web",
             f"fuente: {meta.get('source', 'api')}"]
    if cost is not None:
        parts.append(f"${cost}")
    return " · ".join(p for p in parts if p)


def to_markdown(out: dict, meta: dict, session: str) -> str:
    L = [f"# Macro insights · {session}", "", f"_{_meta_line(out, meta)}_", "", f"**{out['headline']}**", ""]
    bl = out.get("breakout_light")
    if bl:
        line = f"**Semáforo breakouts: {LIGHT_ES.get(bl['color'], bl['color'])} · tamaño {bl['size_multiplier']}**"
        if bl.get("tilt_long"):
            line += f" · long: {', '.join(bl['tilt_long'])}"
        if bl.get("tilt_avoid"):
            line += f" · evitar: {', '.join(bl['tilt_avoid'])}"
        L += [line, "", bl["rationale"], ""]
    L += ["## Régimen", "", out["regime_read"], ""]
    if out["insights"]:
        L.append("## Insights")
    for i, ins in enumerate(out["insights"], 1):
        L += ["", f"### {i}. {ins['title']}  —  convicción {ins['conviction']} · {ins['status']}", "", ins["thesis"], ""]
        if ins["evidence"]:
            L += ["Evidencia:"] + [f"- {e}" for e in ins["evidence"]] + [""]
        if ins["instruments"]:
            L += ["Instrumentos:"] + [f"- **{x['ticker']}** · {x['direction']} · {x['horizon']} — {x['note']}" for x in ins["instruments"]] + [""]
        L.append(f"Invalidación: {ins['invalidation']}")
    if out["watch_triggers"]:
        title = "Disparador para mañana" if len(out["watch_triggers"]) == 1 else "Disparadores a vigilar"
        L += ["", f"## {title}", "", "| Señal | Hoy | Gatillo | Implicancia |", "|---|---|---|---|"]
        L += [f"| {w['signal']} | {w['current']} | {w['trigger']} | {w['implication']} |" for w in out["watch_triggers"]]
    if out["contradictions"]:
        title = "Contradicción del día" if len(out["contradictions"]) == 1 else "Contradicciones"
        L += ["", f"## {title}", ""] + [f"- {c}" for c in out["contradictions"]]
    L += ["", "## Puntos ciegos", "", out["blind_spots"]]
    if out.get("changes_vs_previous"):
        L += ["", "## Cambios vs nota anterior", "", out["changes_vs_previous"]]
    L += ["", "---", f"_{out['dashboard_one_liner']}_", ""]
    return "\n".join(L)


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _style_text(text: str) -> str:
    """Misma convención que el dashboard: tickers en strong, porcentajes en em."""
    t = _esc(text)
    t = re.sub(r"\b([A-Z]{2,5})\b(?!\w)", r"<strong>\1</strong>", t)
    t = re.sub(r"([+\-−]?\d+[.,]?\d*%)", r"<em>\1</em>", t)
    return t


def html_fragment(out: dict, meta: dict, session: str) -> str:
    """Bloque listo para caer dentro de <div class="ai-section"> del dashboard."""
    sub = f"{session} · {_meta_line(out, meta)}"
    conv = {"high": "mi-conv-high", "medium": "mi-conv-med", "low": "mi-conv-low"}
    dirc = {"long": "p-b", "short": "p-r", "avoid": "p-r", "watch": "p-n"}

    bl = out.get("breakout_light")
    light_pill = (f'<span class="mi-light mi-light-{bl["color"]}">breakouts {LIGHT_ES.get(bl["color"], bl["color"])} · {bl["size_multiplier"]}</span>'
                  if bl else "")
    light_block = ""
    if bl:
        tilts = "".join(f'<span class="pill p-b">{_esc(t)}</span>' for t in bl.get("tilt_long", []))
        tilts += "".join(f'<span class="pill p-r">evitar {_esc(t)}</span>' for t in bl.get("tilt_avoid", []))
        light_block = (f'<div class="ai-block"><div class="ai-block-title">Semáforo de breakouts</div>'
                       f'<p class="ai-text">{_style_text(bl["rationale"])}</p>'
                       f'{f"<div class=mi-tilt>{tilts}</div>" if tilts else ""}</div>')

    cards = []
    for ins in out["insights"]:
        instr = "".join(
            f'<span class="pill {dirc.get(x["direction"], "p-n")}" title="{_esc(x["note"])}">{_esc(x["ticker"])} · {x["direction"]} · {x["horizon"]}</span>'
            for x in ins["instruments"])
        ev = "".join(f"<li>{_style_text(e)}</li>" for e in ins["evidence"])
        cards.append(f'''<div class="mi-card">
  <div class="mi-card-head"><span class="mi-title">{_esc(ins["title"])}</span>
    <span class="mi-conv {conv.get(ins["conviction"], "")}">{ins["conviction"]}</span><span class="mi-status">{ins["status"]}</span></div>
  <p class="ai-text">{_style_text(ins["thesis"])}</p>
  {f'<ul class="mi-ev">{ev}</ul>' if ev else ''}
  {f'<div class="mi-instr">{instr}</div>' if instr else ''}
  <div class="mi-inval"><span>Invalidación</span> {_style_text(ins["invalidation"])}</div>
</div>''')

    triggers = "".join(
        f"<tr><td>{_esc(w['signal'])}</td><td>{_esc(w['current'])}</td><td>{_esc(w['trigger'])}</td><td>{_style_text(w['implication'])}</td></tr>"
        for w in out["watch_triggers"])
    contra = "".join(f"<li>{_style_text(c)}</li>" for c in out["contradictions"])
    changes = (f'<div class="ai-block"><div class="ai-block-title">Cambios vs nota anterior</div><p class="ai-text">{_style_text(out["changes_vs_previous"])}</p></div>'
               if out.get("changes_vs_previous") else "")
    day_tag = f'<span class="mi-daytype">{DAY_ES.get(out.get("day_type"), "")}</span>' if out.get("day_type") else ""
    insights_block = (f'<div class="ai-block"><div class="ai-block-title">Insights</div><div class="mi-grid">{"".join(cards)}</div></div>'
                      if cards else "")
    trig_title = "Disparador para mañana" if len(out["watch_triggers"]) == 1 else "Disparadores a vigilar"
    triggers_block = (f'<div class="ai-block"><div class="ai-block-title">{trig_title}</div>'
                      f'<table class="mi-table"><thead><tr><th>Señal</th><th>Hoy</th><th>Gatillo</th><th>Implicancia</th></tr></thead><tbody>{triggers}</tbody></table></div>'
                      if triggers else "")
    contra_title = "Contradicción del día" if len(out["contradictions"]) == 1 else "Contradicciones"
    contra_block = (f'<div class="ai-block"><div class="ai-block-title">{contra_title}</div><ul class="mi-ev">{contra}</ul></div>'
                    if contra else "")
    return f'''<div class="ai-header">
  <div class="ai-badge">Macro Insights</div>{light_pill}
  <div class="ai-model">{_esc(sub)}</div>{day_tag}
</div>
<p class="mi-headline">{_style_text(out["headline"])}</p>
{light_block}
<div class="ai-block"><div class="ai-block-title">Régimen</div><p class="ai-text">{_style_text(out["regime_read"])}</p></div>
{insights_block}
<div class="ai-body">
  <div class="ai-col">{triggers_block}</div>
  <div class="ai-col">{contra_block}
    <div class="ai-block"><div class="ai-block-title">Puntos ciegos</div><p class="ai-text">{_style_text(out["blind_spots"])}</p></div>{changes}</div>
</div>'''


FRAGMENT_CSS = """
.mi-headline{font-family:'IBM Plex Sans',sans-serif;font-size:17px;color:var(--white);line-height:1.6;margin:0 0 22px}
.mi-light{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;font-weight:700;padding:4px 10px;border-radius:4px;border:1px solid}
.mi-light-green{color:var(--teal);background:var(--teal2);border-color:rgba(0,229,204,.4)}
.mi-light-yellow{color:var(--amber);background:var(--amber2);border-color:rgba(245,158,11,.4)}
.mi-light-red{color:var(--red);background:var(--red2);border-color:rgba(255,77,109,.4)}
.mi-daytype{font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:var(--mist);margin-left:auto}
.mi-tilt{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.mi-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.mi-card{background:var(--bg3);border:1px solid var(--rule);border-radius:var(--r);padding:16px 18px}
.mi-card-head{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.mi-title{font-weight:700;color:var(--white);font-size:14px;flex:1}
.mi-conv{font-size:9px;letter-spacing:1.5px;text-transform:uppercase;padding:2px 8px;border-radius:3px;border:1px solid}
.mi-conv-high{color:var(--teal);border-color:rgba(0,229,204,.4);background:var(--teal2)}
.mi-conv-med{color:var(--amber);border-color:rgba(245,158,11,.4);background:var(--amber2)}
.mi-conv-low{color:var(--mist);border-color:var(--rule2)}
.mi-status{font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:var(--mist)}
.mi-ev{margin:8px 0 0;padding-left:18px;font-size:12.5px;color:var(--smoke);line-height:1.7}
.mi-instr{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.mi-inval{margin-top:10px;font-size:12px;color:var(--mist);line-height:1.6}
.mi-inval span{font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:var(--red);margin-right:6px}
.mi-table{width:100%;border-collapse:collapse;font-size:12.5px}
.mi-table th{text-align:left;font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:var(--mist);padding:6px 8px;border-bottom:1px solid var(--rule)}
.mi-table td{padding:8px;border-bottom:1px solid var(--rule);vertical-align:top;color:var(--smoke);line-height:1.5}
.p-r{background:var(--red2);color:var(--red);border:1px solid rgba(255,77,109,.25)}
@media(max-width:1100px){.mi-grid{grid-template-columns:1fr}}
"""

STANDALONE_CSS = """
:root{--bg:#0d1117;--bg2:#111827;--bg3:#1a2235;--rule:#1e2d45;--rule2:#243452;
--teal:#00e5cc;--teal2:rgba(0,229,204,.10);--teal3:rgba(0,229,204,.06);
--red:#ff4d6d;--red2:rgba(255,77,109,.10);--amber:#f59e0b;--amber2:rgba(245,158,11,.10);
--white:#e8ecf2;--smoke:#b7c0cf;--mist:#6b7a90;--r:10px;--font:'IBM Plex Sans',system-ui,sans-serif}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--smoke);font-family:var(--font);font-size:14px;line-height:1.65}
.wrap{max-width:1280px;margin:0 auto;padding:28px 24px}
.ai-section{background:var(--bg2);border:1px solid var(--rule);border-radius:var(--r);padding:32px 36px}
.ai-header{display:flex;align-items:center;gap:12px;margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid var(--rule)}
.ai-badge{background:var(--teal2);border:1px solid rgba(0,229,204,.3);border-radius:4px;padding:4px 10px;font-size:10px;letter-spacing:2px;color:var(--teal);font-weight:700;text-transform:uppercase}
.ai-model{font-size:10px;color:var(--mist);letter-spacing:1px}
.ai-body{display:grid;grid-template-columns:1fr 1fr;gap:40px}
.ai-block{margin-bottom:28px}.ai-block:last-child{margin-bottom:0}
.ai-block-title{font-size:10px;letter-spacing:2px;color:var(--teal);text-transform:uppercase;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--rule)}
.ai-text{font-size:14px;color:var(--smoke);line-height:1.9;letter-spacing:.1px;margin:0}
.ai-text strong{color:var(--white);font-weight:700}.ai-text em{color:var(--teal);font-style:normal}
.pill{display:inline-block;font-size:10px;letter-spacing:1px;padding:3px 9px;border-radius:4px;font-family:'JetBrains Mono',monospace}
.p-b{background:var(--teal2);color:var(--teal);border:1px solid rgba(0,229,204,.2)}
.p-n{background:var(--amber2);color:var(--amber);border:1px solid rgba(245,158,11,.2)}
@media(max-width:1100px){.ai-body{grid-template-columns:1fr;gap:24px}}
"""


def to_html(out: dict, meta: dict, session: str) -> str:
    return f'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Macro insights {session}</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>{STANDALONE_CSS}{FRAGMENT_CSS}</style></head>
<body><div class="wrap"><div class="ai-section">{html_fragment(out, meta, session)}</div></div></body></html>'''


def write_outputs(out: dict, meta: dict, session: str, suffix: str = "") -> list[Path]:
    INSIGHTS_DIR.mkdir(exist_ok=True)
    # No usar with_suffix(): con suffix=".preview" reemplazaría el sufijo en vez de agregarlo
    base = str(INSIGHTS_DIR / f"{session}{suffix}")
    paths = []
    (p := Path(base + ".json")).write_text(json.dumps({"session": session, "meta": meta, "output": out},
                                                      ensure_ascii=False, indent=1)); paths.append(p)
    (p := Path(base + ".md")).write_text(to_markdown(out, meta, session)); paths.append(p)
    (p := Path(base + ".html")).write_text(to_html(out, meta, session)); paths.append(p)
    return paths


# ── Entrada para generate_dashboard.py ─────────────────────────────────────────
def run_for_dashboard(session: str, profile_name: str = DEFAULT_PROFILE, force: bool = False,
                      liquidity_path: Path | None = LIQUIDITY_DATA_JS) -> tuple[dict, dict] | None:
    """Nota del día para el dashboard: reusa insights/<sesión>.json si ya vino
    de la API, si no llama (perfil auto: viernes weekly, resto daily) y escribe
    json/md/html. Devuelve (output, meta) o None si no se pudo generar."""
    existing = INSIGHTS_DIR / f"{session}.json"
    if existing.exists() and not force:
        try:
            d = json.loads(existing.read_text())
            if d.get("meta", {}).get("source") == "api":
                log.info("macro insights: reuso %s (perfil %s).", existing.name, d["meta"].get("profile"))
                return d["output"], d["meta"]
        except Exception as e:
            log.warning("macro insights: %s ilegible (%s), regenero.", existing.name, e)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.warning("macro insights: sin ANTHROPIC_API_KEY — se omite.")
        return None
    profile = resolve_profile(profile_name, session)
    prof = PROFILES[profile]
    payload = build_payload(session, liquidity_path, prev_notes=prof["prev_notes"])
    out, meta = call_claude(payload, profile, prof["model"], prof["effort"], False, prof["max_tokens"])
    write_outputs(out, meta, session)
    return out, meta


# ── CLI ────────────────────────────────────────────────────────────────────────
def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Insights macro sobre el modelo Matrix (Claude).")
    ap.add_argument("--session", help="sesión YYYY-MM-DD a analizar (default: último snapshot); point-in-time")
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=["auto", "daily", "weekly"],
                    help="daily: nota corta (Opus 5.5, medium) · weekly: lectura larga (Fable 5.1, high) · auto: viernes → weekly")
    ap.add_argument("--model", default=None, help="modelo (default: el del perfil)")
    ap.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"], help="default: el del perfil")
    ap.add_argument("--max-tokens", type=int, default=None, help="default: el del perfil")
    ap.add_argument("--web", action="store_true", help="habilita web search (máx 5 usos)")
    ap.add_argument("--no-liquidity", action="store_true", help="no leer el Liquidity Dashboard")
    ap.add_argument("--no-analogs", action="store_true", help="no incluir análogos históricos (matrix_analogs.py)")
    ap.add_argument("--liquidity-js", default=str(LIQUIDITY_DATA_JS), help="path a dashboard_data.js")
    ap.add_argument("--dry-run", action="store_true", help="imprime prompt + payload + tokens estimados y sale")
    ap.add_argument("--payload-out", help="con --dry-run: guarda el payload JSON en este path")
    ap.add_argument("--render", help="re-renderiza md/html desde un JSON de insights existente")
    ap.add_argument("--force", action="store_true", help="re-llama aunque ya exista insights/<sesión>.json")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    _load_dotenv()

    if args.render:
        d = json.loads(Path(args.render).read_text())
        session = d.get("session") or Path(args.render).stem[:10]
        paths = write_outputs(d["output"], d.get("meta", {}), session, suffix=Path(args.render).stem[10:])
        print("\n".join(str(p) for p in paths))
        return 0

    liq = None if args.no_liquidity else Path(args.liquidity_js)
    session = args.session or load_snapshots()[-1]["session"]
    profile = resolve_profile(args.profile, session)
    prof = PROFILES[profile]
    model = args.model or prof["model"]
    effort = args.effort or prof["effort"]
    max_tokens = args.max_tokens or prof["max_tokens"]
    payload = build_payload(session, liq, prev_notes=prof["prev_notes"], analogs=not args.no_analogs)
    session = payload["session"]

    if args.dry_run:
        n, how = estimate_tokens(payload, profile, model)
        p = PRICING.get(model, (None,) * 4)
        print("═" * 78 + f"\nPERFIL {profile} ({prof['label']}) · {model} · effort {effort} · max_tokens {max_tokens}\n"
              + "═" * 78 + "\nSYSTEM PROMPT\n" + "═" * 78 + "\n" + system_prompt(profile))
        print("═" * 78 + f"\nPAYLOAD ({session}) — claves: {list(payload)}\n" + "═" * 78)
        print(json.dumps({k: v for k, v in payload.items() if k not in ("etf_table", "historical_analogs")},
                         ensure_ascii=False, indent=1)[:12000])
        print(f"... (+ etf_table con {len(payload['etf_table'])} filas)")
        if "historical_analogs" in payload:
            print("═" * 78 + "\nHISTORICAL ANALOGS\n" + "═" * 78)
            print(json.dumps(payload["historical_analogs"], ensure_ascii=False, indent=1))
        print("═" * 78)
        print(f"Input ≈ {n:,} tokens ({how}). Con {model}: input ≈ ${n / 1e6 * p[0]:.3f}" if p[0] else f"Input ≈ {n:,} tokens ({how}).")
        print(f"Output típico 2-5k tokens visibles + thinking (effort {effort}) → "
              f"estimá ${(n / 1e6 * p[0]) + 0.003 * p[1]:.2f} – ${(n / 1e6 * p[0]) + 0.015 * p[1]:.2f} por corrida." if p[0] else "")
        if args.payload_out:
            Path(args.payload_out).write_text(json.dumps(payload, ensure_ascii=False, indent=1))
            print(f"payload guardado → {args.payload_out}")
        return 0

    existing = INSIGHTS_DIR / f"{session}.json"
    if existing.exists() and not args.force:
        d = json.loads(existing.read_text())
        if d.get("meta", {}).get("source") == "api":
            log.info("Ya existe %s (fuente api, perfil %s) — no re-llamo. Usá --force para regenerar.",
                     existing.name, d["meta"].get("profile"))
            print(existing)
            return 0

    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("No hay ANTHROPIC_API_KEY (entorno o .env). Probá --dry-run para ver el prompt y el payload.")
        return 2

    try:
        out, meta = call_claude(payload, profile, model, effort, args.web, max_tokens)
    except Exception as e:                       # mensajes útiles por tipo de error
        try:
            import anthropic
            if isinstance(e, anthropic.AuthenticationError):
                log.error("API key inválida."); return 2
            if isinstance(e, anthropic.RateLimitError):
                log.error("Rate limit — reintentá en un rato."); return 3
            if isinstance(e, anthropic.BadRequestError):
                log.error("400 de la API: %s", e.message); return 4
            if isinstance(e, anthropic.APIStatusError):
                log.error("API %s: %s", e.status_code, e.message); return 5
            if isinstance(e, anthropic.APIConnectionError):
                log.error("Sin conexión con la API."); return 6
        except ImportError:
            log.error("Falta el SDK: pip install -U 'anthropic>=2.1'"); return 7
        log.error("%s", e)
        return 1

    paths = write_outputs(out, meta, session)
    print("\n" + to_markdown(out, meta, session))
    print("\n".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
