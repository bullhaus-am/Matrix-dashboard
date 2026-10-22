"""
matrix_analogs.py
=================
Análogos históricos para macro_insights.py: base rates, no predicciones.

Busca en la historia del modelo los días cuyo estado (17 flags de la Matrix +
9 sub-scores del Trend + racha sobre / bajo el 70%) más se parece al de la
sesión analizada y mide qué pasó en las 10 y 20 sesiones siguientes (IVV, TLT,
GLD, Trend Score, Matrix score, flip de régimen). La historia sale de dos
fuentes:

  * los snapshots diarios (Matrix completa, desde 2026-06-10), y
  * una reconstrucción sesión a sesión sobre la historia máxima de
    .backtest_prices.pkl (cache de precios de backtest.py) con las funciones
    de producción compute_matrix / compute_trend_score, recortando las series
    por fecha como build_trend_history. Se cachea en .analogs_cache.json
    (gitignored), keyed por el último día de precios del pickle; si el pickle
    no está, se usan solo los snapshots.

Todo es determinístico y sin red. `analog_facts(session)` devuelve el dict que
build_payload() mete en el payload como `historical_analogs`.

Uso directo:
    python3 matrix_analogs.py                     # última sesión, JSON a stdout
    python3 matrix_analogs.py --session 2026-09-15
    python3 matrix_analogs.py --rebuild           # fuerza la reconstrucción
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from datetime import datetime
from pathlib import Path

from macro_insights import (HERE, MATRIX_TICKERS, TREND_COMPONENTS, TREND_THRESHOLD,
                            REGIME_CONFIRM_DAYS, load_snapshots)

log = logging.getLogger("matrix_analogs")


# ── Análogos históricos ────────────────────────────────────────────────────────
# Base rates, no predicciones: se buscan en la historia del modelo los días
# cuyo estado (17 flags de la Matrix + 9 sub-scores del Trend + racha sobre /
# bajo el 70%) más se parece al de hoy y se mide qué pasó en las 10 y 20
# sesiones siguientes. La historia sale de dos fuentes: los snapshots (Matrix
# completa, desde 2026-06-10) y una reconstrucción sesión a sesión sobre la
# historia máxima de .backtest_prices.pkl (cache de precios de backtest.py)
# con las mismas funciones de producción, cacheada en .analogs_cache.json.
PRICES_CACHE_FILE = HERE / ".backtest_prices.pkl"
ANALOGS_CACHE_FILE = HERE / ".analogs_cache.json"
ANALOGS_CACHE_VERSION = 1
ANALOG_K = 5
ANALOG_EXCLUDE_RECENT = 20   # las últimas 20 sesiones son el episodio actual, no análogos
ANALOG_EPISODE_GAP = 10      # dos candidatos a ≤ N sesiones son el mismo episodio
ANALOG_EPISODE_TOL = 1.0     # dentro del episodio se retrocede al primer día con dist ≤ mejor + TOL
ANALOG_W_FLAGS = 1.0         # peso de la Hamming sobre los 17 flags de la Matrix
ANALOG_W_TREND = 0.5         # peso de la L1 sobre sub-scores normalizados + racha
ANALOG_MIN_PAIRS = 11        # pares computables mínimos para que un día pasado sea candidato
ANALOG_HORIZONS = (10, 20)
ANALOG_FWD_TICKERS = ("IVV", "TLT", "GLD")
MATRIX_PAIRS = list(MATRIX_TICKERS)                  # 17 nombres, mismo orden que MATRIX_DEFS
TREND_MAX_POINTS = [pts for _, _, pts in TREND_COMPONENTS]
# Lo que compute_trend_score() necesita sí o sí (mismo orden que TREND_COMPONENTS)
TREND_TICKERS = ["IVV", "HYG", "IEI", "ISCB", "ILCB", "IMCG", "IMCV", "RSPD", "RSPS", "IDU", "GLD", "GSG"]
TREND_RATIOS = [("HYG", "IEI"), ("ISCB", "ILCB"), ("IMCG", "IMCV"), ("RSPD", "RSPS"),
                ("RSPD", "IDU"), ("IVV", "GLD"), ("IVV", "GSG")]


def load_price_closes(path: Path = PRICES_CACHE_FILE) -> dict:
    """{ticker: Series de closes} desde el pickle de backtest.py (sin red)."""
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    return {t: df["close"].dropna() for t, df in data.items() if "close" in getattr(df, "columns", [])}


def reconstruct_states(prices: dict, window: int | None = None, progress: bool = False) -> dict:
    """Matrix + Trend sesión a sesión sobre toda la historia de `prices`.

    Usa las funciones de producción (compute_matrix / compute_trend_score)
    recortando cada serie por fecha, como build_trend_history. A cada día se
    le pasan solo las últimas `window` barras: las dos funciones miran el
    último valor de la MA100 (y su shift 10) y el índice a 100 de la Matrix
    es invariante a escala, así que con ≥ TREND_MA + 10 barras el resultado
    es idéntico y la corrida es el doble de rápida.

    Arranca el primer día en que los 9 componentes del Trend son computables.
    Un par de la Matrix cuyo ratio todavía no tiene TREND_MA barras ese día
    lleva flag None (producción lo marcaría Neutral; acá no hay que
    confundirlo con un Neutral real). Tickers ausentes en `prices`: los del
    Trend son obligatorios (ValueError); los que solo usa la Matrix dejan su
    par en None y quedan listados en `missing_tickers`."""
    import pandas as pd
    import generate_dashboard as gd
    ma = gd.TREND_MA
    window = window or ma + 50
    needed = sorted({t for _, a, b, _ in gd.MATRIX_DEFS for t in (a, b) if t} | set(ANALOG_FWD_TICKERS))
    missing = [t for t in needed if t not in prices]
    missing_trend = [t for t in TREND_TICKERS if t not in prices]
    if missing_trend:
        raise ValueError(f"Faltan tickers del Trend Score en el cache de precios: {missing_trend}")
    # Ticker ausente → serie de NaN sobre el calendario de IVV: _aligned() la
    # deja vacía (inner join + dropna), compute_matrix lo lee como "sin
    # historia" → Neutral, y acá ese par pasa a None.
    avail = {t: prices[t] if t in prices else pd.Series(float("nan"), index=prices["IVV"].index) for t in needed}
    ivv = avail["IVV"]
    if len(ivv) < ma + 10:
        raise ValueError("IVV no tiene historia suficiente para el Trend Score")

    pair_from = {}                                   # primer día con TREND_MA barras alineadas
    for name, num, den, kind in gd.MATRIX_DEFS:
        if kind == "momentum":
            s = ivv
        else:
            s, _ = gd._aligned(avail, num, den)
        pair_from[name] = s.index[ma - 1] if len(s) >= ma else None
    starts = [ivv.index[ma + 10 - 1]]
    for num, den in TREND_RATIOS:
        s, _ = gd._aligned(avail, num, den)
        if len(s) < ma:
            raise ValueError(f"{num}/{den}: solo {len(s)} barras alineadas (< {ma})")
        starts.append(s.index[ma - 1])
    start = max(starts)

    states = []
    dates = ivv.index[ivv.index >= start]
    for i, cutoff in enumerate(dates):
        p = {t: s.loc[:cutoff].iloc[-window:] for t, s in avail.items()}
        score, sub = gd.compute_trend_score(p)
        m = {r["n"]: r["f"] for r in gd.compute_matrix(p)}
        flags = [m.get(n) if (pair_from.get(n) is not None and pair_from[n] <= cutoff) else None
                 for n in MATRIX_PAIRS]
        states.append({
            "date": cutoff.strftime("%Y-%m-%d"),
            "flags": flags, "sub": [int(x) for x in sub], "trend": float(score),
            "matrix_score": int(sum(f for f in flags if f is not None)),
            "close": {t: float(p[t].iloc[-1]) for t in ANALOG_FWD_TICKERS
                      if t in p and len(p[t]) and pd.notna(p[t].iloc[-1])},
        })
        if progress and (i + 1) % 1000 == 0:
            log.info("reconstrucción: %d/%d sesiones", i + 1, len(dates))
    return {"states": states, "missing_tickers": missing,
            "pair_available_from": {n: (d.strftime("%Y-%m-%d") if d is not None else None)
                                    for n, d in pair_from.items()}}


def _file_sig(path: Path) -> list:
    st = path.stat()
    return [st.st_size, int(st.st_mtime)]


def load_reconstruction(prices_path: Path = PRICES_CACHE_FILE, cache_path: Path = ANALOGS_CACHE_FILE,
                        force: bool = False) -> dict | None:
    """Reconstrucción cacheada en .analogs_cache.json, keyed por el último día
    de precios del pickle (más la versión del formato). Si el pickle no cambió
    (tamaño + mtime) ni se carga. Sin pickle → None (solo snapshots)."""
    if not prices_path.exists():
        # Sin pickle (p. ej. en GitHub Actions) vale el cache commiteado: la
        # reconstrucción termina donde terminan los precios del pickle y los
        # snapshots cubren el resto, así que no pierde actualidad.
        if cache_path.exists() and not force:
            try:
                cached = json.loads(cache_path.read_text())
                if cached.get("version") == ANALOGS_CACHE_VERSION and cached.get("states"):
                    log.info("Sin %s: uso la reconstrucción cacheada en %s (hasta %s).",
                             prices_path.name, cache_path.name, cached.get("prices_last_day"))
                    return cached
            except Exception as e:
                log.warning("cache de análogos ilegible (%s).", e)
        log.warning("No está %s: los análogos usan solo los snapshots (corré backtest.py para bajar la historia).",
                    prices_path.name)
        return None
    cached = None
    if cache_path.exists() and not force:
        try:
            cached = json.loads(cache_path.read_text())
        except Exception as e:
            log.warning("cache de análogos ilegible (%s); reconstruyo.", e)
        if cached and cached.get("version") == ANALOGS_CACHE_VERSION and cached.get("prices_file_sig") == _file_sig(prices_path):
            return cached
    prices = load_price_closes(prices_path)
    if "IVV" not in prices:
        log.warning("%s no trae IVV: los análogos usan solo los snapshots.", prices_path.name)
        return None
    last_day = prices["IVV"].index[-1].strftime("%Y-%m-%d")
    if cached and cached.get("version") == ANALOGS_CACHE_VERSION and cached.get("prices_last_day") == last_day:
        cached["prices_file_sig"] = _file_sig(prices_path)
        cache_path.write_text(json.dumps(cached, separators=(",", ":")))
        return cached
    log.info("Reconstruyendo Matrix + Trend sesión a sesión hasta %s (una sola vez; queda en %s)...",
             last_day, cache_path.name)
    t0 = datetime.now()
    rec = reconstruct_states(prices, progress=True)
    rec.update({"version": ANALOGS_CACHE_VERSION, "prices_last_day": last_day,
                "prices_file": prices_path.name, "prices_file_sig": _file_sig(prices_path),
                "built_at": datetime.now().isoformat(timespec="seconds")})
    cache_path.write_text(json.dumps(rec, separators=(",", ":")))
    log.info("Reconstrucción lista: %d sesiones desde %s, %.0fs.", len(rec["states"]),
             rec["states"][0]["date"] if rec["states"] else "-", (datetime.now() - t0).total_seconds())
    if rec["missing_tickers"]:
        log.warning("Tickers de la Matrix ausentes en %s (sus pares quedan sin flag): %s",
                    prices_path.name, ", ".join(rec["missing_tickers"]))
    return rec


def snapshot_state(snap: dict) -> dict:
    """Estado del día en el mismo formato que la reconstrucción."""
    flags = {m["n"]: m["f"] for m in snap["matrix"]}
    closes = {e["ticker"]: e["close"] for e in snap.get("etfs", []) if e["ticker"] in ANALOG_FWD_TICKERS}
    return {"date": snap["session"], "flags": [flags.get(n) for n in MATRIX_PAIRS],
            "sub": [int(x) for x in snap["trend_sub_scores"]], "trend": float(snap["trend_score"]),
            "matrix_score": int(snap["matrix_score"]), "close": closes}


def merge_states(reconstructed: list[dict], snapshots: list[dict]) -> list[dict]:
    """Une reconstrucción y snapshots por fecha (el snapshot manda donde hay
    ambos) y empalma los closes del tramo de snapshots a los del pickle con un
    factor fijo por ticker (último día en común), para que el cambio de fuente
    no meta un salto en los retornos. Closes faltantes se arrastran (ffill)."""
    by_date = {r["date"]: dict(r, source="reconstruction") for r in reconstructed}
    snap_by_date = {s["date"]: s for s in snapshots}
    factor = {}
    for t in ANALOG_FWD_TICKERS:
        common = [d for d in sorted(snap_by_date)
                  if d in by_date and by_date[d]["close"].get(t) and snap_by_date[d]["close"].get(t)]
        factor[t] = by_date[common[-1]]["close"][t] / snap_by_date[common[-1]]["close"][t] if common else 1.0
    for d, s in snap_by_date.items():
        row = dict(s, source="snapshot")
        if d in by_date:                       # ambas fuentes: señales del snapshot, closes del pickle
            row["close"] = by_date[d]["close"]
        else:
            row["close"] = {t: v * factor[t] for t, v in s["close"].items() if v}
        by_date[d] = row
    rows = [by_date[d] for d in sorted(by_date)]
    prev = {}
    for r in rows:
        c = dict(prev)
        c.update({t: v for t, v in r["close"].items() if v})
        r["close"] = c
        prev = c
    return rows


def regime_path(trend_scores: list[float], threshold: float = TREND_THRESHOLD,
                confirm: int = REGIME_CONFIRM_DAYS) -> list[tuple[str, int]]:
    """(régimen, día del régimen) por sesión, con la histéresis de producción
    (compute_regime: arranca en Bonds, flipea tras `confirm` sesiones seguidas
    del otro lado del umbral). El día cuenta sesiones consecutivas en el
    régimen vigente, como el `regime_days` del dashboard."""
    regime, above, below, day = "Bonds", 0, 0, 0
    out = []
    for s in trend_scores:
        if s >= threshold:
            above += 1
            below = 0
            if regime == "Bonds" and above >= confirm:
                regime, day = "Stocks", 0
        else:
            below += 1
            above = 0
            if regime == "Stocks" and below >= confirm:
                regime, day = "Bonds", 0
        day += 1
        out.append((regime, day))
    return out


def streak_features(trend_scores: list[float], threshold: float = TREND_THRESHOLD,
                    cap: int = REGIME_CONFIRM_DAYS) -> list[float]:
    """Racha sobre (+) / bajo (−) el umbral por sesión, normalizada a [−1, 1]
    con tope `cap` (20 sesiones = lo que tarda un flip)."""
    out, streak, side = [], 0, None
    for s in trend_scores:
        cur = s >= threshold
        streak = streak + 1 if cur == side else 1
        side = cur
        out.append(round((1 if cur else -1) * min(streak, cap) / cap, 4))
    return out


def distance_parts(a: dict, b: dict, min_pairs: int = ANALOG_MIN_PAIRS) -> dict | None:
    """Distancia entre dos estados {flags, sub, streak}, con sus componentes.

    Hamming sobre los flags de la Matrix (peso ANALOG_W_FLAGS), reescalada a
    los 17 pares cuando alguno no es computable en uno de los dos días, más
    L1 sobre los sub-scores normalizados a [0, 1] y la racha (peso
    ANALOG_W_TREND). None si hay menos de `min_pairs` pares comparables."""
    pairs = [(x, y) for x, y in zip(a["flags"], b["flags"]) if x is not None and y is not None]
    if len(pairs) < min_pairs:
        return None
    mism = sum(1 for x, y in pairs if x != y)
    ham = mism * len(MATRIX_PAIRS) / len(pairs)
    l1 = sum(abs(x - y) / m for x, y, m in zip(a["sub"], b["sub"], TREND_MAX_POINTS))
    l1 += abs(a["streak"] - b["streak"])
    return {"distance": round(ANALOG_W_FLAGS * ham + ANALOG_W_TREND * l1, 3),
            "matrix_mismatches": mism, "pairs_compared": len(pairs), "trend_l1": round(l1, 3)}


def analog_distance(a: dict, b: dict, min_pairs: int = ANALOG_MIN_PAIRS) -> float | None:
    p = distance_parts(a, b, min_pairs)
    return p["distance"] if p else None


def select_analogs(dists: list, k: int = ANALOG_K, exclude_recent: int = ANALOG_EXCLUDE_RECENT,
                   gap: int = ANALOG_EPISODE_GAP, tol: float = ANALOG_EPISODE_TOL) -> list[dict]:
    """Los k análogos más cercanos. `dists[i]` es la distancia de la sesión i
    a la actual (la última de la lista); None = no comparable.

    - Quedan fuera la sesión actual y las `exclude_recent` anteriores: son el
      mismo episodio, no análogos.
    - Dos candidatos a ≤ `gap` sesiones son el mismo episodio y queda el más
      cercano; el episodio es la racha de días consecutivos alrededor suyo
      con distancia ≤ mejor + `tol`, y el representante es su primer día (el
      día en que apareció la configuración): los retornos se miden desde ahí."""
    n = len(dists)
    limit = max(0, n - 1 - exclude_recent)
    cands = sorted((d, i) for i, d in enumerate(dists[:limit]) if d is not None)
    chosen = []
    for d, i in cands:
        first, last = i, i
        while first > 0 and dists[first - 1] is not None and dists[first - 1] <= d + tol:
            first -= 1
        while last + 1 < limit and dists[last + 1] is not None and dists[last + 1] <= d + tol:
            last += 1
        if any(last >= c["pos"] - gap and first <= c["last_pos"] + gap for c in chosen):
            continue
        chosen.append({"pos": first, "best_pos": i, "last_pos": last, "best_distance": d,
                       "episode_sessions": last - first + 1})
        if len(chosen) >= k:
            break
    return chosen


def forward_stats(rows: list[dict], regimes: list, pos: int, horizons=ANALOG_HORIZONS) -> dict:
    """Qué pasó en las `horizons` sesiones siguientes a rows[pos]: retornos de
    IVV/TLT/GLD, cambio del Trend Score y del Matrix score, flip de régimen y
    cuántas sesiones después; más el peor cierre de IVV dentro de la ventana
    larga (max adverse excursion desde el día del análogo)."""
    n = len(rows)
    r0 = rows[pos]
    out = {}
    for h in horizons:
        if pos + h >= n:
            out[f"fwd_{h}"] = None
            continue
        r1 = rows[pos + h]
        flip = next((j for j in range(1, h + 1) if regimes[pos + j][0] != regimes[pos][0]), None)
        entry = {}
        for t in ANALOG_FWD_TICKERS:
            c0, c1 = r0["close"].get(t), r1["close"].get(t)
            entry[f"{t.lower()}_pct"] = round((c1 / c0 - 1) * 100, 2) if c0 and c1 else None
        entry["trend_change_pts"] = int(round((r1["trend"] - r0["trend"]) * 100))
        entry["matrix_score_change"] = r1["matrix_score"] - r0["matrix_score"]
        entry["regime_flipped"] = flip is not None
        entry["sessions_to_flip"] = flip
        entry["regime_after"] = regimes[pos + h][0]
        out[f"fwd_{h}"] = entry
    hmax = max(horizons)
    key = f"ivv_worst_drawdown_{hmax}_pct"
    out[key] = None
    if pos + hmax < n and r0["close"].get("IVV"):
        path = [rows[pos + j]["close"].get("IVV") for j in range(1, hmax + 1)]
        path = [c for c in path if c]
        if path:
            out[key] = round(min(0.0, (min(path) / r0["close"]["IVV"] - 1) * 100), 2)
    return out


def summarize_analogs(analogs: list[dict], today_regime: str | None = None) -> dict:
    """Base rates agregadas sobre los análogos elegidos."""
    h = max(ANALOG_HORIZONS)

    def med(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.median(vals), 2) if vals else None

    with_h = [a for a in analogs if a.get(f"fwd_{h}")]
    out = {"n_analogs": len(analogs), f"n_with_{h}_sessions_forward": len(with_h)}
    if today_regime is not None:
        out["analogs_in_same_regime_as_today"] = sum(1 for a in analogs if a.get("regime") == today_regime)
    if not with_h:
        return out
    flips = [a[f"fwd_{h}"]["regime_flipped"] for a in with_h]
    out[f"pct_with_regime_flip_within_{h}"] = round(100 * sum(flips) / len(flips))
    out["median_sessions_to_flip_when_flipped"] = med(
        [a[f"fwd_{h}"]["sessions_to_flip"] for a in with_h if a[f"fwd_{h}"]["regime_flipped"]])
    for hh in ANALOG_HORIZONS:
        rows_hh = [a[f"fwd_{hh}"] for a in analogs if a.get(f"fwd_{hh}")]
        for t in ANALOG_FWD_TICKERS:
            out[f"median_{t.lower()}_{hh}_pct"] = med([r[f"{t.lower()}_pct"] for r in rows_hh])
    ivv_h = [r[f"fwd_{h}"]["ivv_pct"] for r in with_h if r[f"fwd_{h}"]["ivv_pct"] is not None]
    out[f"pct_ivv_positive_{h}"] = round(100 * sum(1 for v in ivv_h if v > 0) / len(ivv_h)) if ivv_h else None
    out[f"ivv_{h}_pct_range"] = [min(ivv_h), max(ivv_h)] if ivv_h else None
    dd = [a.get(f"ivv_worst_drawdown_{h}_pct") for a in with_h]
    dd = [v for v in dd if v is not None]
    out[f"worst_ivv_drawdown_{h}_pct"] = min(dd) if dd else None
    out[f"median_trend_change_{h}_pts"] = med([a[f"fwd_{h}"]["trend_change_pts"] for a in with_h])
    out[f"median_matrix_score_change_{h}"] = med([a[f"fwd_{h}"]["matrix_score_change"] for a in with_h])
    return out


def analog_facts(session: str | None = None, snaps: list[dict] | None = None, k: int = ANALOG_K,
                 prices_path: Path = PRICES_CACHE_FILE, cache_path: Path = ANALOGS_CACHE_FILE) -> dict | None:
    """Los k días pasados más parecidos a `session` y qué pasó después.

    Point-in-time: solo entran sesiones ≤ `session`, las 20 previas quedan
    excluidas y las ventanas forward de los análogos terminan antes de la
    sesión, así que nada de lo que se reporta era desconocido ese día.
    Devuelve None si no hay ningún análogo (historia demasiado corta)."""
    snaps = [s for s in (snaps or load_snapshots(session)) if not session or s["session"] <= session]
    session = snaps[-1]["session"]
    snap = snaps[-1]
    recon = load_reconstruction(prices_path, cache_path)
    rec_states = [r for r in (recon or {}).get("states", []) if r["date"] <= session]
    rows = merge_states(rec_states, [snapshot_state(s) for s in snaps])
    trend = [r["trend"] for r in rows]
    regimes = regime_path(trend)
    streaks = streak_features(trend)
    feats = [{"flags": r["flags"], "sub": r["sub"], "streak": st} for r, st in zip(rows, streaks)]
    today = feats[-1]
    parts = [distance_parts(today, f) for f in feats]
    dists = [p["distance"] if p else None for p in parts]
    chosen = select_analogs(dists, k=k)
    if not chosen:
        log.warning("historical_analogs: sin análogos para %s (%d sesiones de historia, hacen falta > %d).",
                    session, len(rows), ANALOG_EXCLUDE_RECENT + 1)
        return None

    analogs = []
    for c in chosen:
        pos, r, p = c["pos"], rows[c["pos"]], parts[c["pos"]]
        differing = [MATRIX_PAIRS[i] for i, (x, y) in enumerate(zip(today["flags"], r["flags"]))
                     if x is not None and y is not None and x != y]
        a = {
            "date": r["date"], "distance": p["distance"],
            "matrix_mismatches": p["matrix_mismatches"], "pairs_compared": p["pairs_compared"],
            "trend_l1": p["trend_l1"],
            "pairs_differing_from_today": differing,
            "episode_sessions": c["episode_sessions"], "episode_best_match_day": rows[c["best_pos"]]["date"],
            "episode_best_distance": c["best_distance"],
            "regime": regimes[pos][0], "regime_day": regimes[pos][1],
            "trend_score_pct": int(round(r["trend"] * 100)), "matrix_score": r["matrix_score"],
            "streak_sessions": int(round(abs(streaks[pos]) * REGIME_CONFIRM_DAYS)),
            "streak_side": "at_or_above_threshold" if streaks[pos] > 0 else "below_threshold",
            "source": r.get("source", "snapshot"),
        }
        a.update(forward_stats(rows, regimes, pos))
        analogs.append(a)

    n_cand = sum(1 for d in dists[:max(0, len(dists) - 1 - ANALOG_EXCLUDE_RECENT)] if d is not None)
    history = {
        "sessions": len(rows), "from": rows[0]["date"], "to": rows[-1]["date"],
        "reconstructed_sessions": len(rec_states),
        "snapshot_sessions": sum(1 for r in rows if r.get("source") == "snapshot"),
        "candidate_sessions": n_cand,
        "source": ("reconstruction from the price cache up to %s (ISCB/ILCB real, not the IJR/IVV proxy) + "
                   "daily snapshots after that; snapshot-era returns are price returns from each day's close"
                   % recon["prices_last_day"]) if recon else "daily snapshots only (no price cache on disk)",
    }
    if recon:
        history["pairs_not_computable_before"] = {
            n: d for n, d in recon["pair_available_from"].items() if d is None or d > rows[0]["date"]}
        history["matrix_tickers_missing_from_price_cache"] = recon["missing_tickers"]
    return {
        "what_this_is": ("Base rates, not a forecast: the %d past sessions whose model state looked most like today's, "
                         "and what the model and IVV/TLT/GLD did over the next %s sessions." % (k, "/".join(map(str, ANALOG_HORIZONS)))),
        "method": {
            "state": "17 Matrix flags (+1/0/-1), 9 Trend sub-scores normalised to [0,1], and the streak at/above (+) or below (-) 70% capped at 20 sessions and scaled to [-1,1]",
            "distance": f"{ANALOG_W_FLAGS} x Hamming over the Matrix flags (rescaled to 17 when a pair is not computable on one of the days) + {ANALOG_W_TREND} x L1 over the sub-scores and the streak; 0 = identical state, max ≈ {ANALOG_W_FLAGS * len(MATRIX_PAIRS) + ANALOG_W_TREND * (len(TREND_MAX_POINTS) + 2):g}",
            "episodes": f"the last {ANALOG_EXCLUDE_RECENT} sessions are excluded; candidates within {ANALOG_EPISODE_GAP} sessions of each other count as one episode (the run of consecutive days within {ANALOG_EPISODE_TOL} distance of the best match), dated and measured from its first day; `distance` is that first day's, `episode_best_distance` the episode's closest day",
            "regime_for_past_days": "simulated with the production hysteresis over the full Trend Score history (regime_history.json only starts in 2024)",
            "ivv_worst_drawdown": "lowest close vs the analog day's close within the next 20 sessions",
        },
        "history": history,
        "today": {
            "session": session, "regime": snap["regime"], "regime_day": snap["regime_days"],
            "trend_score_pct": int(round(snap["trend_score"] * 100)), "matrix_score": snap["matrix_score"],
            "streak_sessions": int(round(abs(streaks[-1]) * REGIME_CONFIRM_DAYS)),
            "streak_side": "at_or_above_threshold" if streaks[-1] > 0 else "below_threshold",
            "simulated_regime_agrees_with_dashboard": regimes[-1][0] == snap["regime"],
        },
        "analogs": analogs,
        "summary": summarize_analogs(analogs, snap["regime"]),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description="Análogos históricos del modelo Matrix (base rates).")
    ap.add_argument("--session", help="sesión YYYY-MM-DD (default: último snapshot); point-in-time")
    ap.add_argument("--k", type=int, default=ANALOG_K, help=f"cantidad de análogos (default {ANALOG_K})")
    ap.add_argument("--rebuild", action="store_true", help="rehace la reconstrucción aunque el cache esté vigente")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    if args.rebuild:
        load_reconstruction(force=True)
    out = analog_facts(args.session, k=args.k)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
