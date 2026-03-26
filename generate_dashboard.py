"""
generate_dashboard.py
=====================
Generates the Macro ETF Monitor dashboard HTML from Yahoo Finance data.

Usage:
    python generate_dashboard.py

Output:
    macro_etf_dashboard.html
"""

import time, json
from datetime import datetime
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np

# ── Parameters ─────────────────────────────────────────────────────────────────
ACC_THRESHOLD  =  1.5   # % up → accumulation day
DIST_THRESHOLD = -1.5   # % down → distribution day
VOL_MULT       =  1.10  # volume > prev_day × this
VOL_MA_LEN     =  50    # volume SMA period
SLOPE_WINDOW   =  91    # bars for exponential regression
MATRIX_BAND    =  0.015 # ±1.5% of MA100 for Neutral zone
TREND_MA       =  100   # MA period for trend/matrix signals
OUTPUT_FILE    = "macro_etf_dashboard.html"

# ── ETF Universe ───────────────────────────────────────────────────────────────
ETF_UNIVERSE = [
    # Country / Region
    ("IVV","S&P 500","Country/Region"),("EEM","Emerging Markets","Country/Region"),
    ("ILF","Latin America","Country/Region"),("EEMA","EM Asia","Country/Region"),
    ("ECNS","China Small Cap","Country/Region"),("ASEA","Southeast Asia","Country/Region"),
    ("ARGT","Argentina","Country/Region"),("EWA","Australia","Country/Region"),
    ("EWO","Austria","Country/Region"),("EWK","Belgium","Country/Region"),
    ("EWZ","Brazil","Country/Region"),("EWC","Canada","Country/Region"),
    ("ECH","Chile","Country/Region"),("MCHI","China Mid Cap","Country/Region"),
    ("EFNL","Finland","Country/Region"),("EWQ","France","Country/Region"),
    ("EWG","Germany","Country/Region"),("GREK","Greece","Country/Region"),
    ("EWH","Hong Kong","Country/Region"),("INDA","India","Country/Region"),
    ("EIDO","Indonesia","Country/Region"),("EIRL","Ireland","Country/Region"),
    ("EIS","Israel","Country/Region"),("EWI","Italy","Country/Region"),
    ("EWJ","Japan","Country/Region"),("EWM","Malaysia","Country/Region"),
    ("EWW","Mexico","Country/Region"),("EWN","Netherlands","Country/Region"),
    ("ENZL","New Zealand","Country/Region"),("ENOR","Norway","Country/Region"),
    ("EPU","Peru","Country/Region"),("EPHE","Philippines","Country/Region"),
    ("EPOL","Poland","Country/Region"),("EWS","Singapore","Country/Region"),
    ("EZA","South Africa","Country/Region"),("EWY","South Korea","Country/Region"),
    ("EWP","Spain","Country/Region"),("EWD","Sweden","Country/Region"),
    ("EWL","Switzerland","Country/Region"),("EWT","Taiwan","Country/Region"),
    ("THD","Thailand","Country/Region"),("TUR","Turkey","Country/Region"),
    ("EWU","United Kingdom","Country/Region"),("URTH","World","Country/Region"),
    # Industry / Sector
    ("IBB","Biotechnology","Industry/Sector"),("BOTZ","Robotics & AI","Industry/Sector"),
    ("ICLN","Clean Energy","Industry/Sector"),("IDGT","Digital Infrastructure","Industry/Sector"),
    ("IDU","US Utilities","Industry/Sector"),("ESPO","Video Gaming","Industry/Sector"),
    ("IAI","Brokers-Dealers","Industry/Sector"),("IAK","Insurance","Industry/Sector"),
    ("IAT","Regional Banks","Industry/Sector"),("IEO","Oil & Gas Prod.","Industry/Sector"),
    ("IEZ","US Oil Equipment","Industry/Sector"),("IGE","Natural Resources","Industry/Sector"),
    ("IGV","Tech Software","Industry/Sector"),("IHE","Pharmaceuticals","Industry/Sector"),
    ("IHF","Healthcare Providers","Industry/Sector"),("IHI","Medical Devices","Industry/Sector"),
    ("ITA","Aerospace","Industry/Sector"),("ITB","Home Construction","Industry/Sector"),
    ("IYF","Financial","Industry/Sector"),("IYG","Financial Services","Industry/Sector"),
    ("IYH","Healthcare","Industry/Sector"),("IYJ","Industrials","Industry/Sector"),
    ("IYM","Basic Materials","Industry/Sector"),("IYT","Transportation","Industry/Sector"),
    ("IYW","Technology","Industry/Sector"),("IYE","US Energy","Industry/Sector"),
    ("IYZ","Telecom","Industry/Sector"),("REM","Mortgage REIT","Industry/Sector"),
    ("RSPD","Eq. Wt. Discr.","Industry/Sector"),("RSPS","Eq. Wt. Staples","Industry/Sector"),
    ("SNSR","Internet of Things","Industry/Sector"),("SOXX","Semiconductor","Industry/Sector"),
    ("TAN","Solar","Industry/Sector"),("VEGI","Agriculture","Industry/Sector"),
    ("FFTY","IBD 50 ETF","Industry/Sector"),
    # Style
    ("ARKK","ARK Innovation","Style"),("EUSA","Equally Weighted","Style"),
    ("ILCB","Large-cap Blend","Style"),("ILCG","Large-cap Growth","Style"),
    ("ILCV","Large-cap Value","Style"),("IMCB","Mid-cap Blend","Style"),
    ("IMCG","Mid-cap Growth","Style"),("IMCV","Mid-cap Value","Style"),
    ("ISCB","Small-cap Blend","Style"),("ISCG","Small-cap Growth","Style"),
    ("ISCV","Small-cap Value","Style"),("IWC","Micro Cap","Style"),
    ("MTUM","Momentum","Style"),("QQQ","Nasdaq 100","Style"),
    ("QQQJ","NASDAQ Next Gen","Style"),("SPLV","Defensive Equity","Style"),
    # Fixed Income
    ("EMB","EM Govt (7)","Fixed Income"),("EMHY","EM HY Bonds (5)","Fixed Income"),
    ("GHYG","US & Intl HY Corp","Fixed Income"),("HYG","US HY Corp (4)","Fixed Income"),
    ("HYGH","HY Hedged (0)","Fixed Income"),("IEF","7-10Y Treasury (7.5)","Fixed Income"),
    ("IEI","3-7Y Treasury (4.5)","Fixed Income"),("LQDH","IG Corp Hedged (0.3)","Fixed Income"),
    ("MBB","MBS (2.5)","Fixed Income"),("SHV","Short Treasury (0.4)","Fixed Income"),
    ("SHY","1-3Y Treasury (2)","Fixed Income"),("SHYG","0-5Y HY Corp (2.3)","Fixed Income"),
    ("SLQD","0-5Y IG Corp (2.2)","Fixed Income"),("TIP","TIPS (7.5)","Fixed Income"),
    ("TLH","10-20Y Treasury (11)","Fixed Income"),("TLT","20Y+ Treasury (17.5)","Fixed Income"),
    # Commodity / REIT
    ("CPER","Copper","Commodity/REIT"),("ETHE","Ethereum Trust","Commodity/REIT"),
    ("GBTC","Bitcoin Trust","Commodity/REIT"),("GDX","Gold Miners","Commodity/REIT"),
    ("GLD","Gold","Commodity/REIT"),("GSG","Commodities","Commodity/REIT"),
    ("SLV","Silver","Commodity/REIT"),("VNQ","US Real Estate","Commodity/REIT"),
    ("WOOD","Timber","Commodity/REIT"),
]

# Deduplicate
seen = set()
ETFS = [(t,n,c) for t,n,c in ETF_UNIVERSE if not (t in seen or seen.add(t))]

CAT_ORDER = ["Country/Region","Industry/Sector","Style","Fixed Income","Commodity/REIT"]
CAT_DISPLAY = {
    "Country/Region":"Country / Region","Industry/Sector":"Industry / Sector",
    "Style":"Style","Fixed Income":"Fixed Income","Commodity/REIT":"Commodity / REIT",
}

# ── Extra tickers needed for Trend Score and Matrix ────────────────────────────
EXTRA_TICKERS = ['HYG','IEI','JKJ','JKD','JKH','JKI','IYC','IYK','IDU','IAU',
                 'ARKK','TLT','SLV','GLD','CPER','WOOD','EEM','ARGT','QQQJ','QQQ',
                 'FFTY','IVV','GSG','MTUM']

# ── Data fetching ──────────────────────────────────────────────────────────────
def fetch(ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    if raw.empty: return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    cols = [c for c in ["open","high","low","close","volume"] if c in raw.columns]
    return raw[cols].dropna()

# ── Metrics ────────────────────────────────────────────────────────────────────
def adj_slope_cx(close: pd.Series, offset: int = 0) -> float:
    vals = close.values[::-1]
    c0, c1 = offset, offset + SLOPE_WINDOW
    if c1 > len(vals): return np.nan
    log_p = np.log(vals[c0:c1])
    t = np.arange(len(log_p), dtype=float)[::-1]
    valid = ~np.isnan(log_p)
    lp, tv = log_p[valid], t[valid]
    if len(lp) < 30: return np.nan
    tm, lm = tv.mean(), lp.mean()
    stt = ((tv-tm)**2).sum(); sll = ((lp-lm)**2).sum(); stl = ((tv-tm)*(lp-lm)).sum()
    if stt == 0 or sll == 0: return np.nan
    return ((np.exp(stl/stt)**252)-1) * (stl**2/(stt*sll))

def compute_acc_dist(df: pd.DataFrame) -> dict:
    c = df["close"]; v = df["volume"]
    vol_sma = v.rolling(VOL_MA_LEN).mean()
    pchg = c.pct_change() * 100
    vi = v > v.shift(1) * VOL_MULT
    va = v > vol_sma
    is_acc  = (pchg >= ACC_THRESHOLD)  & vi & va
    is_dist = (pchg <= DIST_THRESHOLD) & vi & va
    acc = is_acc.astype(int); dist = is_dist.astype(int)
    result = {}
    for w in (50, 20, 5):
        a = int(acc.rolling(w,  min_periods=1).sum().iloc[-1])
        d = int(dist.rolling(w, min_periods=1).sum().iloc[-1])
        result[f"acc_{w}"] = a; result[f"dist_{w}"] = d; result[f"net_{w}"] = a - d
    return result

def perf(close: pd.Series, days: int) -> float:
    if len(close) <= days: return 0.0
    return round((close.iloc[-1] / close.iloc[-1-days] - 1) * 100, 2)

def analyze(ticker, name, cat, df) -> dict | None:
    if len(df) < 300: return None
    cx0  = adj_slope_cx(df["close"], 0)
    cx30 = adj_slope_cx(df["close"], 30)
    if np.isnan(cx0): return None
    ad = compute_acc_dist(df)
    c = df["close"]
    price   = c.iloc[-1]
    ema20   = c.ewm(span=20, adjust=False).mean().iloc[-1]
    sma50   = c.rolling(50).mean().iloc[-1]
    vs_ema20 = round((price / ema20 - 1) * 100, 2)
    vs_sma50 = round((price / sma50 - 1) * 100, 2)
    return {
        "ticker": ticker, "name": name, "category": cat,
        "close":  round(float(price), 2),
        "cx": round(cx0, 4),
        "cx30": round(cx30, 4) if not np.isnan(cx30) else 0.0,
        "yearly":  perf(c, 252),
        "perf_3m": perf(c, 63),
        "perf_1m": perf(c, 21),
        "vs_ema20": vs_ema20,
        "vs_sma50": vs_sma50,
        **ad,
    }

# ── Trend Score ────────────────────────────────────────────────────────────────
def compute_trend_score(prices: dict) -> tuple[float, list]:
    """9 sub-scores / 10. Returns (score, [sub_scores])."""
    ivv = prices['IVV']
    ma100 = ivv.rolling(TREND_MA).mean()
    delta = ma100 - ma100.shift(12)

    def ratio_idx(num, den):
        r = prices[num] / prices[den]
        return (r / r.iloc[0]) * 100

    def above_ma(series):
        ma = series.rolling(TREND_MA).mean()
        return 1 if series.iloc[-1] > ma.iloc[-1] else 0

    scores = [
        2 if ivv.iloc[-1] > ma100.iloc[-1] else 0,                    # IVV vs MA100
        1 if delta.iloc[-1] > 0 else 0,                                # IVV momentum
        above_ma(ratio_idx('HYG',  'IEI')),                            # HY vs Treasuries
        above_ma(ratio_idx('JKJ',  'JKD')),                            # Small vs Large
        above_ma(ratio_idx('JKH',  'JKI')),                            # Growth vs Value
        above_ma(ratio_idx('IYC',  'IYK')),                            # Discr vs Staples
        above_ma(ratio_idx('IYC',  'IDU')),                            # Services vs Utils
        above_ma(ratio_idx('IVV',  'IAU')),                            # Stocks vs Gold
        above_ma(ratio_idx('IVV',  'GSG')),                            # Stocks vs Commodities
    ]
    return round(sum(scores) / 10, 1), scores

# ── Regime ─────────────────────────────────────────────────────────────────────
def compute_regime(trend_series: list[float], threshold: float = 0.7) -> tuple[str, int]:
    """
    Hysteresis regime detection over last 20 observations.
    Bonds → Stocks: all 20 days >= threshold
    Stocks → Bonds: any day in last 20 < threshold
    Returns (regime, days_in_current_regime)
    """
    if len(trend_series) < 2:
        return "Bonds", 1

    regime = "Bonds"  # start assumption
    for score in reversed(trend_series):
        if score >= threshold:
            regime = "Stocks"
        else:
            regime = "Bonds"
            break

    # Count days in current regime
    days = 0
    current = trend_series[-1] >= threshold
    for s in reversed(trend_series):
        if (s >= threshold) == current:
            days += 1
        else:
            break
    return ("Stocks" if current else "Bonds"), days

# ── Matrix ─────────────────────────────────────────────────────────────────────
MATRIX_DEFS = [
    ("S&P Momentum",              'IVV',  None,    'momentum'),
    ("High Yield vs Treasuries",  'HYG',  'IEI',   'ratio'),
    ("Small Cap vs Large Cap",    'JKJ',  'JKD',   'ratio'),
    ("Growth vs Value",           'JKH',  'JKI',   'ratio'),
    ("IBD 50 vs S&P",             'FFTY', 'IVV',   'ratio'),
    ("Consumer Discr. vs Staples",'IYC',  'IYK',   'ratio'),
    ("Consumer Svcs vs Utils",    'IYC',  'IDU',   'ratio'),
    ("Emerging Markets vs S&P",   'EEM',  'IVV',   'ratio'),
    ("High Beta Growth vs S&P",   'ARKK', 'IVV',   'ratio'),
    ("QQQJ vs QQQ",               'QQQJ', 'QQQ',   'ratio'),
    ("Argentina vs S&P",          'ARGT', 'IVV',   'ratio'),
    ("S&P vs Treasuries 20YR",    'IVV',  'TLT',   'ratio'),
    ("Silver vs Gold",            'SLV',  'GLD',   'ratio'),
    ("Copper vs Gold",            'CPER', 'GLD',   'ratio'),
    ("Wood vs Gold",              'WOOD', 'GLD',   'ratio'),
    ("Stocks vs Gold",            'IVV',  'IAU',   'ratio'),
    ("Stocks vs Commodities",     'IVV',  'GSG',   'ratio'),
]

def compute_matrix(prices: dict) -> list[dict]:
    results = []
    for name, num, den, kind in MATRIX_DEFS:
        if kind == 'momentum':
            s = prices['IVV']
            ma = s.rolling(TREND_MA).mean()
            v = s.iloc[-1]; m = ma.iloc[-1]
            hi = m * (1 + MATRIX_BAND); lo = m * (1 - MATRIX_BAND)
        else:
            r = prices[num] / prices[den]
            idx = (r / r.iloc[0]) * 100
            ma = idx.rolling(TREND_MA).mean()
            v = idx.iloc[-1]; m = ma.iloc[-1]
            hi = m * (1 + MATRIX_BAND); lo = m * (1 - MATRIX_BAND)

        if   v > hi: sig, flag = 'Bullish',  1
        elif v < lo: sig, flag = 'Bearish', -1
        else:        sig, flag = 'Neutral',  0

        pct = int(min(100, max(0, ((v - lo) / (hi - lo)) * 100))) if hi != lo else 50
        results.append({'n': name, 'sig': sig, 'f': flag, 'pct': pct})
    return results

# ── Trend history ──────────────────────────────────────────────────────────────
def build_trend_history(prices: dict, n_days: int = 20) -> list[dict]:
    """Compute trend score for last n_days using rolling window."""
    ivv = prices['IVV']
    history = []
    labels = ['Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec',
              'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct']

    for offset in range(n_days - 1, -1, -1):
        shifted = {k: v.iloc[:len(v)-offset] if offset > 0 else v
                   for k, v in prices.items()}
        # Need aligned series
        aligned = pd.DataFrame(shifted).dropna()
        p = {k: aligned[k] for k in shifted if k in aligned.columns}
        if len(aligned) < TREND_MA + 20:
            continue
        score, _ = compute_trend_score(p)
        date = ivv.index[-1 - offset] if offset < len(ivv.index) else ivv.index[0]
        history.append({
            'd': date.strftime('%b %d'),
            's': score,
        })
    return history

# ── Ranking ────────────────────────────────────────────────────────────────────
def add_ranks(results: list) -> list:
    df = pd.DataFrame(results)
    df['rank_today'] = df['cx'].rank(ascending=False, method='average').astype(int)
    df['rank_30ago'] = df['cx30'].rank(ascending=False, method='average').astype(int)
    df['rank_climb'] = df['rank_30ago'] - df['rank_today']
    return df.to_dict('records')

# ── Download ───────────────────────────────────────────────────────────────────
def download_all():
    all_tickers = list(dict.fromkeys(
        [t for t,_,_ in ETFS] + EXTRA_TICKERS
    ))
    total = len(all_tickers)
    print(f"\nDownloading {total} tickers from Yahoo Finance...")

    data = {}
    for i, ticker in enumerate(all_tickers, 1):
        df = fetch(ticker)
        if not df.empty:
            data[ticker] = df
        print(f"  [{i:>3}/{total}] {ticker:<6} {'ok' if ticker in data else 'failed'}")
        time.sleep(0.15)

    return data

# ── HTML ───────────────────────────────────────────────────────────────────────
def generate_html(results, matrix, trend_score_val, trend_history,
                  regime, regime_days, date_str) -> str:

    ranked = add_ranks(results)
    total  = len(ranked)

    # Category data
    cat_data = {CAT_DISPLAY[c]: [] for c in CAT_ORDER}
    for r in sorted(ranked, key=lambda x: x['rank_today']):
        cat = r['category']
        if cat not in [c for c in CAT_ORDER]: continue
        label = CAT_DISPLAY[cat]
        cat_data[label].append({
            "tkr":r["ticker"],"name":r["name"],
            "y":round(r["yearly"]/100,4),"m3":round(r["perf_3m"]/100,4),"m1":round(r["perf_1m"]/100,4),
            "cx":r["cx"],"r":r["rank_today"],"r30":r["rank_30ago"],"climb":r["rank_climb"],
            "n50":r["net_50"],"n20":r["net_20"],"n5":r["net_5"],
            "e20":r.get("vs_ema20",0),"s50":r.get("vs_sma50",0),
        })

    # Springs
    springs_raw = [r for r in ranked if r['rank_climb'] >= 25 and r['yearly'] < 65]
    springs_raw.sort(key=lambda x: x['rank_climb'], reverse=True)
    colors = ["#c6ff00","#1de9b6","#ffd740","#448aff"]
    springs = [{"tkr":s["ticker"],"name":s["name"],"cat":CAT_DISPLAY.get(s["category"],s["category"]),
                "y":round(s["yearly"]/100,4),"m1":round(s["perf_1m"]/100,4),
                "cx":s["cx"],"cx30":s["cx30"],"r":s["rank_today"],"r30":s["rank_30ago"],
                "climb":s["rank_climb"],"n50":s["net_50"],"n20":s["net_20"],"n5":s["net_5"],
                "col":colors[i%len(colors)]}
               for i,s in enumerate(springs_raw[:8])]

    top4 = sorted(ranked, key=lambda x: x['rank_today'])[:4]
    top4_html = "\n".join(
        f'<div class="mini-row"><span class="mini-tkr">{r["ticker"]}</span>'
        f'<span class="mini-name">{r["name"]}</span>'
        f'<span class="mini-perf pos">#{r["rank_today"]} · Score {r["cx"]:.2f}</span></div>'
        for r in top4
    )

    matrix_score = sum(m['f'] for m in matrix)
    bulls = sum(1 for m in matrix if m['f'] > 0)
    ntrls = sum(1 for m in matrix if m['f'] == 0)
    bears = sum(1 for m in matrix if m['f'] < 0)

    trend_pct    = int(trend_score_val * 100)
    regime_color = "var(--bull)" if regime == "Stocks" else "var(--sky)"
    ms_color     = "var(--bull)" if matrix_score > 0 else "var(--bear)" if matrix_score < 0 else "var(--neut)"

    trend_js    = json.dumps(trend_history, separators=(",",":"))
    matrix_js   = json.dumps(matrix,        separators=(",",":"))
    springs_js  = json.dumps(springs,        separators=(",",":"))
    cat_js_parts= [f'  "{k}":{json.dumps(v,separators=(",",":"))}' for k,v in cat_data.items()]
    cat_js      = "{\n" + ",\n".join(cat_js_parts) + "\n}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Macro ETF Monitor — {date_str}</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{{--ink:#0e1117;--ink2:#161b25;--ink3:#1c2232;--rule:#212840;--rule2:#2a3350;--smoke:#9ba8c4;--mist:#5a6580;--ghost:#2e3a52;--bull:#1de9b6;--bull2:rgba(29,233,182,.08);--bear:#ff5370;--bear2:rgba(255,83,112,.08);--neut:#ffd740;--neut2:rgba(255,215,64,.08);--sky:#448aff;--sky2:rgba(68,138,255,.10);--r:8px}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--ink);color:var(--smoke);font-family:'IBM Plex Sans',sans-serif;font-size:13px;line-height:1.6}}
.topbar{{display:flex;align-items:center;justify-content:space-between;padding:0 36px;height:56px;border-bottom:1px solid var(--rule);background:rgba(14,17,23,.95);position:sticky;top:0;z-index:200}}
.brand{{font-family:'Syne',sans-serif;font-weight:800;font-size:15px;color:#e8eeff}}.brand em{{color:var(--bull);font-style:normal}}
.brand-date{{font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--mist);margin-left:12px}}
.regime-chip{{display:flex;align-items:center;gap:7px;padding:5px 13px;border-radius:20px;background:var(--sky2);border:1px solid rgba(68,138,255,.3);font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--sky);letter-spacing:.8px}}
.pulse{{width:6px;height:6px;border-radius:50%;background:{regime_color};animation:blink 2.4s ease-in-out infinite}}
@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}
.wrap{{max-width:1400px;margin:0 auto;padding:28px 36px 60px}}
.sec-head{{display:flex;align-items:center;gap:10px;margin:28px 0 16px}}.sec-head:first-child{{margin-top:0}}
.sec-line{{flex:1;height:1px;background:var(--rule)}}
.sec-label{{font-family:'IBM Plex Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:1.8px;color:var(--mist);white-space:nowrap}}
.hero-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.card{{background:var(--ink2);border:1px solid var(--rule);border-radius:var(--r);padding:20px 22px}}
.card-tag{{font-family:'IBM Plex Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:1.5px;color:var(--mist);margin-bottom:12px}}
.stat-big{{font-family:'Syne',sans-serif;font-size:44px;font-weight:700;line-height:1;margin-bottom:8px}}
.stat-sub{{font-size:11px;color:var(--mist);margin-top:6px}}.stat-sub strong{{color:var(--smoke)}}
.gauge{{height:4px;border-radius:2px;background:var(--rule2);overflow:hidden;margin:10px 0 4px}}
.gauge-fill{{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--bear) 0%,var(--neut) 50%,var(--bull) 100%)}}
.gauge-ticks{{display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:9px;color:var(--ghost)}}
.pills{{display:flex;gap:6px;margin-top:10px;flex-wrap:wrap}}
.pill{{padding:3px 9px;border-radius:20px;font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:500}}
.p-b{{background:var(--bull2);color:var(--bull);border:1px solid rgba(29,233,182,.2)}}
.p-n{{background:var(--neut2);color:var(--neut);border:1px solid rgba(255,215,64,.2)}}
.p-r{{background:var(--bear2);color:var(--bear);border:1px solid rgba(255,83,112,.2)}}
.reg-bar{{height:5px;border-radius:3px;background:var(--rule2);overflow:hidden;margin-top:10px}}
.reg-fill{{height:100%;border-radius:3px;background:{regime_color};opacity:.7}}
.reg-labels{{display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:9px;color:var(--mist);margin-top:4px}}
.mini-list{{display:flex;flex-direction:column;gap:7px;margin-top:8px}}
.mini-row{{display:flex;align-items:center;gap:8px;padding:6px 10px;border-radius:5px;background:var(--ink3);border:1px solid var(--rule)}}
.mini-tkr{{font-family:'IBM Plex Mono',monospace;font-weight:500;font-size:12px;color:#e8eeff;min-width:38px}}
.mini-name{{flex:1;font-size:11px;color:var(--mist)}}
.mini-perf{{font-family:'IBM Plex Mono',monospace;font-size:12px}}
.pos{{color:var(--bull)}}.neg{{color:var(--bear)}}
.content-row{{display:grid;grid-template-columns:1.15fr .85fr;gap:14px}}
.chart-card,.mx-card{{background:var(--ink2);border:1px solid var(--rule);border-radius:var(--r);padding:22px 24px}}
.chart-title{{font-family:'IBM Plex Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:var(--mist);margin-bottom:16px;display:flex;align-items:center;justify-content:space-between}}
.chart-legend{{display:flex;gap:16px}}
.leg-item{{display:flex;align-items:center;gap:5px;font-size:10px;color:var(--mist);font-family:'IBM Plex Mono',monospace}}
.leg-dot{{width:8px;height:3px;border-radius:2px}}
.chart-wrap{{height:130px;position:relative}}
.stat-box{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:6px;overflow:hidden;margin-top:14px}}
.sbi{{background:var(--ink3);padding:10px 12px;text-align:center}}
.sbi-val{{font-family:'IBM Plex Mono',monospace;font-size:14px;font-weight:500;color:#e8eeff;display:block}}
.sbi-lbl{{font-size:9px;color:var(--mist);margin-top:2px;text-transform:uppercase;letter-spacing:.8px;font-family:'IBM Plex Mono',monospace}}
.mx-title{{font-family:'IBM Plex Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:var(--mist);margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid var(--rule)}}
.mx-row{{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid rgba(33,40,64,.5)}}.mx-row:last-child{{border-bottom:none}}
.mx-name{{flex:1;font-size:11.5px;color:var(--smoke)}}
.mx-bar{{width:54px;height:3px;border-radius:2px;background:var(--rule2);overflow:hidden}}
.mx-fill{{height:100%;border-radius:2px}}
.sig{{min-width:56px;text-align:center;padding:2px 7px;border-radius:4px;font-family:'IBM Plex Mono',monospace;font-size:9px;font-weight:500;letter-spacing:.5px}}
.sig-b{{background:var(--bull2);color:var(--bull);border:1px solid rgba(29,233,182,.18)}}
.sig-r{{background:var(--bear2);color:var(--bear);border:1px solid rgba(255,83,112,.18)}}
.sig-n{{background:var(--neut2);color:var(--neut);border:1px solid rgba(255,215,64,.18)}}
.springs-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
.spr-card{{background:var(--ink2);border:1px solid var(--rule);border-radius:var(--r);padding:16px;position:relative;overflow:hidden}}
.spr-accent{{position:absolute;top:0;left:0;right:0;height:2px}}
.spr-cat{{font-family:'IBM Plex Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:.8px;color:var(--mist);margin-bottom:4px}}
.spr-tkr{{font-family:'Syne',sans-serif;font-weight:700;font-size:18px;color:#e8eeff;line-height:1;margin-bottom:2px}}
.spr-name{{font-size:11px;color:var(--mist);margin-bottom:10px}}
.spr-row{{display:flex;justify-content:space-between;margin-bottom:4px}}
.spr-lbl{{font-size:10px;color:var(--mist)}}
.spr-val{{font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:500}}
.sv-up{{color:var(--bull)}}.sv-dn{{color:var(--bear)}}.sv-wh{{color:var(--smoke)}}
.climb-val{{font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:500;color:var(--bull)}}
.rank-bar{{height:5px;border-radius:3px;background:var(--rule2);overflow:hidden;position:relative;margin-top:4px}}
.rb-ago{{position:absolute;left:0;top:0;height:100%;border-radius:3px;background:var(--ghost)}}
.rb-now{{position:absolute;left:0;top:0;height:100%;border-radius:3px;background:var(--bull)}}
.acc-row{{display:flex;gap:12px;margin-top:8px;padding-top:8px;border-top:1px solid var(--rule)}}
.acc-item{{text-align:center}}
.acc-lbl{{font-size:9px;color:var(--mist);font-family:'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:.6px}}
.acc-val{{font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:500}}
.etf-section{{background:var(--ink2);border:1px solid var(--rule);border-radius:var(--r);overflow:hidden}}
.cat-tabs{{display:flex;border-bottom:1px solid var(--rule);overflow-x:auto;scrollbar-width:none}}.cat-tabs::-webkit-scrollbar{{display:none}}
.cat-tab{{padding:12px 20px;font-family:'IBM Plex Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:.8px;color:var(--mist);cursor:pointer;white-space:nowrap;border-bottom:2px solid transparent;background:transparent;border-top:none;border-left:none;border-right:none;transition:color .15s}}
.cat-tab:hover{{color:var(--smoke)}}.cat-tab.on{{color:var(--bull);border-bottom-color:var(--bull)}}
.cat-panel{{display:none}}.cat-panel.on{{display:block}}
table.etf{{width:100%;border-collapse:collapse}}
table.etf th{{padding:9px 14px;font-family:'IBM Plex Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:1px;color:var(--mist);text-align:left;background:var(--ink3);border-bottom:1px solid var(--rule);white-space:nowrap}}
table.etf th.r{{text-align:right}}
table.etf th[onclick]{{cursor:pointer;user-select:none}}
table.etf th[onclick]:hover{{color:var(--smoke);background:var(--rule)}}
table.etf td{{padding:9px 14px;border-bottom:1px solid rgba(33,40,64,.6);font-size:12px;color:var(--smoke)}}
table.etf tr:last-child td{{border-bottom:none}}
table.etf tr:hover td{{background:rgba(255,255,255,.015)}}
table.etf td.r{{text-align:right}}
.etf-tkr{{font-family:'IBM Plex Mono',monospace;font-weight:500;font-size:13px;color:#e8eeff}}
.etf-name{{color:var(--mist);font-size:11px}}
.cx-cell{{display:flex;align-items:center;gap:8px;justify-content:flex-end}}
.cx-bar{{width:48px;height:3px;border-radius:2px;background:var(--rule2);overflow:hidden}}
.cx-fill{{height:100%;border-radius:2px}}
.rb{{font-family:'IBM Plex Mono',monospace;font-size:10px;padding:1px 6px;border-radius:3px}}
.rb-up{{background:var(--bull2);color:var(--bull)}}.rb-dn{{background:var(--bear2);color:var(--bear)}}.rb-flat{{background:var(--rule);color:var(--mist)}}
.foot{{margin-top:40px;padding:16px 0;border-top:1px solid var(--rule);display:flex;justify-content:space-between}}
.foot-note{{font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--mist)}}
@media(max-width:1100px){{.hero-row{{grid-template-columns:1fr 1fr}}.content-row{{grid-template-columns:1fr}}.springs-grid{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:640px){{.wrap{{padding:16px}}.topbar{{padding:0 16px}}.hero-row{{grid-template-columns:1fr}}.springs-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header class="topbar">
  <div style="display:flex;align-items:baseline;gap:4px">
    <div class="brand">Macro <em>ETF</em> Monitor</div>
    <div class="brand-date">{date_str} · Python + Yahoo Finance</div>
  </div>
  <div style="display:flex;align-items:center;gap:16px">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--mist)">Regime avg: <strong style="color:var(--neut)">43 days</strong> · Day {regime_days}</span>
    <div class="regime-chip"><div class="pulse"></div>{regime.upper()} REGIME</div>
  </div>
</header>
<div class="wrap">
  <div class="sec-head"><div class="sec-label">Market Overview</div><div class="sec-line"></div></div>
  <div class="hero-row">
    <div class="card">
      <div class="card-tag">Trend Score</div>
      <div class="stat-big" style="color:{'var(--bull)' if trend_score_val>=0.7 else 'var(--bear)'}">{trend_pct}%</div>
      <div class="gauge"><div class="gauge-fill" style="width:{trend_pct}%"></div></div>
      <div class="gauge-ticks"><span>Bearish</span><span>Neutral</span><span>Bullish</span></div>
      <div class="stat-sub">Threshold: <strong>70%</strong> · Regime switch point</div>
    </div>
    <div class="card">
      <div class="card-tag">Matrix Score</div>
      <div class="stat-big" style="color:{ms_color}">{'+' if matrix_score>0 else ''}{matrix_score}</div>
      <div class="pills">
        <span class="pill p-b">{bulls} Bullish</span>
        <span class="pill p-n">{ntrls} Neutral</span>
        <span class="pill p-r">{bears} Bearish</span>
      </div>
      <div class="stat-sub">17 macro indicators · Risk-on/off signal</div>
    </div>
    <div class="card">
      <div class="card-tag">Current Regime</div>
      <div class="stat-big" style="color:{regime_color};font-size:32px;margin-top:4px">{regime.upper()}</div>
      <div class="reg-bar"><div class="reg-fill" style="width:{min(100,int(regime_days/43*100))}%"></div></div>
      <div class="reg-labels"><span>Day {regime_days} of ~43 avg</span><span>{'~'+str(max(0,43-regime_days))+' days to avg' if regime_days<43 else 'Past avg duration'}</span></div>
      <div class="stat-sub">Bonds → Stocks threshold: <strong>70%</strong> for 20+ days</div>
    </div>
    <div class="card">
      <div class="card-tag">Exponential Regression Ranking — Top 4</div>
      <div class="mini-list">{top4_html}</div>
    </div>
  </div>

  <div class="sec-head" style="margin-top:28px"><div class="sec-label">Trend History &amp; Matrix Signals</div><div class="sec-line"></div></div>
  <div class="content-row">
    <div class="chart-card">
      <div class="chart-title">
        <span>Trend Score — Last 20 Trading Days</span>
        <div class="chart-legend">
          <div class="leg-item"><div class="leg-dot" style="background:var(--sky)"></div>Bonds</div>
          <div class="leg-item"><div class="leg-dot" style="background:var(--bull)"></div>Stocks</div>
        </div>
      </div>
      <div class="chart-wrap"><canvas id="trendChart"></canvas></div>
      <div class="stat-box">
        <div class="sbi"><span class="sbi-val" style="color:{regime_color}">{regime_days} days</span><span class="sbi-lbl">In {regime}</span></div>
        <div class="sbi"><span class="sbi-val" style="color:var(--neut)">43 days</span><span class="sbi-lbl">Avg. Duration</span></div>
        <div class="sbi"><span class="sbi-val" style="color:{'var(--bull)' if trend_score_val>=0.7 else 'var(--bear)'}">{trend_pct}%</span><span class="sbi-lbl">Score Today</span></div>
      </div>
    </div>
    <div class="mx-card">
      <div class="mx-title">Matrix Indicators — All 17</div>
      <div id="matrixList" style="max-height:320px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--rule2) transparent"></div>
    </div>
  </div>

  <div class="sec-head" style="margin-top:28px">
    <div class="sec-label">Coiled Springs — 30D Variation in Ranking</div>
    <div class="sec-line"></div>
  </div>
  <p style="font-size:12px;color:var(--mist);margin-bottom:14px;max-width:700px">ETFs gaining the most ranking positions over the last 30 trading days. Ranking = annualised log-linear slope × R² over 91 days.</p>
  <div class="springs-grid" id="springsGrid"></div>

  <div class="sec-head" style="margin-top:28px"><div class="sec-label">ETF Rankings by Category</div><div class="sec-line"></div></div>
  <div class="etf-section">
    <div class="cat-tabs" id="catTabs"></div>
    <div id="catPanels"></div>
  </div>
</div>
<footer style="max-width:1400px;margin:0 auto;padding:0 36px">
  <div class="foot">
    <div class="foot-note">Data: Yahoo Finance · Acc/Dist ±{ACC_THRESHOLD}% · Vol ×{VOL_MULT} · SMA{VOL_MA_LEN} · Matrix band ±{int(MATRIX_BAND*100)}%</div>
    <div class="foot-note">Exp. Regression: log-linear slope × R² · {SLOPE_WINDOW}-day window · {total} ETFs · {date_str}</div>
  </div>
</footer>
<script>
const TREND_HISTORY={trend_js};
const MATRIX={matrix_js};
const SPRINGS={springs_js};
const CAT_DATA={cat_js};
const SORT_STATE={{}};
const TOTAL={total};

// Chart
(function(){{
  const ctx=document.getElementById('trendChart').getContext('2d');
  const reg='{regime}';
  new Chart(ctx,{{type:'bar',data:{{labels:TREND_HISTORY.map(d=>d.d),datasets:[{{data:TREND_HISTORY.map(d=>d.s),backgroundColor:TREND_HISTORY.map(d=>d.s>=0.7?'rgba(29,233,182,.75)':'rgba(68,138,255,.75)'),borderRadius:3,borderSkipped:false}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{backgroundColor:'#1c2232',borderColor:'#2a3350',borderWidth:1,titleFont:{{family:'IBM Plex Mono',size:10}},bodyFont:{{family:'IBM Plex Mono',size:11}},callbacks:{{label:c=>` Score: ${{(c.raw*100).toFixed(0)}}%`}}}}}},scales:{{x:{{grid:{{color:'rgba(255,255,255,.03)'}},ticks:{{color:'#5a6580',font:{{family:'IBM Plex Mono',size:8}},maxRotation:45}}}},y:{{min:0,max:1,grid:{{color:'rgba(255,255,255,.04)'}},ticks:{{color:'#5a6580',font:{{family:'IBM Plex Mono',size:9}},stepSize:.2,callback:v=>v*100+'%'}}}}}}}}}});
}})();

// Matrix
document.getElementById('matrixList').innerHTML=MATRIX.map(m=>{{
  const cls=m.f>0?'sig-b':m.f<0?'sig-r':'sig-n';
  const lbl=m.f>0?'Bullish':m.f<0?'Bearish':'Neutral';
  const bc=m.f>0?'var(--bull)':m.f<0?'var(--bear)':'var(--neut)';
  return`<div class="mx-row"><div class="mx-name">${{m.n}}</div><div class="mx-bar"><div class="mx-fill" style="width:${{m.pct}}%;background:${{bc}}"></div></div><div class="sig ${{cls}}">${{lbl}}</div></div>`;
}}).join('');

// Springs
document.getElementById('springsGrid').innerHTML=SPRINGS.map((s,i)=>{{
  const yc=s.y>.005?'sv-up':s.y<-.005?'sv-dn':'sv-wh';
  const m1c=s.m1>.005?'sv-up':s.m1<-.005?'sv-dn':'sv-wh';
  const n50c=s.n50>0?'sv-up':s.n50<0?'sv-dn':'sv-wh';
  const n20c=s.n20>0?'sv-up':s.n20<0?'sv-dn':'sv-wh';
  const n5c=s.n5>0?'sv-up':s.n5<0?'sv-dn':'sv-wh';
  const tp=Math.round(((TOTAL-s.r)/(TOTAL-1))*100);
  const ap=Math.round(((TOTAL-s.r30)/(TOTAL-1))*100);
  return`<div class="spr-card">
    <div class="spr-accent" style="background:linear-gradient(90deg,${{s.col}}88,${{s.col}}22)"></div>
    <div class="spr-cat">${{s.cat}}</div>
    <div class="spr-tkr">${{s.tkr}}</div>
    <div class="spr-name">${{s.name}}</div>
    <div class="spr-row"><span class="spr-lbl">Annual</span><span class="spr-val ${{yc}}">${{(s.y*100).toFixed(1)}}%</span></div>
    <div class="spr-row"><span class="spr-lbl">Last Month</span><span class="spr-val ${{m1c}}">${{(s.m1*100).toFixed(1)}}%</span></div>
    <div class="spr-row"><span class="spr-lbl">Exp. Score</span><span class="spr-val sv-up">${{s.cx.toFixed(3)}}</span></div>
    <div class="spr-row"><span class="spr-lbl">30D Variation</span><span class="climb-val">↑ +${{s.climb}}</span></div>
    <div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:9px;color:var(--mist);margin-top:6px">
      <span>#${{s.r}} today</span><span>#${{s.r30}} 30d ago</span>
    </div>
    <div class="rank-bar"><div class="rb-ago" style="width:${{ap}}%"></div><div class="rb-now" style="width:${{tp}}%"></div></div>
    <div class="acc-row">
      <div class="acc-item"><div class="acc-lbl">Net 50D</div><div class="acc-val ${{n50c}}">${{s.n50>0?'+':''}}${{s.n50}}</div></div>
      <div class="acc-item"><div class="acc-lbl">Net 20D</div><div class="acc-val ${{n20c}}">${{s.n20>0?'+':''}}${{s.n20}}</div></div>
      <div class="acc-item"><div class="acc-lbl">Net 5D</div><div class="acc-val ${{n5c}}">${{s.n5>0?'+':''}}${{s.n5}}</div></div>
    </div>
  </div>`;
}}).join('');

// Category tabs
function buildRows(cat,col,dir){{
  let d=[...CAT_DATA[cat]];
  const keys=[null,'r','y','m3','m1','cx','climb','e20','s50','n50','n20','n5'];
  const k=keys[col];
  if(k) d.sort((a,b)=>dir==='asc'?a[k]-b[k]:b[k]-a[k]);
  return d.map(e=>{{
    const yc=e.y>.005?'pos':e.y<-.005?'neg':'';
    const m3c=e.m3>.005?'pos':e.m3<-.005?'neg':'';
    const m1c=e.m1>.005?'pos':e.m1<-.005?'neg':'';
    const cxc=e.cx>0.5?'pos':e.cx>0?'':e.cx<0?'neg':'';
    const cxP=Math.min(100,Math.max(0,(e.cx/4)*100));
    const cxF=e.cx>0?'var(--bull)':'var(--bear)';
    const ca=Math.abs(e.climb);
    const cc=e.climb>=5?'rb-up':e.climb<=-5?'rb-dn':'rb-flat';
    const ct=e.climb>=5?`↑${{e.climb}}`:e.climb<=-5?`↓${{ca}}`:`${{e.climb>0?'+':''}}${{e.climb}}`;
    const n50c=e.n50>0?'var(--bull)':e.n50<0?'var(--bear)':'var(--smoke)';
    const n20c=e.n20>0?'var(--bull)':e.n20<0?'var(--bear)':'var(--smoke)';
    const n5c=e.n5>0?'var(--bull)':e.n5<0?'var(--bear)':'var(--smoke)';
    const e20c=e.e20>0?'var(--bull)':e.e20<0?'var(--bear)':'var(--smoke)';
    const s50c=e.s50>0?'var(--bull)':e.s50<0?'var(--bear)':'var(--smoke)';
    return`<tr>
      <td><div class="etf-tkr">${{e.tkr}}</div><div class="etf-name">${{e.name}}</div></td>
      <td class="r" style="font-family:'IBM Plex Mono',monospace;font-size:11px"><strong style="color:#e8eeff">#${{e.r}}</strong></td>
      <td class="r ${{yc}}">${{(e.y*100).toFixed(1)}}%</td>
      <td class="r ${{m3c}}">${{(e.m3*100).toFixed(1)}}%</td>
      <td class="r ${{m1c}}">${{(e.m1*100).toFixed(1)}}%</td>
      <td class="r"><div class="cx-cell"><span class="${{cxc}}">${{e.cx.toFixed(3)}}</span><div class="cx-bar"><div class="cx-fill" style="width:${{cxP}}%;background:${{cxF}}"></div></div></div></td>
      <td class="r"><span class="rb ${{cc}}">${{ct}}</span></td>
      <td class="r" style="font-family:'IBM Plex Mono',monospace;font-weight:500;color:${{e20c}}">${{e.e20>0?'+':''}}${{e.e20}}%</td>
      <td class="r" style="font-family:'IBM Plex Mono',monospace;font-weight:500;color:${{s50c}}">${{e.s50>0?'+':''}}${{e.s50}}%</td>
      <td class="r" style="font-family:'IBM Plex Mono',monospace;font-weight:500;color:${{n50c}}">${{e.n50>0?'+':''}}${{e.n50}}</td>
      <td class="r" style="font-family:'IBM Plex Mono',monospace;font-weight:500;color:${{n20c}}">${{e.n20>0?'+':''}}${{e.n20}}</td>
      <td class="r" style="font-family:'IBM Plex Mono',monospace;font-weight:500;color:${{n5c}}">${{e.n5>0?'+':''}}${{e.n5}}</td>
    </tr>`;
  }}).join('');
}}
function sortTable(pid,cat,col){{
  const st=SORT_STATE[pid]||{{col:1,dir:'asc'}};
  const dir=st.col===col?(st.dir==='desc'?'asc':'desc'):'desc';
  SORT_STATE[pid]={{col,dir}};
  document.querySelector(`#${{pid}} tbody`).innerHTML=buildRows(cat,col,dir);
  document.querySelectorAll(`#${{pid}} th`).forEach((th,i)=>{{
    th.innerHTML=th.innerHTML.replace(/ [▲▼]$/,'');
    if(i===col) th.innerHTML+=dir==='desc'?' ▼':' ▲';
  }});
}}
(function(){{
  const cats=Object.keys(CAT_DATA);
  const hdrs=['ETF','Exp. Rank','Annual','3M','1M','Exp. Score','30D Variation','vs EMA20','vs SMA50','Net 50D','Net 20D','Net 5D'];
  document.getElementById('catTabs').innerHTML=cats.map((c,i)=>`<button class="cat-tab ${{i===0?'on':''}}" onclick="swTab(this,'cp${{i}}')">${{c}}</button>`).join('');
  document.getElementById('catPanels').innerHTML=cats.map((cat,i)=>{{
    const pid=`cp${{i}}`;
    SORT_STATE[pid]={{col:1,dir:'asc'}};
    const ths=hdrs.map((h,ci)=>{{
      const cls=ci===0?'':' class="r"';
      const s=ci>0?`onclick="sortTable('${{pid}}','${{cat}}',${{ci}})"` :'';
      return`<th${{cls}} ${{s}}>${{h}}${{ci===1?' ▲':''}}</th>`;
    }}).join('');
    return`<div id="${{pid}}" class="cat-panel ${{i===0?'on':''}}"><table class="etf"><thead><tr>${{ths}}</tr></thead><tbody>${{buildRows(cat,1,'asc')}}</tbody></table></div>`;
  }}).join('');
}})();
function swTab(el,pid){{
  document.querySelectorAll('.cat-tab').forEach(t=>t.classList.remove('on'));
  document.querySelectorAll('.cat-panel').forEach(p=>p.classList.remove('on'));
  el.classList.add('on'); document.getElementById(pid).classList.add('on');
}}
</script>
</body>
</html>"""

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    date_str = datetime.now().strftime("%b %d, %Y")
    print(f"Macro ETF Dashboard Generator — {date_str}")

    # Download everything
    all_data = download_all()

    # Build aligned price series for trend/matrix
    matrix_tickers = list(set(t for _,n,d,_ in MATRIX_DEFS for t in [n, d] if d) |
                          {'IVV','HYG','IEI','JKJ','JKD','JKH','JKI','IYC','IYK','IDU','IAU','GSG'})
    price_df = pd.DataFrame({
        t: all_data[t]["close"]
        for t in matrix_tickers if t in all_data
    }).dropna()
    prices = {t: price_df[t] for t in price_df.columns}

    # Trend Score + history
    print("\nComputing Trend Score...")
    trend_score_val, sub_scores = compute_trend_score(prices)
    print(f"  Score: {trend_score_val} | sub-scores: {sub_scores}")

    # Build 20-day trend history
    print("Building trend history...")
    trend_history = []
    close_ivv = all_data['IVV']['close']
    for offset in range(19, -1, -1):
        shifted = {t: price_df[t].iloc[:len(price_df)-offset] if offset > 0 else price_df[t]
                   for t in prices}
        al = pd.DataFrame(shifted).dropna()
        if len(al) < TREND_MA + 15:
            continue
        p = {t: al[t] for t in al.columns}
        s, _ = compute_trend_score(p)
        idx = -1 - offset
        if abs(idx) <= len(close_ivv.index):
            d = close_ivv.index[idx].strftime('%b %d')
        else:
            d = f"T-{offset}"
        trend_history.append({"d": d, "s": s})

    # Regime
    scores_list = [h['s'] for h in trend_history]
    regime, regime_days = compute_regime(scores_list)
    print(f"  Regime: {regime} | Day {regime_days}")

    # Matrix
    print("Computing Matrix signals...")
    matrix = compute_matrix(prices)
    matrix_score = sum(m['f'] for m in matrix)
    print(f"  Matrix Score: {matrix_score}")
    for m in matrix:
        print(f"    {m['n']:<35} {m['sig']}")

    # ETF analysis
    print("\nAnalyzing ETFs...")
    results = []
    for ticker, name, cat in ETFS:
        if ticker not in all_data:
            continue
        row = analyze(ticker, name, cat, all_data[ticker])
        if row:
            results.append(row)

    ranked = add_ranks(results)
    print(f"Ranked {len(ranked)} ETFs.")

    # Generate HTML
    html = generate_html(ranked, matrix, trend_score_val, trend_history,
                         regime, regime_days, date_str)
    Path(OUTPUT_FILE).write_text(html, encoding="utf-8")
    print(f"\n✓ Dashboard saved → {OUTPUT_FILE}")
