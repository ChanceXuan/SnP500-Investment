"""Cross-market snapshot, correlations with the S&P 500 and sector moves."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fetch import NEXT_DAY_MARKETS, SECTORS, UNIVERSE


def _chg(s: pd.Series, n: int, how: str) -> float:
    if len(s) <= n:
        return float("nan")
    return float(s.iloc[-1] - s.iloc[-1 - n]) if how == "diff" else float(s.iloc[-1] / s.iloc[-1 - n] - 1)


def cross_market(raw: dict) -> pd.DataFrame:
    y = raw["yahoo"]
    spx = y["^GSPC"]["close"]
    spx_r = spx.pct_change()
    last = spx.index[-1]
    rows = []
    for sym, name, group, how in UNIVERSE:
        if sym not in y:
            continue
        s = y[sym]["close"].dropna()
        s = s[s.index <= last + pd.Timedelta(days=1)]
        r = s.diff() if how == "diff" else s.pct_change()
        ytd_base = s[s.index < pd.Timestamp(f"{last.year}-01-01")]
        ytd = (float(s.iloc[-1] - ytd_base.iloc[-1]) if how == "diff" else float(s.iloc[-1] / ytd_base.iloc[-1] - 1)) \
            if len(ytd_base) else float("nan")
        # Markets that close before New York react to a US day on their next session.
        lead = sym in NEXT_DAY_MARKETS
        rr = r.shift(-1) if lead else r
        j = pd.concat([spx_r, rr.reindex(spx_r.index)], axis=1, keys=["spx", "x"]).dropna()
        c60 = j.tail(60).corr().iloc[0, 1] if sym != "^GSPC" else 1.0
        c252 = j.tail(252).corr().iloc[0, 1] if sym != "^GSPC" else 1.0
        c_prev = j.iloc[-312:-252].corr().iloc[0, 1] if sym != "^GSPC" and len(j) > 312 else float("nan")
        beta = (np.cov(j.tail(252)["x"], j.tail(252)["spx"])[0, 1] / j.tail(252)["spx"].var()) if sym != "^GSPC" else 1.0
        rows.append({
            "symbol": sym, "name": name, "group": group, "unit": "pp" if how == "diff" else "pct",
            "last": float(s.iloc[-1]), "last_date": s.index[-1].strftime("%Y-%m-%d"),
            "chg_1d": _chg(s, 1, how), "chg_1w": _chg(s, 5, how), "chg_1m": _chg(s, 21, how),
            "chg_3m": _chg(s, 63, how), "chg_ytd": ytd,
            "corr_60d": float(c60), "corr_1y": float(c252), "corr_60d_a_year_ago": float(c_prev),
            "beta_1y": float(beta), "lag": "次日" if lead else "同日",
        })
    return pd.DataFrame(rows)


def sectors(raw: dict) -> pd.DataFrame:
    y = raw["yahoo"]
    rows = []
    for sym, name in SECTORS:
        s = y[sym]["close"].dropna()
        ma200 = s.rolling(200).mean()
        rows.append({"symbol": sym, "sector": name, "last": float(s.iloc[-1]),
                     "chg_1d": _chg(s, 1, "pct"), "chg_1w": _chg(s, 5, "pct"),
                     "chg_1m": _chg(s, 21, "pct"), "chg_3m": _chg(s, 63, "pct"),
                     "vs_200dma": float(s.iloc[-1] / ma200.iloc[-1] - 1),
                     "above_200dma": bool(s.iloc[-1] > ma200.iloc[-1])})
    return pd.DataFrame(rows).sort_values("chg_1m")


def rolling_corr(raw: dict, syms=("^TNX", "^VIX", "GC=F", "DX-Y.NYB", "BTC-USD", "^N225"), win=60) -> pd.DataFrame:
    y = raw["yahoo"]
    spx_r = y["^GSPC"]["close"].pct_change()
    out = {}
    how = {s: h for s, _, _, h in UNIVERSE}
    for sym in syms:
        s = y[sym]["close"]
        r = s.diff() if how.get(sym) == "diff" else s.pct_change()
        if sym in NEXT_DAY_MARKETS:
            r = r.shift(-1)
        j = pd.concat([spx_r, r.reindex(spx_r.index)], axis=1).dropna()
        out[sym] = j.iloc[:, 0].rolling(win).corr(j.iloc[:, 1])
    df = pd.DataFrame(out)
    return df[df.index >= df.index[-1] - pd.Timedelta(days=3 * 365)].dropna(how="all")
