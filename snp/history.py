"""Long-run history of the S&P 500 (price index since 1927-12-30).

Produces: annual returns, a monthly series for the log chart, bear and
near-bear episodes, loss odds by holding period, event studies that match
today's situation, CAPE-bucketed 10-year returns and seasonality.
Dividends are approximated from multpl's monthly dividend yield.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

THRESH = 0.18   # an episode starts with an 18% fall from a peak (closing basis)
REBOUND = 0.25  # ...and ends once the index rallies 25% off its low or regains the peak

CAUSES = {  # peak month -> cause (written from the historical record)
    "1929-09": "1929年大崩盘：杠杆投机泡沫破裂，引发大萧条",
    "1930-04": "大萧条深化：银行倒闭潮、通缩、关税战(斯穆特-霍利法案)",
    "1931-02": "大萧条：欧洲银行危机蔓延(奥地利信贷银行倒闭)",
    "1931-06": "英国放弃金本位，美联储为保汇率加息",
    "1931-11": "大萧条最深处：银行倒闭潮与通缩，1932年7月见底(较1929年高点约-86%)",
    "1932-09": "大萧条余波：银行业危机与金本位动荡",
    "1933-07": "罗斯福新政初期的反复：美元贬值后投机退潮",
    "1934-02": "复苏中的反复：经济放缓与新政监管",
    "1937-03": "1937年衰退：货币与财政过早收紧",
    "1938-11": "二战爆发与战争不确定性",
    "1939-10": "二战扩大：德国闪击西欧、法国沦陷",
    "1940-11": "二战升级与美国参战(珍珠港事件)",
    "1946-05": "战后复员：通胀飙升、价格管制解除",
    "1948-06": "1948-49年衰退",
    "1956-08": "艾森豪威尔衰退、苏伊士运河危机",
    "1961-12": "肯尼迪滑坡：高估值+钢铁价格危机",
    "1966-02": "1966年信贷紧缩：美联储收紧、越战支出",
    "1968-11": "通胀抬头、美联储加息，滞胀前奏",
    "1973-01": "石油危机、布雷顿森林体系瓦解、水门事件、滞胀",
    "1976-09": "通胀再起、美元走弱",
    "1980-11": "沃尔克大幅加息遏制通胀，双底衰退",
    "1987-08": "1987年黑色星期一：程序化交易与估值偏高，单日跌20.5%",
    "1990-07": "伊拉克入侵科威特、油价飙升、美国衰退",
    "1998-07": "亚洲金融危机、俄罗斯违约、LTCM爆仓",
    "2000-03": "互联网泡沫破裂、9·11、会计丑闻",
    "2007-10": "全球金融危机：次贷与雷曼破产",
    "2011-04": "欧债危机、美国主权评级被下调",
    "2018-09": "美联储加息+缩表、中美贸易战",
    "2020-02": "新冠疫情冲击，史上最快熊市",
    "2022-01": "通胀高企、美联储40年来最激进加息",
    "2025-02": "“解放日”全面关税冲击",
    "2026-10": "本轮：AI资本开支与估值担忧、美联储加息周期、10年美债≈5%",
}


def _cause(peak: pd.Timestamp) -> str:
    best, gap = "", 99
    for k, v in CAUSES.items():
        y, m = map(int, k.split("-"))
        g = abs((peak.year - y) * 12 + peak.month - m)
        if g < gap:
            best, gap = v, g
    return best if gap <= 5 else "—"


def episodes(close: pd.Series) -> pd.DataFrame:
    peak_d, peak_v = close.index[0], close.iloc[0]
    trough_d, trough_v = None, None
    state = "up"
    out = []
    for d, v in close.items():
        if state == "up":
            if v > peak_v:
                peak_d, peak_v = d, v
            elif v <= peak_v * (1 - THRESH):
                state, trough_d, trough_v = "down", d, v
        else:
            if v < trough_v:
                trough_d, trough_v = d, v
            elif v >= trough_v * (1 + REBOUND) or v >= peak_v:
                out.append((peak_d, peak_v, trough_d, trough_v))
                state, peak_d, peak_v = "up", d, v
    ongoing = state == "down"
    if ongoing:
        out.append((peak_d, peak_v, trough_d, trough_v))
    rows = []
    for i, (pd_, pv, td, tv) in enumerate(out):
        after = close[close.index > td]
        rec = after[after >= pv]
        rec_d = rec.index[0] if len(rec) else None
        seg = close[(close.index >= pd_) & (close.index <= td)]
        first10 = seg[seg <= pv * 0.9]
        f10 = first10.index[0] if len(first10) else None

        def fwd(d0, n=252):
            i0 = close.index.get_loc(d0)
            return float(close.iloc[i0 + n] / close.iloc[i0] - 1) if i0 + n < len(close) else None

        depth = tv / pv - 1
        rows.append({
            "peak_date": pd_.strftime("%Y-%m-%d"), "peak": float(pv),
            "trough_date": td.strftime("%Y-%m-%d"), "trough": float(tv),
            "depth": float(depth),
            "kind": "熊市(≥20%)" if depth <= -0.20 else "准熊市(18%-20%)",
            "months_down": round((td - pd_).days / 30.44, 1),
            "recovery_date": rec_d.strftime("%Y-%m-%d") if rec_d is not None else None,
            "months_to_recover": round((rec_d - td).days / 30.44, 1) if rec_d is not None else None,
            "ret_12m_after_trough": fwd(td),
            "ret_12m_after_first_minus10": fwd(f10) if f10 is not None else None,
            "cause": _cause(pd_),
            "ongoing": bool(ongoing and i == len(out) - 1),
        })
    return pd.DataFrame(rows)


def current_drawdown(df: pd.DataFrame) -> dict:
    close = df["close"]
    ath_d = close[close == close.cummax()].index[-1]
    seg = close[close.index >= ath_d]
    low_d = seg.idxmin()
    return {"peak_date": ath_d.strftime("%Y-%m-%d"), "peak": float(close[ath_d]),
            "low_date": low_d.strftime("%Y-%m-%d"), "low": float(seg.min()),
            "dd_now": float(close.iloc[-1] / close[ath_d] - 1),
            "days_since_peak": int(len(seg) - 1)}


def monthly(df: pd.DataFrame, raw: dict) -> pd.DataFrame:
    m = df["close"].resample("ME").last().to_frame("close")
    dy = raw["multpl"]["s-p-500-dividend-yield"].copy()
    dy.index = dy.index.to_period("M").to_timestamp("M")
    dy = dy[~dy.index.duplicated(keep="last")]
    cape = raw["multpl"]["shiller-pe"].copy()
    cape.index = cape.index.to_period("M").to_timestamp("M")
    cape = cape[~cape.index.duplicated(keep="last")]
    m["div_yield"] = dy.reindex(m.index).ffill()
    m["cape"] = cape.reindex(m.index).ffill()
    m["price_ret"] = m["close"].pct_change()
    m["total_ret"] = m["price_ret"] + m["div_yield"].shift(1).fillna(m["div_yield"]) / 100 / 12
    m["tr_index"] = (1 + m["total_ret"].fillna(0)).cumprod() * m["close"].iloc[0]
    m["dd"] = m["close"] / m["close"].cummax() - 1
    rec = raw["fred"]["USREC"].copy()
    rec.index = rec.index.to_period("M").to_timestamp("M")
    m["recession"] = rec.reindex(m.index).fillna(0)
    return m


def annual(df: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    ye = close.groupby(close.index.year).last()
    rows = []
    prev = None
    for yr, c in ye.items():
        if prev is not None:
            seg = close[close.index.year == yr]
            run = pd.concat([pd.Series([prev]), seg.reset_index(drop=True)])
            mdd = float((run / run.cummax() - 1).min())
            dy = m.loc[m.index.year == yr, "div_yield"].mean()
            pr = c / prev - 1
            rows.append({"year": int(yr), "close": float(c), "price_return": float(pr),
                         "total_return_est": float(pr + (dy or 0) / 100),
                         "max_drawdown": mdd, "ytd": bool(yr == close.index[-1].year)})
        prev = c
    return pd.DataFrame(rows)


def holding_periods(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for yrs in (1, 3, 5, 10, 15, 20, 30):
        n = yrs * 12
        for col, label in (("close", "仅价格"), ("tr_index", "含股息(估算)")):
            s = m[col]
            cagr = ((s.shift(-n) / s) ** (1 / yrs) - 1).dropna()
            rows.append({"years": yrs, "basis": label, "windows": int(len(cagr)),
                         "p_loss": float((cagr < 0).mean()), "median_cagr": float(cagr.median()),
                         "worst_cagr": float(cagr.min()), "best_cagr": float(cagr.max()),
                         "worst_start": cagr.idxmin().strftime("%Y-%m")})
    return pd.DataFrame(rows)


def _first_in(mask: pd.Series, gap: int) -> pd.DatetimeIndex:
    hits = mask[mask].index
    out, last_pos = [], -10**9
    pos = pd.Series(np.arange(len(mask)), index=mask.index)
    for d in hits:
        if pos[d] - last_pos > gap:
            out.append(d)
        last_pos = pos[d]
    return pd.DatetimeIndex(out)


def event_studies(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df[df.index >= "1950-01-01"]
    events = {}
    # first close >= 10% below a record since that record was set
    since_record = (d["close"] >= d["ath"]).cumsum()
    hit = d["dd"] <= -0.10
    events["首次跌入10%回调"] = d.index[hit & (hit.astype(int).groupby(since_record).cumsum() == 1)]
    events["VIX收于35以上(60日内首次)"] = _first_in(d["vix"] >= 35, 60) if "vix" in d else pd.DatetimeIndex([])
    events["5日跌幅≥8%(60日内首次)"] = _first_in(d["ret5d"] <= -0.08, 60)
    down = (d["ret1d"] < 0).astype(int)
    streak = down.groupby((down == 0).cumsum()).cumsum()
    events["连跌7个交易日"] = _first_in(streak == 7, 10)
    mid = [y for y in range(1950, 2027) if y % 4 == 2]
    events["中期选举年11月20日前后"] = pd.DatetimeIndex(
        [d.index[d.index.get_indexer([pd.Timestamp(f"{y}-11-20")], method="bfill")[0]] for y in mid])
    detail, summary = [], []
    for name, dates in events.items():
        sub = df.loc[dates]
        for dt, r in sub.iterrows():
            detail.append({"event": name, "date": dt.strftime("%Y-%m-%d"), "close": float(r["close"]),
                           "dd": float(r["dd"]),
                           "fwd21": r["fwd21"], "fwd63": r["fwd63"], "fwd126": r["fwd126"],
                           "fwd252": r["fwd252"], "fwdmin126": r["fwdmin126"]})
        k = sub.dropna(subset=["fwd252"])
        summary.append({"event": name, "n": int(len(sub)), "n_complete": int(len(k)),
                        "since": int(sub.index[0].year) if len(sub) else None,
                        "median_1m": float(sub["fwd21"].median()), "median_3m": float(sub["fwd63"].median()),
                        "median_6m": float(sub["fwd126"].median()), "median_12m": float(k["fwd252"].median()),
                        "p_up_3m": float((sub["fwd63"].dropna() > 0).mean()),
                        "p_up_12m": float((k["fwd252"] > 0).mean()),
                        "p_further_10": float((sub["fwdmin126"].dropna() <= -0.10).mean()),
                        "worst_12m": float(k["fwd252"].min())})
    det = pd.DataFrame(detail)
    det = det.astype({c: float for c in ("fwd21", "fwd63", "fwd126", "fwd252", "fwdmin126")})
    return pd.DataFrame(summary), det


def cape_buckets(m: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x = m[["cape", "tr_index"]].copy()
    x["fwd10"] = (x["tr_index"].shift(-120) / x["tr_index"]) ** 0.1 - 1
    k = x.dropna()
    k = k[k.index >= "1928-01-01"]
    edges = [0, 10, 15, 20, 25, 30, 100]
    labels = ["<10", "10-15", "15-20", "20-25", "25-30", ">30"]
    k["bucket"] = pd.cut(k["cape"], edges, labels=labels)
    g = k.groupby("bucket", observed=True)["fwd10"].agg(["median", "min", "max", "size"]).reset_index()
    g.columns = ["cape_range", "median_10y", "worst_10y", "best_10y", "months"]
    b, a = np.polyfit(np.log(k["cape"]), k["fwd10"], 1)
    resid = k["fwd10"] - (a + b * np.log(k["cape"]))
    now = float(m["cape"].dropna().iloc[-1])
    pred = float(a + b * np.log(now))
    fit = {"cape_now": now, "pred_10y": pred, "resid_sd": float(resid.std()),
           "slope": float(b), "intercept": float(a), "r2": float(1 - resid.var() / k["fwd10"].var()),
           "pct_rank": float((m["cape"].dropna() < now).mean()), "sample": f"{k.index[0]:%Y}-{k.index[-1]:%Y}"}
    return g.astype({"cape_range": str}), fit


def seasonality(m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = m["price_ret"].dropna()
    r = r[r.index >= "1950-01-01"]
    g = r.groupby(r.index.month).agg(["mean", "median", lambda s: (s > 0).mean(), "size"]).reset_index()
    g.columns = ["month", "mean", "median", "p_up", "n"]
    yrs = m["close"].groupby(m.index.year).last().pct_change().dropna()
    yrs = yrs[(yrs.index >= 1950) & (yrs.index < m.index[-1].year)]
    names = {1: "大选后第1年", 2: "中期选举年", 3: "大选前一年", 0: "大选年"}
    cyc = yrs.groupby(yrs.index % 4).agg(["mean", "median", lambda s: (s > 0).mean(), "size"]).reset_index()
    cyc.columns = ["phase", "mean", "median", "p_up", "n"]
    cyc["name"] = cyc["phase"].map(names)
    cyc["order"] = cyc["phase"].map({1: 1, 2: 2, 3: 3, 0: 4})
    return g, cyc.sort_values("order")
