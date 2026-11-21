"""Build every dashboard data file into out/.

    python -m snp.build            # fetch fresh data, rebuild out/*.json
    python -m snp.build --cached   # rebuild from data/raw/ without network

Writes four files, each {"meta": [...], "<dataset>": [rows], ...}:
  out/today.json    signal, model card, factors, analogs, cross-market, sectors
  out/series.json   3-year price + probability history, rolling correlations, playback
  out/history.json  long-run history (annual, monthly, bear markets, event studies...)
  out/notes.json    news, drivers and event calendar from data/manual/notes.json
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import factors as F
from . import fetch, history as H, markets as MK, model as M

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
MANUAL = ROOT / "data" / "manual" / "notes.json"


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if (math.isnan(v) or math.isinf(v)) else round(float(v), 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def rows(df: pd.DataFrame, date_index: bool = False) -> list[dict]:
    d = df.reset_index() if date_index else df
    return [_clean(r) for r in d.to_dict("records")]


def code_meta(*modules: str, extra: list[dict] | None = None) -> list[dict]:
    meta = [{"label": f"snp/{m}", "value": (ROOT / "snp" / m).read_text(), "type": "code", "lang": "python"}
            for m in modules]
    return meta + (extra or [])


def write(name: str, payload: dict) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(_clean(payload), ensure_ascii=False, separators=(",", ":")))


def stance(sig: dict) -> tuple[str, str, str]:
    """Turn the probabilities into a plain action for a long-term holder."""
    dd, up, risk, rb = sig["dd"], sig["p_up12"], sig["p_risk"], sig["base_risk"]
    trim, tb, cape_pct = sig["p_trim"], sig["base_trim"], sig["cape_pct"]
    if dd <= -0.10 and up >= 0.60:
        name = "分批加仓"
        if risk >= 1.3 * rb:
            plan = "短期再跌风险显著高于常态：把计划加仓资金分4-6份，每周或指数每再跌约5%投入一份，不一次性满仓。"
        else:
            plan = "短期风险接近常态：可分2-3份在未来1-2个月内投入。"
        why = "指数已较高点回撤超过10%，相似历史状态下12个月后上涨的比例较高。"
    elif dd <= -0.05:
        name = "小幅加仓 / 按计划定投"
        plan = "回撤5%-10%属于每年常见的调整，按既定定投节奏执行，可略微加码。"
        why = "回撤幅度一般，没有极端恐慌或极端估值信号。"
    elif trim >= 1.5 * tb and cape_pct >= 0.90 and dd >= -0.03:
        name = "少量减持(5%-10%)"
        plan = "仅卖出计划内的一小部分（如持仓的5%-10%）锁定收益，主体仓位继续持有。"
        why = "指数接近高点、估值处于历史前10%，且相似状态下随后出现8%以上回落的比例偏高。"
    else:
        name = "持有 / 正常定投"
        plan = "没有足够强的加仓或减持信号，维持长期持有和定投。"
        why = "各项概率接近历史常态。"
    return name, why, plan


def build(cached: bool = False) -> dict:
    raw = fetch.load_cached() if cached else fetch.fetch_all()

    df = M.build_features(raw)
    df = F.add_cross_market(df, raw)
    asof = df.index[-1]
    now = df.iloc[-1]

    # ---------------------------------------------------------- models
    fitted = {k: M.final_model(df, k) for k in M.TARGETS}
    probs = {k: float(fitted[k].predict(df.iloc[[-1]])[0]) for k in M.TARGETS}
    bases = {k: fitted[k].base for k in M.TARGETS}
    oos = {k: M.walk_forward(df, k) for k in M.TARGETS}
    card = [M.evaluate(df, oos[k], k) for k in M.TARGETS]
    dist = M.analog_distribution(df, fitted["add"])
    ana = M.analog_table(df, fitted["add"])
    rc = fitted["risk"].contributions(df.iloc[[-1]])
    risk_drivers = [{"factor": lab, "key": f, "value": float(now[f]), "contribution": float(c)}
                    for f, lab, c in zip(M.RISK_FEATS, ["距高点回撤", "波动率(对数)", "偏离200日线", "近1个月涨跌"], rc)]

    ft = F.factor_table(df)
    cape_pct = float((df["cape"].dropna() < now["cape"]).mean())

    prev = df.iloc[-2]
    week = df.iloc[-6]
    sig = {
        "date": asof, "close": now["close"], "chg_1d": now["close"] / prev["close"] - 1,
        "chg_1w": now["close"] / week["close"] - 1, "chg_ytd": now["close"] / df[df.index.year < asof.year]["close"].iloc[-1] - 1,
        "dd": now["dd"], "ath": now["ath"], "ath_date": df.index[df["close"] == now["ath"]][-1],
        "ma50": now["ma50"], "ma200": now["ma200"], "rsi14": now["rsi14"], "vix": now.get("vix"),
        "rv20": now["rv20"], "cape": now["cape"], "cape_pct": cape_pct, "y10": now["y10"],
        "p_add": probs["add"], "base_add": bases["add"],
        "p_up12": probs["up12"], "base_up12": bases["up12"],
        "p_trim": probs["trim"], "base_trim": bases["trim"],
        "p_risk": probs["risk"], "base_risk": bases["risk"],
        "p_add_prev": float(oos["add"].iloc[-2]), "p_trim_prev": float(oos["trim"].iloc[-2]),
        "p_risk_prev": float(oos["risk"].iloc[-2]), "p_up12_prev": float(oos["up12"].iloc[-2]),
        "analog_median_12m": dist[252]["median"], "analog_p10_12m": dist[252]["p10"],
        "analog_p90_12m": dist[252]["p90"], "analog_median_3m": dist[63]["median"],
        "analog_episodes": dist["episodes"],
        "streak_down": int((df["ret1d"].iloc[::-1] < 0).cumprod().sum()),
    }
    sig["stance"], sig["stance_why"], sig["stance_plan"] = stance(sig)
    sig["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # keep one row per trading day so the signal's track record stays in git
    hist_path = ROOT / "data" / "signal_history.csv"
    cols = ["date", "close", "dd", "vix", "p_add", "p_up12", "p_trim", "p_risk", "stance"]
    row = pd.DataFrame([{c: _clean(sig[c]) for c in cols}])
    if hist_path.exists():
        old = pd.read_csv(hist_path, dtype={"date": str})
        row = pd.concat([old[old["date"] != row["date"].iloc[0]], row]).sort_values("date")
    row.to_csv(hist_path, index=False, float_format="%.6g")

    analog_dist = [{"horizon": {21: "1个月", 63: "3个月", 126: "6个月", 252: "12个月"}[h], "days": h,
                    **{k: v for k, v in dist[h].items()}} for h in (21, 63, 126, 252)]
    analog_rows = [{"date": d.strftime("%Y-%m-%d"), "close": r["close"], "dd": r["dd"], "rv20": r["rv20"],
                    "ret1m": r["ret1m"], "fwd63": r["fwd63"], "fwd252": r["fwd252"],
                    "fwdmin126": r["fwdmin126"], "distance": r["distance"]} for d, r in ana.iterrows()]

    # Year-end: analog outcomes over the trading days left in the calendar year
    from pandas.tseries.holiday import USFederalHolidayCalendar
    ye = pd.Timestamp(f"{asof.year}-12-31")
    hol = [h for h in USFederalHolidayCalendar().holidays(asof, ye)
           if not (h.month == 10 or (h.month == 11 and h.day < 15))]  # NYSE trades Columbus & Veterans Day
    h_ye = max(1, len(pd.bdate_range(asof + pd.Timedelta(days=1), ye)) - len(hol))
    nb = df.loc[fitted["add"].index[fitted["add"].neighbours(df.iloc[[-1]])[0]]]
    c = df["close"]
    pos = c.index.get_indexer(nb.index)
    ok = pos + h_ye < len(c)
    r_ye = c.iloc[pos[ok] + h_ye].to_numpy() / c.iloc[pos[ok]].to_numpy() - 1
    mx_ye = np.array([c.iloc[p + 1:p + h_ye + 1].max() / c.iloc[p] - 1 for p in pos[ok]])
    lvl = lambda q: float(now["close"] * (1 + np.quantile(r_ye, q)))
    yearend = [
        {"metric": "交易日剩余", "value": h_ye, "unit": "天"},
        {"metric": "年末高于今日收盘", "value": float((r_ye > 0).mean()), "unit": "prob"},
        {"metric": "年末较今日涨≥5%", "value": float((r_ye >= 0.05).mean()), "unit": "prob"},
        {"metric": "年末较今日跌≥5%", "value": float((r_ye <= -0.05).mean()), "unit": "prob"},
        {"metric": "年底前收复历史高点", "value": float((mx_ye >= now["ath"] / now["close"] - 1).mean()), "unit": "prob"},
        {"metric": "年末点位10%分位", "value": lvl(0.1), "unit": "level"},
        {"metric": "年末点位中位数", "value": lvl(0.5), "unit": "level"},
        {"metric": "年末点位90%分位", "value": lvl(0.9), "unit": "level"},
    ]

    cm = MK.cross_market(raw)
    sec = MK.sectors(raw)

    method = (
        "追加/12个月上涨/减持概率：在1930年以来的交易日中，按回撤、波动率、偏离200日线、近1月和12-1月动量"
        "找出与今天最相似的1000天，统计其结果，并向历史平均收缩1/3。3个月再跌≥10%风险：同类状态变量的正则化"
        "逻辑回归。全部模型均做1960年起逐年滚动样本外检验，分数见模型可信度表。"
    )
    write("today.json", {
        "meta": code_meta("model.py", "factors.py", "markets.py", "build.py", extra=[
            {"label": "数据源", "value": "Yahoo Finance(指数/ETF/期货/汇率)、FRED(美债、信用利差、金融条件、衰退)、multpl.com(席勒CAPE、股息率)"},
            {"label": "方法", "value": method},
            {"label": "数据截至", "value": asof.strftime("%Y-%m-%d")},
        ]),
        "signal": [sig], "model_card": card, "factors": rows(ft), "risk_drivers": risk_drivers,
        "analogs": analog_rows, "analog_dist": analog_dist, "yearend": yearend, "markets": rows(cm), "sectors": rows(sec),
    })

    # ---------------------------------------------------------- series
    recent = df[df.index >= asof - pd.Timedelta(days=3 * 365)]
    price = pd.DataFrame({"close": recent["close"], "ma50": recent["ma50"], "ma200": recent["ma200"],
                          "p_add": oos["add"].reindex(recent.index), "p_up12": oos["up12"].reindex(recent.index),
                          "p_trim": oos["trim"].reindex(recent.index), "p_risk": oos["risk"].reindex(recent.index)})
    price.index.name = "date"
    rc_df = MK.rolling_corr(raw).iloc[::5]
    rc_df.columns = ["us10y", "vix", "gold", "dxy", "btc", "nikkei"]
    rc_df.index.name = "date"
    pb = pd.DataFrame({"close": df["close"], "p_add": oos["add"], "p_risk": oos["risk"], "dd": df["dd"]})
    pb = pb[pb.index >= "1990-01-01"].iloc[::5].dropna()
    pb.index.name = "date"
    write("series.json", {
        "meta": code_meta("model.py", "markets.py", extra=[
            {"label": "说明", "value": "概率序列全部为滚动样本外预测：每年1月只用当时已知结果的数据重新拟合，再预测当年。"}]),
        "price": rows(price, True), "corr_roll": rows(rc_df, True), "playback": rows(pb, True),
    })

    # ---------------------------------------------------------- history
    mo = H.monthly(df, raw)
    ann = H.annual(df, mo)
    bears = H.episodes(df["close"])
    cur = H.current_drawdown(df)
    hold = H.holding_periods(mo)
    ev_sum, ev_det = H.event_studies(df)
    cb, cfit = H.cape_buckets(mo)
    seas, cyc = H.seasonality(mo)
    mo_out = mo[["close", "tr_index", "cape", "dd", "recession"]].copy()
    mo_out.index.name = "date"
    up_years = ann[~ann["ytd"]]
    stats = [{
        "years": int(len(up_years)), "p_up_year": float((up_years["price_return"] > 0).mean()),
        "mean_year": float(up_years["price_return"].mean()), "median_year": float(up_years["price_return"].median()),
        "best_year": int(up_years.loc[up_years["price_return"].idxmax(), "year"]),
        "best_return": float(up_years["price_return"].max()),
        "worst_year": int(up_years.loc[up_years["price_return"].idxmin(), "year"]),
        "worst_return": float(up_years["price_return"].min()),
        "cagr_price": float((df["close"].iloc[-1] / df["close"].iloc[0]) ** (365.25 / (asof - df.index[0]).days) - 1),
        "cagr_total_est": float((mo["tr_index"].iloc[-1] / mo["tr_index"].iloc[0]) ** (365.25 / (asof - df.index[0]).days) - 1),
        "p_intra_year_dd10": float((up_years["max_drawdown"] <= -0.10).mean()),
        "median_intra_year_dd": float(up_years["max_drawdown"].median()),
        "n_bears": int((bears["depth"] <= -0.20).sum()),
        "cur_peak_date": cur["peak_date"], "cur_peak": cur["peak"], "cur_dd": cur["dd_now"],
        "cur_days": cur["days_since_peak"],
        **{f"cape_{k}": v for k, v in cfit.items()},
    }]
    write("history.json", {
        "meta": code_meta("history.py", extra=[
            {"label": "说明", "value": "标普500价格指数(1927-12-30起，Yahoo ^GSPC)；含股息回报用multpl月度股息率估算；熊市/准熊市按收盘价从高点跌18%以上开始、从低点反弹25%以上或收复前高结束；起因注释为历史常识整理。"}]),
        "stats": stats, "annual": rows(ann), "monthly": rows(mo_out, True), "bears": rows(bears),
        "holding": rows(hold), "events": rows(ev_sum), "event_detail": rows(ev_det),
        "cape_buckets": rows(cb), "seasonality": rows(seas), "cycle": rows(cyc),
    })

    # ---------------------------------------------------------- notes
    if MANUAL.exists():
        notes = json.loads(MANUAL.read_text())
        for t in notes.get("timeline", []):  # attach the index level on each dated event
            d = pd.Timestamp(t["date"])
            if d in df.index:
                t["close"], t["chg_1d"] = float(df.loc[d, "close"]), float(df.loc[d, "ret1d"])
        write("notes.json", {
            "meta": [{"label": "说明", "value": "新闻、近期驱动与事件日历由 Claude 每日检索公开报道后整理；"
                      "情景概率与影响幅度是基于市场定价(如CME FedWatch)、历史同类事件当日波动和当前VIX隐含波动的估算，不是市场报价。"},
                     {"label": "整理日期", "value": notes.get("date", "")}],
            **{k: v for k, v in notes.items() if isinstance(v, list)},
            "brief": [{k: v for k, v in notes.items() if not isinstance(v, list)}],
        })
    return {"signal": sig, "card": card, "dist": dist, "factors": ft, "cm": cm, "sec": sec,
            "bears": bears, "events": ev_sum, "stats": stats[0], "hold": hold, "cape": cfit, "cyc": cyc,
            "seas": seas, "analogs": analog_rows, "risk_drivers": risk_drivers}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true")
    a = ap.parse_args()
    res = build(cached=a.cached)
    s = res["signal"]
    print(json.dumps(_clean({k: s[k] for k in ("date", "close", "dd", "p_add", "base_add", "p_up12", "base_up12",
                                                "p_trim", "base_trim", "p_risk", "base_risk", "stance")}),
                     ensure_ascii=False, indent=1))
    for f in sorted(OUT.glob("*.json")):
        print(f.name, f.stat().st_size)
