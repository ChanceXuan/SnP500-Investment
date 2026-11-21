"""Factor table: where each factor stands today and what history says about it.

For every factor the table gives today's value, its percentile in its own
history, and the outcome base rates on the historical days whose value fell in
the same decile as today's (12-month >= +10%, 12-month up, and a >= 10% drop
within 3 months). Those conditional rates are compared with the base rate over
the same days the factor exists, so each factor reads as leaning toward adding,
toward caution, or neither.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# column, label, group, unit, what it measures, mechanism note
FACTORS = [
    ("dd", "距历史高点回撤", "趋势/位置", "pct", "收盘价相对历史最高收盘的跌幅",
     "回撤越深，未来回报的分布越宽：长期回报往往更高，但短期继续下跌的风险也更高。"),
    ("dist200", "偏离200日均线", "趋势/位置", "pct", "收盘价相对200日均线的偏离",
     "跌破200日线常被视为趋势转弱；历史上跌破后的波动明显放大。"),
    ("ret1m", "近1个月涨跌", "动量", "pct", "最近21个交易日涨跌幅",
     "短期急跌后常有反弹，但急跌本身也提示波动聚集。"),
    ("mom12_1", "12-1个月动量", "动量", "pct", "过去12个月(剔除最近1个月)涨幅",
     "中期动量为正说明大趋势仍向上，回调更可能是牛市中的调整。"),
    ("rsi14", "RSI(14)", "动量", "num", "14日相对强弱指数，<30超卖、>70超买",
     "极端超卖通常对应恐慌阶段，短线反弹概率上升。"),
    ("rv20", "20日实际波动率", "波动/情绪", "pct", "近20日收益率的年化标准差",
     "波动率高度聚集：今天的高波动最能预测未来几周的高波动。"),
    ("vix", "VIX恐慌指数", "波动/情绪", "num", "标普500期权隐含波动率(1990年起)",
     "VIX≥35的恐慌期，历史上12个月后的回报普遍较好，但过程颠簸。"),
    ("vix_ts", "VIX期限结构(VIX/VIX3M)", "波动/情绪", "ratio", ">1为倒挂，代表短期恐慌(2006年起)",
     "期限结构倒挂意味着市场在为眼前风险付高价，常出现在急跌的中后段。"),
    ("cape", "席勒市盈率CAPE", "估值", "num", "价格/10年平均实际盈利",
     "CAPE对1年回报几乎没有预测力，对10年回报有较强预测力：越高，长期回报越低。"),
    ("ey_gap", "股债性价比", "估值", "pp", "100/CAPE − 10年美债收益率(百分点)",
     "数值越低，股票相对债券越贵，长期回报的安全垫越薄。"),
    ("y10", "10年期美债收益率", "利率", "pctpt", "美国10年期国债收益率",
     "高利率抬高贴现率、压制高估值成长股。"),
    ("y10_chg3m", "10年美债3个月变化", "利率", "pp", "近63个交易日收益率变化",
     "利率快速上行往往先压估值，再影响盈利预期。"),
    ("real10", "10年期实际利率(TIPS)", "利率", "pctpt", "通胀保值国债收益率(2003年起)",
     "实际利率是成长股估值的锚，越高越不利于高估值板块。"),
    ("curve", "期限利差(10年-3个月)", "利率", "pp", "10年减3个月美债收益率",
     "倒挂(负值)常领先衰退；倒挂后重新转正的阶段也常伴随经济放缓。"),
    ("baa", "信用利差(Baa-10年)", "信用", "pp", "穆迪Baa公司债与10年美债利差",
     "信用利差是风险偏好的温度计，快速走阔往往领先股市下跌。"),
    ("baa_chg3m", "信用利差3个月变化", "信用", "pp", "Baa利差近3个月变化",
     "利差在走阔(正值)说明信用市场开始担忧。"),
    ("nfci", "芝加哥联储金融条件NFCI", "信用", "num", "负值=金融条件宽松(1971年起)",
     "金融条件收紧会通过融资成本传导到企业与估值。"),
    ("rsp_rel3m", "等权/市值加权 3个月相对", "广度", "pct", "RSP相对SPY近3个月表现(2003年起)",
     "为负说明涨跌集中在大权重股，市场广度弱。"),
    ("rut_rel3m", "小盘/大盘 3个月相对", "广度", "pct", "罗素2000相对标普500近3个月表现",
     "小盘股跑输通常反映融资环境与经济预期偏弱。"),
    ("sox_rel3m", "半导体/标普 3个月相对", "跨市场", "pct", "费城半导体相对标普500近3个月表现",
     "本轮行情的龙头。半导体相对走弱，指数就难以走强。"),
    ("hyg_rel3m", "高收益债/中期国债 3个月相对", "信用", "pct", "HYG相对IEF近3个月表现(2007年起)",
     "垃圾债跑输国债=信用风险偏好下降。"),
    ("dxy_3m", "美元指数3个月变化", "跨市场", "pct", "美元指数近63个交易日变化",
     "美元急升通常意味着全球美元流动性收紧。"),
    ("cu_au_3m", "铜金比3个月变化", "跨市场", "pct", "铜价/金价近63个交易日变化",
     "铜金比下跌反映增长预期走弱、避险需求上升。"),
]


def add_cross_market(df: pd.DataFrame, raw: dict) -> pd.DataFrame:
    y = raw["yahoo"]

    def c(sym):
        return y[sym]["close"].reindex(df.index).ffill(limit=3) if sym in y else pd.Series(np.nan, index=df.index)

    def rel(a, b, n=63):
        r = c(a) / c(b)
        return r / r.shift(n) - 1

    df = df.copy()
    df["rsp_rel3m"] = rel("RSP", "SPY")
    df["rut_rel3m"] = rel("^RUT", "^GSPC")
    df["sox_rel3m"] = rel("^SOX", "^GSPC")
    df["hyg_rel3m"] = rel("HYG", "IEF")
    df["dxy_3m"] = c("DX-Y.NYB") / c("DX-Y.NYB").shift(63) - 1
    cu_au = c("HG=F") / c("GC=F")
    df["cu_au_3m"] = cu_au / cu_au.shift(63) - 1
    return df


def factor_table(df: pd.DataFrame) -> pd.DataFrame:
    now = df.iloc[-1]
    rows = []
    for col, label, group, unit, what, note in FACTORS:
        if col not in df or pd.isna(now.get(col)):
            continue
        s = df[col].dropna()
        s = s[s.index >= "1930-01-01"]
        v = float(now[col])
        pct = float((s < v).mean())
        hist = df.loc[s.index]
        known = hist.dropna(subset=["y_add", "y_risk"])
        qs = s.quantile(np.linspace(0, 1, 11)).to_numpy()
        b = int(np.clip(np.searchsorted(qs, v, side="right") - 1, 0, 9))
        lo, hi = qs[b], qs[b + 1]
        inb = known[(known[col] >= lo) & (known[col] <= hi)]
        base_add, base_risk = known["y_add"].mean(), known["y_risk"].mean()
        p_add, p_risk = inb["y_add"].mean(), inb["y_risk"].mean()
        p_up = inb["y_up12"].mean()
        d_add, d_risk = p_add - base_add, p_risk - base_risk
        lean_long = "偏多" if d_add >= 0.05 else ("偏空" if d_add <= -0.05 else "中性")
        lean_short = "风险偏高" if d_risk >= 0.04 else ("风险偏低" if d_risk <= -0.03 else "风险正常")
        rows.append({
            "factor": label, "key": col, "group": group, "unit": unit, "value": v,
            "percentile": pct, "since": int(s.index[0].year),
            "bin_lo": float(lo), "bin_hi": float(hi),
            "p_add_bin": float(p_add), "p_add_base": float(base_add),
            "p_up12_bin": float(p_up),
            "p_risk_bin": float(p_risk), "p_risk_base": float(base_risk),
            "fwd252_bin_median": float(inb["fwd252"].median()),
            "n_years_bin": int(len({d.year for d in inb.index})),
            "lean_12m": lean_long, "lean_3m": lean_short,
            "what": what, "note": note,
        })
    return pd.DataFrame(rows)
