"""Probability models for the S&P 500 dashboard.

What a long-term holder decides, and how each probability is defined:

* ADD   P(12个月后涨幅 >= +10%)  -- buying here produced an above-average year.
* UP12  P(12个月后高于今天)
* TRIM  P(today's close is within +2% of the highest close of the next 126
         trading days AND the index later falls >= 8% below it) -- a slice sold
         here could have been bought back clearly cheaper.
* RISK  P(the index falls >= 10% below today's close within 63 trading days)

ADD / UP12 / TRIM come from historical analogs: the 1,000 most similar days
since 1930 in a space of price-only state variables (drawdown, volatility,
trend, momentum), shrunk one third toward the base rate. RISK comes from a
regularised logistic regression on the same state variables.

Every model is tested walk-forward: each January since 1960 it is refit only on
days whose outcome was already known, then predicts that calendar year. Those
out-of-sample scores (AUC, Brier skill) are published next to the numbers.
Macro and valuation factors are not in these models because their link to
12-month returns flipped sign between eras; they are reported as conditional
base rates instead (see factor_table).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRAIN_START = "1930-01-01"
TEST_FROM = 1960
K = 1000          # analog neighbours
SHRINK = 0.5      # prior weight = SHRINK * K pseudo-days at the base rate

STATE = ["dd", "lrv", "dist200", "ret1m", "mom12_1"]
RISK_FEATS = ["dd", "lrv", "dist200", "ret1m"]

TARGETS = {
    "add": dict(col="y_add", horizon=252, kind="knn", label="追加概率",
                definition="12个月后涨幅≥10%的概率"),
    "up12": dict(col="y_up12", horizon=252, kind="knn", label="12个月上涨概率",
                 definition="12个月后指数高于今日收盘的概率"),
    "trim": dict(col="y_trim", horizon=126, kind="knn", label="减持概率",
                 definition="今日接近未来半年高点(未来最高不超过+2%)且随后跌≥8%的概率"),
    "risk": dict(col="y_risk", horizon=63, kind="logit", label="3个月再跌≥10%风险",
                 definition="未来63个交易日内最低收盘比今日低10%以上的概率"),
}


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def _monthly_available(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Monthly averages dated the 1st are only known once that month is over.

    Rows dated on another day (multpl's live latest value) are known that day.
    """
    s = s.dropna().copy()
    s.index = pd.DatetimeIndex([d + pd.offsets.MonthBegin(1) if d.day == 1 else d for d in s.index])
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)


def build_features(raw: dict) -> pd.DataFrame:
    spx = raw["yahoo"]["^GSPC"]["close"].astype(float)
    df = pd.DataFrame(index=spx.index)
    df["close"] = spx
    lr = np.log(spx).diff()
    df["ret1d"] = spx.pct_change()
    df["ma50"] = spx.rolling(50).mean()
    df["ma200"] = spx.rolling(200).mean()
    df["dist200"] = np.log(spx / df["ma200"])
    df["mom12_1"] = np.log(spx.shift(21) / spx.shift(252))
    df["ret1m"] = np.log(spx / spx.shift(21))
    df["ret5d"] = spx / spx.shift(5) - 1
    df["ath"] = spx.cummax()
    df["dd"] = spx / df["ath"] - 1
    df["rsi14"] = rsi(spx)
    df["rv20"] = lr.rolling(20).std() * np.sqrt(252)
    df["lrv"] = np.log(df["rv20"])

    f, y = raw["fred"], raw["yahoo"]

    def daily(fred_s: pd.Series, yahoo_sym: str | None = None) -> pd.Series:
        # FRED posts with a one-day lag; Yahoo's CBOE yield index fills the newest days.
        s = fred_s.copy()
        if yahoo_sym and yahoo_sym in y:
            s = pd.concat([s, y[yahoo_sym]["close"][y[yahoo_sym].index > s.index[-1]]])
        return s.reindex(s.index.union(df.index)).ffill().reindex(df.index)

    df["y10"] = daily(f["DGS10"], "^TNX")
    df["tb3"] = daily(f["DTB3"], "^IRX")
    df["curve"] = df["y10"] - df["tb3"]
    df["y10_chg3m"] = df["y10"] - df["y10"].shift(63)
    df["cape"] = _monthly_available(raw["multpl"]["shiller-pe"], df.index)
    df["ey_gap"] = 100 / df["cape"] - df["y10"]
    df["baa"] = daily(f["BAA10Y"]).where(df.index >= f["BAA10Y"].index[0],
                                          _monthly_available(f["BAA10YM"], df.index))
    df["baa_chg3m"] = df["baa"] - df["baa"].shift(63)
    df["real10"] = daily(f["DFII10"]).where(df.index >= f["DFII10"].index[0])
    df["nfci"] = daily(f["NFCI"]).where(df.index >= f["NFCI"].index[0])
    if "^VIX" in y:
        df["vix"] = y["^VIX"]["close"].reindex(df.index)
    if "^VIX3M" in y:
        df["vix_ts"] = df["vix"] / y["^VIX3M"]["close"].reindex(df.index)

    # forward-looking outcomes (NaN where the future is not known yet)
    for h in (5, 21, 63, 126, 252, 756):
        df[f"fwd{h}"] = spx.shift(-h) / spx - 1
    rev = spx[::-1]
    for h in (63, 126):
        df[f"fwdmin{h}"] = rev.rolling(h, min_periods=h).min()[::-1].shift(-1) / spx - 1
        df[f"fwdmax{h}"] = rev.rolling(h, min_periods=h).max()[::-1].shift(-1) / spx - 1

    def lab(cond: pd.Series, known: pd.Series) -> np.ndarray:
        return np.where(known, cond.astype(float), np.nan)

    df["y_add"] = lab(df["fwd252"] >= 0.10, df["fwd252"].notna())
    df["y_up12"] = lab(df["fwd252"] > 0, df["fwd252"].notna())
    df["y_trim"] = lab((df["fwdmax126"] <= 0.02) & (df["fwdmin126"] <= -0.08), df["fwdmin126"].notna())
    df["y_risk"] = lab(df["fwdmin63"] <= -0.10, df["fwdmin63"].notna())
    return df


# ------------------------------------------------------------------ models --

def fit_logit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n, p = X.shape
    Xb = np.c_[np.ones(n), X]
    w = np.zeros(p + 1)
    w[0] = np.log(y.mean() / (1 - y.mean()))
    R = lam * np.eye(p + 1)
    R[0, 0] = 0.0
    for _ in range(100):
        pr = 1 / (1 + np.exp(-(Xb @ w)))
        g = Xb.T @ (pr - y) + R @ w
        H = (Xb * (pr * (1 - pr))[:, None]).T @ Xb + R
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w


class Fitted:
    """A model fitted on `train`, able to score new rows."""

    def __init__(self, train: pd.DataFrame, col: str, kind: str):
        feats = STATE if kind == "knn" else RISK_FEATS
        d = train.dropna(subset=feats + [col])
        self.kind, self.feats, self.col = kind, feats, col
        X = d[feats].to_numpy()
        self.mu, self.sd = X.mean(0), X.std(0)
        self.Z = (X - self.mu) / self.sd
        self.y = d[col].to_numpy()
        self.base = float(self.y.mean())
        self.index = d.index
        if kind == "logit":
            sub = slice(None, None, 5)  # weekly rows: less overlap between labels
            self.w = fit_logit(self.Z[sub], self.y[sub], lam=0.02 * len(self.y[sub]))

    def _zt(self, rows: pd.DataFrame) -> np.ndarray:
        return (rows[self.feats].to_numpy() - self.mu) / self.sd

    def neighbours(self, rows: pd.DataFrame) -> np.ndarray:
        T = self._zt(rows)
        D = ((T[:, None, :] - self.Z[None, :, :]) ** 2).sum(-1)
        k = min(K, len(self.y) - 1)
        return np.argpartition(D, k, axis=1)[:, :k]

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        if self.kind == "logit":
            return 1 / (1 + np.exp(-(self.w[0] + self._zt(rows) @ self.w[1:])))
        idx = self.neighbours(rows)
        k = idx.shape[1]
        a = SHRINK * k
        return (self.y[idx].mean(1) * k + self.base * a) / (k + a)

    def contributions(self, row: pd.DataFrame) -> np.ndarray:
        return self._zt(row)[0] * self.w[1:]


def _labelled_before(df: pd.DataFrame, cut: pd.Timestamp, horizon: int) -> pd.DataFrame:
    hist = df[(df.index >= TRAIN_START) & (df.index < cut)]
    return hist.iloc[: max(0, len(hist) - horizon)]


def walk_forward(df: pd.DataFrame, key: str) -> pd.Series:
    t = TARGETS[key]
    preds = []
    for yr in range(TEST_FROM, df.index[-1].year + 1):
        cut = pd.Timestamp(f"{yr}-01-01")
        m = Fitted(_labelled_before(df, cut, t["horizon"]), t["col"], t["kind"])
        feats = m.feats
        test = df[(df.index >= cut) & (df.index < pd.Timestamp(f"{yr + 1}-01-01"))].dropna(subset=feats)
        if len(test):
            preds.append(pd.Series(m.predict(test), index=test.index))
    return pd.concat(preds)


def final_model(df: pd.DataFrame, key: str) -> Fitted:
    t = TARGETS[key]
    return Fitted(_labelled_before(df, df.index[-1] + pd.Timedelta(days=1), 0), t["col"], t["kind"])


def auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(df: pd.DataFrame, oos: pd.Series, key: str) -> dict:
    col = TARGETS[key]["col"]
    d = pd.DataFrame({"p": oos, "y": df[col].reindex(oos.index)}).dropna().iloc[::5]
    y, p = d["y"].to_numpy(), d["p"].to_numpy()
    lab = df[col].dropna()
    clim = lab[lab.index >= TRAIN_START].expanding().mean().reindex(d.index, method="ffill").to_numpy()
    brier, ref = float(np.mean((p - y) ** 2)), float(np.mean((clim - y) ** 2))
    eras = {}
    for a, b in (("1960", "1989"), ("1990", "2007"), ("2008", "2026")):
        e = d.loc[a:b]
        if len(e) > 50 and e["y"].nunique() == 2:
            eras[f"{a}-{b}"] = auc(e["y"].to_numpy(), e["p"].to_numpy())
    s = auc(y, p)
    skill = 1 - brier / ref
    verdict = ("有效" if s >= 0.62 and skill > 0.02 else
               "较弱" if s >= 0.53 else "接近随机")
    return {"key": key, "label": TARGETS[key]["label"], "definition": TARGETS[key]["definition"],
            "n_weeks": int(len(d)), "test_start": d.index[0].strftime("%Y-%m-%d"),
            "test_end": d.index[-1].strftime("%Y-%m-%d"), "auc": s, "brier": brier,
            "brier_climatology": ref, "brier_skill": skill, "realized_rate": float(y.mean()),
            "auc_1960_1989": eras.get("1960-1989"), "auc_1990_2007": eras.get("1990-2007"),
            "auc_2008_2026": eras.get("2008-2026"), "verdict": verdict}


def analog_table(df: pd.DataFrame, m: Fitted, n: int = 20, gap: int = 63) -> pd.DataFrame:
    """The closest historical days, de-clustered so each episode appears once."""
    T = m._zt(df.iloc[[-1]])
    dist = np.sqrt(((m.Z - T) ** 2).sum(1))
    order = np.argsort(dist)
    pos = pd.Series(np.arange(len(df)), index=df.index)
    picked: list[int] = []
    for i in order:
        d = m.index[i]
        if df.loc[d, ["fwd252"]].isna().any():
            continue
        if all(abs(pos[d] - pos[m.index[j]]) > gap for j in picked):
            picked.append(i)
        if len(picked) >= n:
            break
    rows = df.loc[m.index[picked]].copy()
    rows["distance"] = dist[picked]
    return rows


def analog_distribution(df: pd.DataFrame, m: Fitted) -> dict:
    idx = m.neighbours(df.iloc[[-1]])[0]
    nb = df.loc[m.index[idx]]
    out = {}
    for h in (21, 63, 126, 252):
        r = nb[f"fwd{h}"].dropna()
        out[h] = {"p10": r.quantile(0.1), "p25": r.quantile(0.25), "median": r.median(),
                  "p75": r.quantile(0.75), "p90": r.quantile(0.9), "p_up": (r > 0).mean(), "n": len(r)}
    years = sorted({d.year for d in nb.index})
    out["episodes"] = len(years)
    out["years"] = years
    return out
