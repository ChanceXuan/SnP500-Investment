"""Data fetchers: Yahoo Finance chart API, FRED CSV, multpl.com tables.

Every fetch is cached to data/raw/ as CSV so a failed request on a later day
falls back to the last good copy instead of breaking the build.
"""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) snp-dashboard/1.0"}


def _get(url: str, params: dict | None = None, timeout: int = 60, tries: int = 4,
         headers: dict | None = UA) -> requests.Response:
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code} for {url}")
        except requests.RequestException as e:  # network hiccup: back off and retry
            last = e
        time.sleep(2 * (i + 1))
    raise last  # type: ignore[misc]


def _cache_name(kind: str, key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_")
    return RAW / f"{kind}_{safe}.csv"


def yahoo(symbol: str, start: str = "1927-01-01") -> pd.DataFrame:
    """Daily OHLC + adjusted close for a Yahoo symbol. Index = local exchange date."""
    path = _cache_name("yahoo", symbol)
    try:
        p1 = int(pd.Timestamp(start).timestamp())
        r = _get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(symbol)}",
            params=dict(period1=p1, period2=int(time.time()) + 86400, interval="1d",
                        includeAdjustedClose="true", events="div,splits"),
        )
        res = r.json()["chart"]["result"][0]
        off = res["meta"].get("gmtoffset", 0) or 0
        idx = pd.to_datetime([t + off for t in res["timestamp"]], unit="s").normalize()
        q = res["indicators"]["quote"][0]
        df = pd.DataFrame({k: q.get(k) for k in ("open", "high", "low", "close", "volume")}, index=idx)
        adj = res["indicators"].get("adjclose")
        df["adjclose"] = adj[0]["adjclose"] if adj else df["close"]
        df = df[~df.index.duplicated(keep="last")].dropna(subset=["close"]).sort_index()
        df.index.name = "date"
        df.to_csv(path)
        return df
    except Exception as e:  # fall back to cache
        if path.exists():
            print(f"[warn] yahoo {symbol}: {e}; using cache")
            return pd.read_csv(path, index_col=0, parse_dates=True)
        raise


def fred(series: str) -> pd.Series:
    path = _cache_name("fred", series)
    try:
        # FRED's bot filter rejects browser-like user agents that are not real browsers,
        # so it gets the plain python-requests agent.
        r = _get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series},
                 timeout=90, headers=None)
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["date", series]
        df["date"] = pd.to_datetime(df["date"])
        df[series] = pd.to_numeric(df[series], errors="coerce")
        df = df.dropna().set_index("date")
        df.to_csv(path)
        return df[series]
    except Exception as e:
        if path.exists():
            print(f"[warn] fred {series}: {e}; using cache")
            return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        raise


def multpl(slug: str) -> pd.Series:
    """Monthly table from multpl.com, e.g. 'shiller-pe', 's-p-500-dividend-yield'."""
    path = _cache_name("multpl", slug)
    try:
        r = _get(f"https://www.multpl.com/{slug}/table/by-month")
        rows = re.findall(r"<tr[^>]*>\s*<td[^>]*>([^<]+)</td>\s*<td[^>]*>(.*?)</td>", r.text, re.S)
        recs = []
        for d, v in rows:
            v = re.sub(r"<[^>]+>|&#x2002;|%|,", "", v).strip()
            try:
                recs.append((pd.to_datetime(d.strip()), float(v)))
            except ValueError:
                continue
        s = pd.Series(dict(recs)).sort_index()
        s.index.name = "date"
        s.name = slug
        if len(s) < 100:
            raise RuntimeError(f"multpl {slug}: only {len(s)} rows parsed")
        s.to_csv(path)
        return s
    except Exception as e:
        if path.exists():
            print(f"[warn] multpl {slug}: {e}; using cache")
            return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        raise


# Cross-market universe: (symbol, Chinese name, group, how to measure change)
UNIVERSE = [
    ("^GSPC", "标普500", "美股", "pct"),
    ("^NDX", "纳斯达克100", "美股", "pct"),
    ("^DJI", "道琼斯工业", "美股", "pct"),
    ("^RUT", "罗素2000小盘", "美股", "pct"),
    ("RSP", "标普500等权ETF", "美股", "pct"),
    ("^SOX", "费城半导体", "美股", "pct"),
    ("^VIX", "VIX恐慌指数", "波动率", "pct"),
    ("^TNX", "美债10年收益率", "利率", "diff"),
    ("^IRX", "美债3个月收益率", "利率", "diff"),
    ("TLT", "20年+美债ETF", "利率", "pct"),
    ("HYG", "高收益债ETF", "信用", "pct"),
    ("DX-Y.NYB", "美元指数", "汇率", "pct"),
    ("JPY=X", "美元/日元", "汇率", "pct"),
    ("CNY=X", "美元/人民币", "汇率", "pct"),
    ("GC=F", "黄金", "商品", "pct"),
    ("CL=F", "WTI原油", "商品", "pct"),
    ("HG=F", "铜", "商品", "pct"),
    ("BTC-USD", "比特币", "加密", "pct"),
    ("^STOXX50E", "欧洲斯托克50", "全球股市", "pct"),
    ("^GDAXI", "德国DAX", "全球股市", "pct"),
    ("^FTSE", "英国富时100", "全球股市", "pct"),
    ("^N225", "日经225", "全球股市", "pct"),
    ("^KS11", "韩国KOSPI", "全球股市", "pct"),
    ("^HSI", "恒生指数", "全球股市", "pct"),
    ("000001.SS", "上证综指", "全球股市", "pct"),
]

SECTORS = [
    ("XLK", "信息技术"), ("XLC", "通信服务"), ("XLY", "可选消费"), ("XLF", "金融"),
    ("XLV", "医疗保健"), ("XLI", "工业"), ("XLP", "必需消费"), ("XLE", "能源"),
    ("XLU", "公用事业"), ("XLB", "原材料"), ("XLRE", "房地产"),
]

# Asian/European sessions close before the US session that shares their date,
# so their reaction to a US day shows up on the next local date.
NEXT_DAY_MARKETS = {"^N225", "^KS11", "^HSI", "000001.SS", "^STOXX50E", "^GDAXI", "^FTSE"}

FRED_SERIES = ["DGS10", "DTB3", "BAA10YM", "BAA10Y", "T10Y3M", "NFCI", "USREC", "DFII10", "T10YIE"]


def fetch_all() -> dict:
    out: dict = {"yahoo": {}, "fred": {}, "multpl": {}}
    for sym, *_ in UNIVERSE:
        out["yahoo"][sym] = yahoo(sym, start="1927-01-01" if sym == "^GSPC" else "1985-01-01")
    for sym in ("^VIX3M", "SPY", "IEF"):
        out["yahoo"][sym] = yahoo(sym, start="2000-01-01")
    for sym, _ in SECTORS:
        out["yahoo"][sym] = yahoo(sym, start="2015-01-01")
    for s in FRED_SERIES:
        out["fred"][s] = fred(s)
    for slug in ("shiller-pe", "s-p-500-dividend-yield"):
        out["multpl"][slug] = multpl(slug)
    return out


def load_cached() -> dict:
    """Rebuild the fetch_all() structure from data/raw/ without the network."""
    out: dict = {"yahoo": {}, "fred": {}, "multpl": {}}
    syms = [u[0] for u in UNIVERSE] + ["^VIX3M", "SPY", "IEF"] + [s[0] for s in SECTORS]
    for sym in syms:
        p = _cache_name("yahoo", sym)
        if p.exists():
            out["yahoo"][sym] = pd.read_csv(p, index_col=0, parse_dates=True)
    for s in FRED_SERIES:
        p = _cache_name("fred", s)
        if p.exists():
            out["fred"][s] = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
    for slug in ("shiller-pe", "s-p-500-dividend-yield"):
        p = _cache_name("multpl", slug)
        if p.exists():
            out["multpl"][slug] = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
    return out
