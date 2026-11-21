"""Write the dashboard's store documents (datasets + page) for the Data artifact.

    python -m snp.store today=/_blob/<id> series=/_blob/<id> history=/_blob/<id> notes=/_blob/<id> [--page | --update]

--page also writes the page and dash/meta (first setup or a page change).
--update writes merge-updates that only swap each dataset's file url and timestamp (daily refresh).

Each argument maps a data file to the blob url its upload returned. Output goes
to out/store/<collection>__<doc_id>.json plus out/store/writes.json, the list of
writes for one ArtifactData batch call (add if_version per document when the
documents already exist).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "store"

SCHED = "每个美股交易日收盘后由 Claude 定时任务重新计算并上传"
NOTES_SCHED = "每个美股交易日由 Claude 检索公开报道后整理并上传"

# dataset id -> (file, title, description)
DATASETS = {
    "signal": ("today", "今日信号与四项概率", "收盘数据、追加/上涨/减持/风险概率、历史平均与操作倾向"),
    "model_card": ("today", "模型样本外检验", "1960年以来逐年滚动样本外检验的AUC与Brier技能分数"),
    "factors": ("today", "因子面板", "各因子当前值、历史分位与同区间的历史条件概率"),
    "risk_drivers": ("today", "短期风险来源", "3个月风险模型中各状态变量对今日风险的推动"),
    "analogs": ("today", "最相似的历史时点", "与今日状态最接近的20个历史交易日及其后续涨跌"),
    "analog_dist": ("today", "相似状态的未来回报分布", "1000个相似交易日之后1/3/6/12个月的回报分位数"),
    "yearend": ("today", "到年底的概率与点位", "相似状态下到当年最后一个交易日的涨跌概率与点位分位数"),
    "markets": ("today", "跨市场行情与相关性", "25个指数、利率、汇率、商品与加密资产的涨跌及与标普的相关性"),
    "sectors": ("today", "标普11个板块", "SPDR板块ETF的涨跌与200日均线位置"),
    "price": ("series", "近3年价格与概率走势", "标普收盘、50/200日均线与各概率的滚动样本外预测"),
    "corr_roll": ("series", "60日滚动相关性", "标普与美债收益率、黄金、美元、比特币、日经的60日滚动相关系数(每周取样)"),
    "stats": ("history", "长周期统计摘要", "1928年以来的年度胜率、年化回报、回撤频率、CAPE回归与本轮回调"),
    "annual": ("history", "逐年回报", "1928年以来每年的价格回报、估算含股息回报与年内最大回撤"),
    "monthly": ("history", "月度长周期序列", "1927年以来月末收盘、含股息指数、CAPE、回撤与NBER衰退标记"),
    "bears": ("history", "历次熊市与准熊市", "从高点跌18%以上的历次下跌、恢复时间与起因"),
    "holding": ("history", "持有期亏损概率", "不同持有年限的滚动窗口亏损概率与年化回报"),
    "events": ("history", "类似情形之后", "首次回调、VIX≥35、单周急跌、连跌7日、中期选举年之后的表现"),
    "cape_buckets": ("history", "估值区间与10年回报", "按CAPE区间统计随后10年含股息年化回报"),
    "seasonality": ("history", "月度季节性", "1950年以来各月平均涨跌与上涨年份占比"),
    "cycle": ("history", "总统任期周期", "1950年以来大选周期各年份的平均涨跌"),
    "brief": ("notes", "今日要点(Claude整理)", "当日标题、展望段落与声明，由 Claude 根据公开报道撰写"),
    "news": ("notes", "今日重要新闻(Claude整理)", "当日最重要的消息、数据与解读，附原文链接"),
    "drivers": ("notes", "近期下跌驱动(Claude估计)", "本轮回调的主要驱动、证据与 Claude 估计的贡献权重"),
    "timeline": ("notes", "近期时间线", "关键事件日期及当日标普收盘"),
    "calendar": ("notes", "未来事件情景(Claude估计)", "未来重要事件、各情景概率与对标普的影响估计"),
}


def main(argv: list[str]) -> None:
    blobs = dict(a.split("=", 1) for a in argv if "=" in a)
    page = "--page" in argv
    update = "--update" in argv  # daily refresh: only point existing datasets at the new files
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    writes = []
    for ds, (f, title, desc) in DATASETS.items():
        if f not in blobs:
            continue
        doc = {
            "title": title, "description": desc,
            "source": {"kind": "file", "url": blobs[f], "name": f"{f}.json", "format": "json"},
            "updated": {"at": now, "by": "Claude"},
            "schedule": NOTES_SCHED if f == "notes" else SCHED,
            "staleAfterHours": 100,
        }
        if update:
            doc = {"source": {"url": blobs[f]}, "updated": {"at": now, "by": "Claude"}}
        path = OUT / f"datasets__{ds}.json"
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        writes.append({"op": "update" if update else "set", "collection": "datasets", "doc_id": ds,
                       "file_path": str(path)})
    if page:
        p = OUT / "files__index.html.json"
        p.write_text(json.dumps({"text": (ROOT / "dashboard" / "index.html").read_text()}, ensure_ascii=False))
        writes.append({"op": "set", "collection": "files", "doc_id": "index.html", "file_path": str(p)})
        m = OUT / "dash__meta.json"
        m.write_text(json.dumps({"title": "标普500 每日决策看板", "lang": "zh-CN"}, ensure_ascii=False))
        writes.append({"op": "set", "collection": "dash", "doc_id": "meta", "file_path": str(m)})
    (OUT / "writes.json").write_text(json.dumps(writes, ensure_ascii=False, indent=1))
    print(json.dumps(writes, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
