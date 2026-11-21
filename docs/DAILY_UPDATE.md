# 每日更新流程（给定时任务里的 Claude 执行）

看板：https://claude.ai/artifact/KnPTAoUBXus6vty4GnX5dC （Claude「Data」类型的 Artifact）
分支：`claude/laughing-euler-9237wy`

每个美股交易日收盘后执行一次。全部步骤都只读公开数据，并且只写这个看板和这个分支。

## 0. 准备

```bash
git fetch origin claude/laughing-euler-9237wy && git checkout claude/laughing-euler-9237wy && git pull --ff-only
pip install -q -r requirements.txt
```

网络需要能访问 query1.finance.yahoo.com、fred.stlouisfed.org、www.multpl.com（已加入本环境的允许列表）。

## 1. 重新计算量化部分

```bash
python -m snp.build
```

输出 `out/today.json`、`out/series.json`、`out/history.json`，并把当天一行追加到 `data/signal_history.csv`。
如果某个数据源失败，fetch 会回退到 `data/raw/` 缓存并打印 `[warn]`；新会话里没有缓存时会直接报错，此时重试一次，仍失败就在总结里说明哪个源失败，不要编造数据。

确认最后一个交易日：`python -c "import json;print(json.load(open('out/today.json'))['signal'][0]['date'])"`。若日期没有前进（美股休市），跳过第 2-4 步，只在总结里说明。

## 2. 检索新闻并改写 `data/manual/notes.json`

用 WebSearch（必要时 extended 模式）检索：当日标普500收盘与原因、主要个股/板块异动、美联储与利率、重要经济数据、政策新闻，以及未来 4-6 周的事件日历（经济数据、议息、重要财报、期权到期、假期）。

按现有结构整体改写 `data/manual/notes.json`：

- `date`（整理日期）、`data_through`（数据截至的交易日）
- `headline`：一句话概括当日市场；`outlook`：一段解读（对长期持有者意味着什么）；`disclosure` 保持不变
- `news`：6-10 条，字段 `rank, date, impact(利多/利空/中性), magnitude(1-3), title, detail, take, source, url`。`url` 必须是检索结果里真实出现过的 https 链接
- `drivers`：近期变化的 3-6 个驱动，`weight` 合计为 1，标注为 Claude 估计
- `timeline`：最近 1-2 个月的关键日期（`date, event`），build 会自动附上当日收盘
- `calendar`：未来事件，每个情景一行：`id, date, time_bj(北京时间), event, importance(1-3), scenario, prob, spx_move, note`。同一 `id` 的 `prob` 合计为 1；已过去的事件删掉，新事件补上。概率依据市场定价(CME FedWatch 等)、同类事件历史当日波动与当前 VIX 隐含日波动（VIX/√252）给出，并在 note 里写明依据

数字要与 `out/today.json` 一致（收盘、涨跌、VIX、回撤等直接从那里取）。不确定的日期标注“待确认”。

改完后：

```bash
python -m snp.build --cached     # 生成 out/notes.json，并给 timeline 附上收盘价
cp data/manual/notes.json data/notes_archive/$(date -u +%F).json
```

## 3. 上传数据并切换看板的数据源

1. 记下旧文件和版本：`ArtifactData list`，`collection: "datasets"`，记下全部 25 个数据集文档各自的 `version`，以及 `signal`、`price`、`stats`、`news` 四个文档的 `source.url`（分别指向旧的 today/series/history/notes 文件）。
2. 上传四个新文件（每个文件单独一次调用）：`Artifact` publish，`url` 为看板地址，`asset: true`，`file_path` 分别为 `out/today.json`、`out/series.json`、`out/history.json`、`out/notes.json`。记下返回的 `/_blob/<id>`。
3. 生成更新：

   ```bash
   python -m snp.store today=/_blob/<id> series=/_blob/<id> history=/_blob/<id> notes=/_blob/<id> --update
   ```

4. 用 `ArtifactData batch` 提交 `out/store/writes.json` 中的写入（全部是 `update`），每条加上第 1 步读到的 `if_version`。
5. 旧文件不再被引用后，用 `Artifact` 的 `delete` 动作（`url` + `path` 为旧 asset id）逐个删除第 1 步记下的四个旧文件。删除前确认新数据源已提交成功。

页面代码(`files/index.html`)日常不需要改。只有改了 `dashboard/index.html` 时才用 `--page` 重新写入页面（写之前先读 `files/index.html` 的 version，并读 `artifact-type/reference/one-writer.md` 的规则）。

## 4. 提交

```bash
git add data/signal_history.csv data/manual/notes.json data/notes_archive/
git commit -m "Daily update $(date -u +%F)"
git push -u origin claude/laughing-euler-9237wy
```

## 5. 总结

最后用中文简要回报：当日收盘与涨跌、四项概率(及前一日)、操作倾向、最重要的 2-3 条新闻、下一个关键事件，以及任何失败的步骤。
