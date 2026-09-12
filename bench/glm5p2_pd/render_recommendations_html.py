#!/usr/bin/env python3
"""Render the TP+DPA recommendation CSV as a self-contained HTML dashboard."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TP+DPA 实测拓扑推荐</title>
<style>
:root{color-scheme:light dark;--bg:#f6f7f9;--surface:#fff;--surface2:#f0f2f5;
--text:#1d2433;--muted:#667085;--border:#d8dde6;--accent:#2563eb;
--green:#15803d;--greenbg:#dcfce7;--blue:#0369a1;--bluebg:#e0f2fe;
--amber:#a16207;--amberbg:#fef3c7;--neutral:#eef0f3}
@media(prefers-color-scheme:dark){:root{--bg:#111318;--surface:#181b22;
--surface2:#20242d;--text:#eef1f6;--muted:#9aa3b2;--border:#343a46;
--accent:#6ea8fe;--green:#86efac;--greenbg:#143923;--blue:#7dd3fc;
--bluebg:#123447;--amber:#fde68a;--amberbg:#403414;--neutral:#282d37}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI","PingFang SC",
"Microsoft YaHei",sans-serif;line-height:1.5}
main{width:min(1240px,calc(100% - 32px));margin:auto;padding:36px 0 56px}
h1{margin:0;font-size:clamp(24px,4vw,34px);letter-spacing:-.02em}
h2{margin:0;font-size:19px}.subtitle{margin:6px 0 0;color:var(--muted)}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;
margin:24px 0}.stat{padding:16px;background:var(--surface);
border:1px solid var(--border);border-radius:10px}.stat strong{display:block;
font-size:25px;line-height:1.2}.stat span{display:block;margin-top:5px;
color:var(--muted);font-size:13px}.callout{padding:16px 18px;margin-bottom:24px;
border:1px solid var(--border);border-left:4px solid var(--accent);
border-radius:8px;background:var(--surface)}.callout strong{display:block;
margin-bottom:4px}.callout p{margin:0;color:var(--muted)}
.toolbar,.toolbar-left{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.toolbar{justify-content:space-between;margin-bottom:12px}
select,.button{min-height:36px;border:1px solid var(--border);border-radius:7px;
background:var(--surface);color:var(--text);font:inherit;font-size:14px}
select{padding:6px 34px 6px 10px}.button{display:inline-flex;align-items:center;
padding:6px 12px;text-decoration:none}.button:hover{border-color:var(--accent)}
.count,footer{color:var(--muted);font-size:12px}.table-wrap{max-height:650px;
overflow:auto;background:var(--surface);border:1px solid var(--border);
border-radius:10px}table{width:100%;border-collapse:collapse;font-size:14px;
font-variant-numeric:tabular-nums}th,td{padding:11px 12px;
border-bottom:1px solid var(--border);text-align:left;white-space:nowrap}
th{position:sticky;top:0;z-index:1;background:var(--surface2);color:var(--muted);
font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.num{text-align:right}tbody tr:nth-child(even){background:var(--surface2)}
tbody tr:hover{outline:1px solid var(--border)}code{padding:2px 6px;
border-radius:5px;background:var(--surface2);font-family:ui-monospace,monospace}
.badge{display:inline-block;padding:3px 8px;border-radius:999px;font-size:12px;
font-weight:650}.tput{color:var(--green);background:var(--greenbg)}
.inv{color:var(--blue);background:var(--bluebg)}
.only{color:var(--amber);background:var(--amberbg)}
.tradeoff{background:var(--neutral)}footer{margin-top:14px}
@media(max-width:760px){.stats{grid-template-columns:repeat(2,minmax(0,1fr))}
main{width:min(100% - 20px,1240px);padding-top:22px}}
</style>
</head>
<body>
<main>
<header><h1>TP+DPA 实测拓扑推荐</h1><p class="subtitle">
每个 concurrency 内同时最大化 Interactivity 和 total tokens/s/GPU，
仅保留 Pareto 不被支配的拓扑。</p></header>
<section class="stats">
<div class="stat"><strong>57</strong><span>C≥8 候选点</span></div>
<div class="stat"><strong>28</strong><span>推荐实测点</span></div>
<div class="stat"><strong>10</strong><span>Concurrency 档位</span></div>
<div class="stat"><strong>TP+DPA</strong><span>P/D 统一并行模式</span></div>
</section>
<aside class="callout"><strong>稳定的推荐结构</strong><p>
C8–C64 保留四档：P4D4 最高吞吐/GPU、P4D8 低成本折中、1P2D
进一步提高 Interactivity、1P3D 取得最高 Interactivity。C96–C168
保留 1P1D P8D8、1P2D 和 1P3D 三档。C192 以上目前只有 2P2D 数据，
不能称为跨拓扑 Pareto 结论。</p></aside>
<section>
<div class="toolbar"><div class="toolbar-left"><h2>按 Concurrency 查看</h2>
<select id="filter" aria-label="筛选 Concurrency"><option value="all">全部</option>
<option>8</option><option>16</option><option>32</option><option>64</option>
<option>96</option><option>128</option><option>168</option><option>192</option>
<option>256</option><option>336</option></select><span class="count" id="count"></span>
</div><a class="button" href="recommended_by_concurrency.csv" download>下载推荐 CSV</a>
</div>
<div class="table-wrap"><table><thead><tr><th>Conc</th><th>推荐角色</th>
<th>P/D 拓扑</th><th class="num">GPU</th><th class="num">Interactivity</th>
<th class="num">Total tok/s/GPU</th></tr></thead><tbody id="body"></tbody>
</table></div></section>
<footer>Source: tp-dpa-all-topologies-v3 · ISL 218922 · OSL 1077 ·
throughput 口径为 input + output tokens/s/GPU。Interactivity 越高越好。</footer>
</main>
<script>
const data=__DATA__;
const labels={max_tput_per_gpu:"最高吞吐/GPU",tradeoff:"Pareto 折中",
max_interactivity:"最高 Interactivity",
"max_tput_per_gpu+max_interactivity":"唯一已测候选"};
const classes={max_tput_per_gpu:"tput",tradeoff:"tradeoff",
max_interactivity:"inv","max_tput_per_gpu+max_interactivity":"only"};
const filter=document.getElementById("filter"),body=document.getElementById("body"),
count=document.getElementById("count");
function render(){const selected=filter.value;
const visible=data.filter(row=>selected==="all"||row.concurrency===Number(selected));
body.replaceChildren(...visible.map(row=>{const tr=document.createElement("tr");
tr.innerHTML=`<td><code>C${row.concurrency}</code></td>
<td><span class="badge ${classes[row.role]}">${labels[row.role]}</span></td>
<td>${row.topology}</td><td class="num">${row.total_gpus}</td>
<td class="num">${row.interactivity_tok_s_user.toFixed(2)}</td>
<td class="num">${row.total_throughput_tps_per_gpu.toFixed(2)}</td>`;return tr}));
count.textContent=`显示 ${visible.length} 点`}
filter.addEventListener("change",render);render();
</script>
</body></html>
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recommendations_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.recommendations_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    data = [
        {
            "concurrency": int(row["concurrency"]),
            "role": row["role"],
            "topology": row["topology"],
            "total_gpus": int(row["total_gpus"]),
            "interactivity_tok_s_user": float(row["interactivity_tok_s_user"]),
            "total_throughput_tps_per_gpu": float(
                row["total_throughput_tps_per_gpu"]
            ),
        }
        for row in rows
    ]
    if len(data) != 28:
        raise SystemExit(f"expected 28 recommendations, got {len(data)}")
    args.output.write_text(
        HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False)),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
