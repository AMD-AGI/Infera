#!/usr/bin/env python3
"""Comparison tables for tuning rounds, from the published per-round artifacts.

Usage: tune_report.py [--write FILE] ROUND/CONC [ROUND/CONC ...]
  e.g. tune_report.py --write RESULT.zh-CN.md r2/144 t1-base/144 t1-r1/144 t1-r1aff/144
Reads results/<round>/c<CONC>/: agentx_conc*.json (client), prefill_balance.txt,
decode_balance.txt, engine_pace.txt, router.session_metrics.txt, and the AIPerf export
(results/... or runtime/runs/r2/agentx-c<CONC>/ for the r2 sweep) for effective concurrency.
With --write, the block between "<!-- tune-tables:begin -->" and "<!-- tune-tables:end -->"
in FILE is replaced; otherwise the tables go to stdout.
"""

import json
import re
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
BEGIN, END = "<!-- tune-tables:begin -->", "<!-- tune-tables:end -->"


def load(spec: str) -> dict:
    name, conc = spec.split("/")
    d = KIT / "results" / name / f"c{conc}"
    j = json.loads(next(d.glob("agentx_conc*.json")).read_text())
    m = j["request_metrics"]
    lat, thr = m["latency"], m["throughput"]
    cache = (j.get("server_metrics") or {}).get("cache") or {}
    r = {"name": name, "conc": conc,
         "tps": thr["per_gpu"]["total_tput_tps"], "out": thr["per_gpu"]["output_tput_tps"],
         "ok": j["num_requests_successful"], "inp": m["tokens"]["input"]["mean"],
         "i50": 1 / lat["full_response_itl"]["p50"], "i90": 1 / lat["full_response_itl"]["p90"],
         "itl50": lat["full_response_itl"]["p50"] * 1000, "itl90": lat["full_response_itl"]["p90"] * 1000,
         "t50": lat["ttft"]["p50"], "t90": lat["ttft"]["p90"], "tmean": lat["ttft"]["mean"],
         "ghit": cache.get("gpu_cache_hit_rate"), "chit": cache.get("cpu_cache_hit_rate"),
         "theo": m["cache"]["theoretical_cache_hit_rate"]}
    export = d / "profile_export_aiperf.json"
    if not export.exists():
        export = KIT / "runtime" / "runs" / "r2" / f"agentx-c{conc}" / "aiperf_artifacts" / "profile_export_aiperf.json"
    if export.exists():
        e = json.loads(export.read_text())
        r["eff"] = e["effective_concurrency"]["avg"]
        r["eff_d"] = e["effective_decode_concurrency"]["avg"]
        r["eff_p"] = e["effective_prefill_concurrency"]["avg"]
    pb = d / "prefill_balance.txt"
    r["prefill"] = {}
    if pb.exists():
        for line in pb.read_text().splitlines():
            f = line.split()
            if f and f[0].startswith("prefill-"):
                r["prefill"][f[0]] = {"new_tps": float(f[7]), "cache": float(f[4].rstrip("%")),
                                      "queue": float(f[5]), "boot": float(f[6])}
            elif f and f[0] == "all":
                r["p_cache"], r["p_new_tps"] = float(f[3].rstrip("%")), float(f[4])
    db = d / "decode_balance.txt"
    r["decode"] = {}
    if db.exists():
        for line in db.read_text().splitlines():
            f = line.split()
            if f and (f[0].isdigit() or f[0] == "all"):
                r["decode"][f[0]] = {"run": float(f[2]), "use": float(f[3]), "hot": float(f[4].rstrip("%")),
                                     "pre": float(f[5]), "xfer": float(f[6])}
    ep = d / "engine_pace.txt"
    r["pace"] = {}
    if ep.exists():
        lines = ep.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.startswith("decode ms/step"):
                cols = [int(x) for x in lines[i + 1].split()]
                vals = re.findall(r"(-|[0-9.]+\[\s*\d+\])", lines[i + 2])
                for c, v in zip(cols, vals):
                    r["pace"][c] = None if v == "-" else float(v.split("[")[0])
            m2 = re.match(r"\s+(prefill-\d)\s+(\d+) /", line)
            if m2:
                r["pace"][m2.group(1)] = int(m2.group(2))
    sm = d / "router.session_metrics.txt"
    if sm.exists():
        t = sm.read_text()
        h = re.search(r'session_hits_total\{role="prefill"\} (\d+)', t)
        s = re.search(r'session_selected_total\{role="prefill"\} (\d+)', t)
        r["sess"] = (int(h.group(1)), int(s.group(1))) if h and s else None
    return r


def fmt(v, spec="{:.1f}", missing="—"):
    return missing if v is None else spec.format(v)


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out)


def render(rs):
    label = [f"{r['name']} C{r['conc']}" for r in rs]
    parts = ["**A. 客户端结果**（AgentX；interactivity = 1/ITL 分位数）", table(
        ["轮次", "tok/s/GPU", "输出 tok/s/GPU", "完成请求", "平均输入", "P50 / P90 intv", "ITL p50 / p90 (ms)",
         "TTFT p50 / p90 / 均值 (s)", "GPU / host 命中（理论）"],
        [[l, f"{r['tps']:,.0f}", f"{r['out']:.1f}", f"{r['ok']:,}", f"{r['inp'] / 1000:.1f}k",
          f"{r['i50']:.1f} / {r['i90']:.1f}", f"{r['itl50']:.1f} / {r['itl90']:.1f}",
          f"{r['t50']:.1f} / {r['t90']:.1f} / {r['tmean']:.1f}",
          f"{fmt(r['ghit'], '{:.1%}')} / {fmt(r['chit'], '{:.1%}')}（{r['theo']:.1%}）"] for l, r in zip(label, rs)])]
    rows = []
    for l, r in zip(label, rs):
        p = list(r["prefill"].values())
        q = sum(x["queue"] for x in p) / len(p) if p else None
        a = r["decode"].get("all", {})
        rows.append([l, fmt(r.get("p_cache"), "{:.1f}%"), fmt(r.get("p_new_tps"), "{:,.0f}"), fmt(q, "{:.2f}"),
                     fmt(r.get("eff_p")), fmt(r.get("eff_d")), fmt(a.get("run"), "{:.2f}"), fmt(a.get("use"), "{:.2f}"),
                     fmt(a.get("hot"), "{:.1f}%"), fmt(a.get("pre"), "{:.2f}"), fmt(a.get("xfer"), "{:.2f}")])
    parts += ["**B. 瓶颈链**（服务端为测量窗口均值；首 token 前在途 / decode 有效并发为 AIPerf 的 effective prefill / decode concurrency）",
              table(["轮次", "prefill 缓存份额", "prefill 新算 tok/s", "prefill 排队（实例均值）", "首 token 前在途",
                     "decode 有效并发", "decode 每 rank 在跑", "decode KV 使用率", "KV>90% 时间", "等预分配 / rank",
                     "等 KV 传输 / rank"], rows)]
    rows = []
    for l, r in zip(label, rs):
        cells = [(lambda x: f"{x['new_tps'] / 1000:.1f}k / {x['cache']:.1f}% / {x['queue']:.1f}" if x else "—")(
            r["prefill"].get(f"prefill-{i}")) for i in range(3)]
        rows.append([l] + cells)
    parts += ["**C. prefill 实例均衡**（新算 tok/s / 缓存份额 / 排队）", table(["轮次", "prefill-0（135）", "prefill-1（137）", "prefill-2（137）"], rows)]
    rows = []
    for l, r in zip(label, rs):
        ranks = [v for k, v in r["decode"].items() if k != "all"]
        if not ranks:
            rows.append([l] + ["—"] * 5)
            continue
        hot = max(range(len(ranks)), key=lambda i: ranks[i]["pre"])
        rows.append([l, f"{min(x['run'] for x in ranks):.2f} – {max(x['run'] for x in ranks):.2f}",
                     f"{min(x['use'] for x in ranks):.2f} – {max(x['use'] for x in ranks):.2f}",
                     f"{min(x['hot'] for x in ranks):.1f}% – {max(x['hot'] for x in ranks):.1f}%",
                     f"DP{hot} {ranks[hot]['pre']:.2f}",
                     f"{min(x['xfer'] for x in ranks):.2f} – {max(x['xfer'] for x in ranks):.2f}"])
    parts += ["**D. decode rank 均衡**（8 个 DP rank 的最小 – 最大）",
              table(["轮次", "在跑请求", "KV 使用率", "KV>90% 时间", "等预分配最多的 rank", "等 KV 传输"], rows)]
    rows = []
    for l, r in zip(label, rs):
        pc = r["pace"]
        rows.append([l] + [fmt(pc.get(c)) for c in (6, 8, 10, 12)] +
                    [" / ".join(str(pc.get(f"prefill-{i}", "—")) for i in range(3))] +
                    [f"{r['sess'][0]:,} / {r['sess'][1]:,}" if r.get("sess") else "—"])
    parts += ["**E. 引擎节奏（判断栈是否漂移）与 router 会话**：decode 每步 ms 按同一 3 s 内最忙 rank 的在跑数分组；"
              "prefill 为排队时满 8192-token chunk 的耗时中位数 (ms)",
              table(["轮次", "decode @6", "@8", "@10", "@12", "prefill-0 / 1 / 2", "会话命中 / 新绑定"], rows)]
    return "\n\n".join(parts) + "\n"


def main() -> int:
    args = sys.argv[1:]
    target = None
    if args[:1] == ["--write"]:
        target, args = Path(args[1]), args[2:]
    text = render([load(a) for a in args])
    if target is None:
        print(text)
        return 0
    if not target.is_absolute():
        target = KIT / target
    doc = target.read_text()
    i, j = doc.index(BEGIN), doc.index(END)
    target.write_text(doc[:i + len(BEGIN)] + "\n\n" + text + "\n" + doc[j:])
    print(f"updated {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
