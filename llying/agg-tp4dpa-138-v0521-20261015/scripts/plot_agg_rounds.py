#!/usr/bin/env python3
"""Per-experiment table and figures for the aggregated (mixed) rounds.

Table (Markdown + CSV): one row per experiment with its effective configuration and measured data.
The configuration is resolved the way the launch scripts do it (config/config.agg.sh with
config/round.<ROUND>.sh sourced) and cross-checked against the published round.env; client and
engine data come from agg_report.py. B300 rows are the InferenceX Dynamo + SGLang points
(../inferencex-glm52-b300-dynamo-agg-20261015/).

Figures (PNG + SVG): the scored rounds side by side with B300 C48; throughput and latency against
concurrency with the B300 curve; the GPU time breakdown of the a6b profiling (results/prof-a6b/).

The table needs only the standard library (Python >= 3.12 for agg_report.py) and also writes
agg_rounds.json; the figures need matplotlib and can be drawn from that JSON alone.
Usage: plot_agg_rounds.py [--table-only | --plot-only] [--output DIR]
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
RESULTS = KIT / "results"
B300 = KIT.parent / "inferencex-glm52-b300-dynamo-agg-20261015/evidence/api-glm52-b300-rows.json"
PROFILE = RESULTS / "prof-a6b"

# (round, published variant or None, concurrency, status). Rounds without a scored window keep a row.
EXPERIMENTS = [
    ("a1", None, None, "参数校验后停掉（按要求先不开 HiCache 调通流程）"),
    ("a2", "a2-debug2", 16, "调通流程，600 s，不计分"),
    ("a3", "a3-wt", 48, ""),
    ("a3", "a3-wt", 64, ""),
    ("a4", "a4-wb-m092", 64, ""),
    ("a5", "a5-wb-m092-dly20", 64, ""),
    ("a6", None, 64, "预热中 DP2 报 HSA_STATUS_ERROR_OUT_OF_RESOURCES 后挂起"),
    ("a6b", "a6b-wb-m090-dly120", 64, "TP4 最好"),
    ("a7", None, 64, "负载下 decode 检查 32/32 通过；C64 预热 2 分钟 HSA_STATUS_ERROR_OUT_OF_RESOURCES 崩溃"),
    ("a7b", None, 64, "同 a7"),
    ("a7c", "a7c-coord-ck2k-m085-nsr0", 64, ""),
    ("a7d", "a7d-coord-ck8k-m085-nsr0", 64, ""),
    ("t8a", "t8a-wb-m092", 128, "预热中 HIP OOM（MoE 第二阶段缓冲 6.75 GiB）"),
    ("t8b", "t8b-wb-m092-ck4k", 128, "单个 tokenizer 卡住 worker，AgentX 预热中止"),
    ("t8c", "t8c-wb-m092-ck4k-tok4", 128, ""),
    ("t8d", None, 128, "未跑"),
    ("b1", "b1-a3-piggyback", 48, "a3 + delayer piggyback，140"),
    ("b0", "b0-a3-140", 48, "无效：计分窗口内同节点有他人作业（FAILED.txt）"),
    ("b2", "b2-a3-piggyback-138", 48, "b1 的设置放到 a3 的节点上"),
    ("b0r", "b0r-a3-140", 48, "b0 重跑（140 空闲后）"),
    ("b3", None, 48, "a3 去掉 DP attention（纯 TP4），chunk 8192：预热中止（prefill 只有 DP4 的 1/4）"),
    ("b3b", "b3b-a3-tp4-nodpa-ck32k", 48, "a3 去掉 DP attention（纯 TP4），chunk 32768"),
]

CONFIG_VARS = [
    "GPUS_mixed_0", "MIXED_TP", "MIXED_DP", "MIXED_MEM_FRACTION", "MIXED_CHUNK_SIZE",
    "MIXED_MAX_PREFILL_TOKENS", "MIXED_MAX_RUNNING", "MIXED_PREFILL_DELAYER",
    "MIXED_PREFILL_DELAYER_MAX_PASSES", "MIXED_HICACHE", "MIXED_HICACHE_SIZE",
    "MIXED_HICACHE_WRITE_POLICY", "MIXED_TOKENIZER_WORKERS", "MIXED_EXTRA_ENV", "HSA_NO_SCRATCH_RECLAIM",
    "AGG_NODE",
]
# Keys whose round.env value must equal the resolved configuration.
CHECKED = ["MIXED_TP", "MIXED_DP", "MIXED_MEM_FRACTION", "MIXED_CHUNK_SIZE", "MIXED_MAX_PREFILL_TOKENS",
           "MIXED_PREFILL_DELAYER_MAX_PASSES", "MIXED_HICACHE", "MIXED_HICACHE_SIZE",
           "MIXED_HICACHE_WRITE_POLICY", "MIXED_TOKENIZER_WORKERS", "MIXED_EXTRA_ENV"]

# B300 recipe agg-tp4-variants@caa2199e (InferenceX README §2); delayer passes unset = SGLang default.
B300_CONFIG = {
    24: {"chunk": 65536, "mem": "0.85", "hicache": "write_back 295 GB", "delayer": "默认"},
    32: {"chunk": 32768, "mem": "0.83", "hicache": "write_back 270 GB", "delayer": "默认"},
    48: {"chunk": 32768, "mem": "0.85", "hicache": "write_back 270 GB", "delayer": "60"},
}


def resolve(round_: str) -> dict:
    script = (f'ROUND={round_}; source "{KIT}/config/config.agg.sh" >/dev/null 2>&1; '
              f'for v in {" ".join(CONFIG_VARS)}; do printf "%s=%s\\n" "$v" "${{!v-}}"; done')
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    return dict(line.split("=", 1) for line in out.splitlines())


def config_fields(cfg: dict) -> dict:
    dp = int(cfg["MIXED_DP"])
    hicache = ("off" if cfg["MIXED_HICACHE"] == "0"
               else f"{cfg['MIXED_HICACHE_WRITE_POLICY']} {cfg['MIXED_HICACHE_SIZE']} GB")
    delayer = "off" if cfg["MIXED_PREFILL_DELAYER"] == "0" else cfg["MIXED_PREFILL_DELAYER_MAX_PASSES"]
    return {
        "GPUs": len(cfg["GPUS_mixed_0"].split(",")),
        "Parallel": f"TP{cfg['MIXED_TP']}+DPA{dp}",
        "HiCachePerRank": hicache,
        "MemFraction": cfg["MIXED_MEM_FRACTION"],
        "ChunkPerRank": int(cfg["MIXED_CHUNK_SIZE"]) // dp,
        "MaxPrefillTokens": int(cfg["MIXED_MAX_PREFILL_TOKENS"]),
        "DelayerMaxPasses": delayer,
        "TokenizerWorkers": int(cfg["MIXED_TOKENIZER_WORKERS"]),
        "Coordination": "on" if "SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION=1" in cfg["MIXED_EXTRA_ENV"] else "off",
        "Piggyback": "on" if "SGLANG_PREFILL_DELAYER_PIGGYBACK=1" in cfg["MIXED_EXTRA_ENV"] else "off",
        "HsaNoScratchReclaim": cfg["HSA_NO_SCRATCH_RECLAIM"],
        "Node": cfg["AGG_NODE"].rsplit("-", 1)[-1],
    }


def usage_hit(d: Path) -> float:
    for values in csv.reader((d / "profile_export_aiperf.csv").read_text().splitlines()):
        if values and values[0] == "Overall Usage Prompt Cache Read % (%)":
            return float(values[1])
    raise ValueError(f"{d}: no usage cache hit in the AIPerf export")


def measured(d: Path) -> dict:
    sys.path.insert(0, str(KIT / "scripts"))
    from agg_report import client, engine
    j = json.loads(next(d.glob("agentx_conc*.json")).read_text())
    m = j["request_metrics"]
    lat, tok = m["latency"], m["tokens"]
    c = client(d)
    row = {
        "TokPerSecPerGPU": c["tps"], "OutputTokPerSecPerGPU": c["out"],
        "Requests": c["ok"], "ReqPerSec": c["ok"] / m["throughput"]["duration_seconds"],
        "MeanInputK": c["inp"] / 1000, "MeanOutput": tok["output_actual"]["mean"],
        "ITLp50Ms": c["itl50"], "ITLp90Ms": c["itl90"], "TTFTp50S": c["t50"], "TTFTp90S": c["t90"],
        "IntvtyP50": lat["intvty"]["p50"], "IntvtyP90": lat["intvty"]["p90"],
        "TTFTMeanS": lat["ttft"]["mean"], "E2EMeanS": lat["e2el"]["mean"],
        "DecodeMsPerToken": 1000 * (lat["e2el"]["mean"] - lat["ttft"]["mean"]) / tok["output_actual"]["mean"],
        "UsageHitPct": usage_hit(d),
        "GPUHitPct": 100 * c["ghit"] if c["ghit"] is not None else None,
        "HostHitPct": 100 * c["chit"] if c["chit"] is not None else None,
        "TheoreticalHitPct": 100 * c["theo"],
        "EffConc": c.get("eff"), "EffConcBeforeFirstToken": c.get("eff_p"),
    }
    e = engine(d)
    if e:
        g = e["gauges"]
        run = [v for r, v in g.get("num_running_reqs", {}).items() if r != "-"]
        queue = [v for r, v in g.get("num_queue_reqs", {}).items() if r != "-"]
        row.update({"RunningPerRankMean": sum(run) / len(run), "RunningPerRankMin": min(run),
                    "RunningPerRankMax": max(run), "QueuePerRankMean": sum(queue) / len(queue),
                    "QueuePerRankMin": min(queue), "QueuePerRankMax": max(queue),
                    "EngineGenTokPerSec": e["gen_tps"]})
    return row


def load_rows() -> list[dict]:
    rows = []
    for round_, variant, conc, status in EXPERIMENTS:
        cfg = resolve(round_)
        row = {"Series": "MI355X", "Round": round_, "Concurrency": conc, "Status": status, **config_fields(cfg)}
        d = RESULTS / variant / f"c{conc}" if variant else None
        if d and (d / "round.env").exists():
            env = dict(x.split("=", 1) for x in (d / "round.env").read_text().splitlines() if "=" in x)
            for key in CHECKED:
                if key in env and env[key] != cfg[key]:
                    raise ValueError(f"{d}: round.env {key}={env[key]} differs from resolved {cfg[key]}")
        row["Scored"] = bool(d and not (d / "FAILED.txt").exists() and any(d.glob("agentx_conc*.json"))
                             and round_ != "a2")
        if d and not (d / "FAILED.txt").exists() and any(d.glob("agentx_conc*.json")):
            row.update(measured(d))
            row["Source"] = str(d.relative_to(KIT))
        rows.append(row)
    for r in json.loads(B300.read_text())["rows"]:
        if r["framework"] != "dynamo-sglang" or not r["decode_dp_attention"]:
            continue
        m, conc = r["metrics"], r["conc"]
        b = B300_CONFIG[conc]
        rows.append({
            "Series": "B300", "Round": "B300 Dynamo", "Concurrency": conc, "Status": "InferenceX 参考",
            "Scored": True, "GPUs": r["num_decode_gpu"], "Parallel": "TP4+DPA4",
            "HiCachePerRank": b["hicache"], "MemFraction": b["mem"], "ChunkPerRank": b["chunk"] // 4,
            "MaxPrefillTokens": 8192, "DelayerMaxPasses": b["delayer"], "TokenizerWorkers": 1,
            "Coordination": "off", "Piggyback": "off", "HsaNoScratchReclaim": "", "Node": "B300",
            "TokPerSecPerGPU": m["tput_per_gpu"], "OutputTokPerSecPerGPU": m["output_tput_per_gpu"],
            "Requests": m["total_requests_completed"],
            "ReqPerSec": m["total_requests_completed"] / m["duration_seconds"],
            "MeanInputK": m["mean_input_tokens"] / 1000, "MeanOutput": m["mean_output_tokens_actual"],
            "ITLp50Ms": 1000 * m["median_full_response_itl"], "ITLp90Ms": 1000 * m["p90_full_response_itl"],
            "TTFTp50S": m["median_ttft"], "TTFTp90S": m["p90_ttft"], "TTFTMeanS": m["mean_ttft"],
            "IntvtyP50": m["median_intvty"], "IntvtyP90": m["p90_intvty"],
            "E2EMeanS": m["mean_e2el"],
            "DecodeMsPerToken": 1000 * (m["mean_e2el"] - m["mean_ttft"]) / m["mean_output_tokens_actual"],
            "UsageHitPct": 100 * (m["server_gpu_cache_hit_rate"] + m["server_cpu_cache_hit_rate"]),
            "GPUHitPct": 100 * m["server_gpu_cache_hit_rate"], "HostHitPct": 100 * m["server_cpu_cache_hit_rate"],
            "TheoreticalHitPct": 100 * m["theoretical_cache_hit_rate"], "Source": B300.name,
        })
    return rows


def f(v, spec):
    return "—" if v is None else spec.format(v)


def write_table(rows: list[dict], out: Path) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "agg_rounds.csv").open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (out / "agg_rounds.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n")
    head = ["轮次", "节点", "卡数", "并发", "HiCache / rank", "mem", "chunk / rank", "max-prefill", "delayer 上限",
            "tokenizer", "coordination", "piggyback", "NSR", "tok/s/GPU", "interactivity p50/p90", "每秒请求", "平均输入",
            "ITL p50/p90 (ms)",
            "TTFT p50/p90 (s)", "decode ms/token", "命中 总 / 理论", "服务端命中 GPU / 主机",
            "每 rank 在跑 / 排队", "状态"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        hit = (f"{r['UsageHitPct']:.1f}% / {r['TheoreticalHitPct']:.1f}%" if "UsageHitPct" in r else "—")
        split = (f"{r['GPUHitPct']:.1f}% / {f(r.get('HostHitPct'), '{:.1f}%')}"
                 if r.get("GPUHitPct") is not None else "—")
        rq = (f"{r['RunningPerRankMean']:.1f} / {r['QueuePerRankMean']:.1f}" if "RunningPerRankMean" in r else "—")
        lines.append("| " + " | ".join([
            f"**{r['Round']}**" if r["Round"] == "a6b" else r["Round"], r["Node"], str(r["GPUs"]),
            f(r["Concurrency"], "C{}"), r["HiCachePerRank"], r["MemFraction"], str(r["ChunkPerRank"]),
            str(r["MaxPrefillTokens"]), str(r["DelayerMaxPasses"]), str(r["TokenizerWorkers"]),
            r["Coordination"], r["Piggyback"], r["HsaNoScratchReclaim"] or "—",
            f(r.get("TokPerSecPerGPU"), "{:,.0f}"),
            f"{r['IntvtyP50']:.1f} / {r['IntvtyP90']:.1f}" if "IntvtyP50" in r else "—",
            f(r.get("ReqPerSec"), "{:.3f}"),
            f(r.get("MeanInputK"), "{:.1f}k"),
            f"{r['ITLp50Ms']:.1f} / {r['ITLp90Ms']:.1f}" if "ITLp50Ms" in r else "—",
            f"{r['TTFTp50S']:.1f} / {r['TTFTp90S']:.1f}" if "TTFTp50S" in r else "—",
            f(r.get("DecodeMsPerToken"), "{:.1f}"), hit, split, rq, r["Status"] or "—"]) + " |")
    note = [
        "", "- 每卡吞吐的分母是该实验的 GPU 数（TP4 为 4，TP8 为 8，B300 为 4）；每秒请求为整个实例的完成速率。",
        "- interactivity 为每请求 tok/s/user（AgentX `intvty`）；p90 是较慢的那一端（数值低于 p50），InferenceX 上希望它尽量大。",
        "- decode ms/token =（端到端延迟均值 − TTFT 均值）/ 输出 token 均值，包含被 prefill 打断的时间。",
        "- 命中 总：MI355X 为 AIPerf 的 API 计入命中（缓存 token / 输入 token），B300 为服务端 GPU + 主机命中之和"
        "（其 API 数据无前者）。服务端 GPU / 主机拆分来自 AgentX 记录的引擎计数器，a3 C48 与 a4 两者之和超过 100%，"
        "拆分只作参考。",
        "- 每 rank 在跑 / 排队为引擎 60 分钟窗口内各 rank 均值的平均（a3 C64 为事后找回的结果，无引擎采样）。",
        "- NSR 为 HSA_NO_SCRATCH_RECLAIM；MI355X 各轮 MTP 均为 EAGLE 5/1/6、模拟接受长度 3.61，与 B300 相同。",
        f"- 由 `scripts/plot_agg_rounds.py` 生成（{datetime.now().strftime('%Y-%m-%d')}），同目录的 `agg_rounds.csv` 含全部字段。",
    ]
    (out / "agg_rounds.md").write_text("\n".join(lines + note) + "\n")
    print(out / "agg_rounds.md")
    print(out / "agg_rounds.csv")


def profile_data() -> dict:
    caps = {r["file"]: r for r in json.loads((PROFILE / "breakdown.json").read_text())}
    pick = lambda cap, tp: next(r for name, r in caps.items() if name.startswith(cap) and f"-TP-{tp}-" in name)

    def merged(parts):
        total = {}
        for cats in parts:
            for k, v in cats.items():
                total[k] = total.get(k, 0) + v
        return total

    verify = [pick("cap2", tp)["batches"]["VERIFY"] for tp in (0, 1)]
    decode_cats = merged(b["cat"] for b in verify)
    step_ms = sum(b["dur"] for b in verify) / sum(b["n"] for b in verify) / 1000
    extend = pick("cap1", 1)["batches"]["EXTEND"]
    waiting = pick("cap1", 0)
    wait_cats = {k: v - waiting["batches"]["VERIFY"]["cat"].get(k, 0) for k, v in waiting["main_cat"].items()}

    def passes(name):
        d = {}
        for line in (PROFILE / name).read_text().splitlines():
            m = re.search(r'dp_rank="(\d+)".*mode="([^"]+)".*\} ([\d.]+)', line)
            if m:
                d[(int(m.group(1)), m.group(2))] = float(m.group(3))
        return d

    start, end = passes("cap1.passes.start"), passes("cap2.passes.end")
    delta = {k: end[k] - start.get(k, 0) for k in end}
    stamps = re.findall(r"^(\S+Z) (capture 1|done)", (PROFILE / "trigger.txt").read_text(), re.M)
    t = {label: datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ") for ts, label in stamps}
    window = (t["done"] - t["capture 1"]).total_seconds()
    decode_passes = delta[(0, "decode_cuda_graph")]
    return {"decode_cats": decode_cats, "decode_step_ms": step_ms, "prefill_cats": extend["cat"],
            "prefill_step_ms": extend["dur"] / extend["n"] / 1000, "wait_cats": wait_cats,
            "prefill_passes": [delta[(r, "prefill_none")] for r in range(4)],
            "decode_passes": decode_passes, "window_s": window,
            "decode_share": decode_passes * step_ms / 1000 / window}


def plot(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "legend.fontsize": 9})
    ours = [r for r in rows if r["Series"] == "MI355X" and r["Scored"]]
    b300 = {r["Concurrency"]: r for r in rows if r["Series"] == "B300"}
    bars = ours + [b300[48]]
    labels = [f"{r['Round']}\nC{r['Concurrency']}" + ("\n8 GPUs" if r["GPUs"] == 8 else "") for r in ours]
    labels.append("B300\nC48")
    x = list(range(len(bars)))
    best = next(i for i, r in enumerate(bars) if r["Round"] == "a6b")
    colors = ["#2563a8" if r["Series"] == "MI355X" else "#8a8a8a" for r in bars]
    colors[best] = "#c66020"

    def style(ax, title, ylabel):
        ax.set(title=title, ylabel=ylabel, xticks=x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig, axes = plt.subplots(4, 2, figsize=(17, 19), layout="constrained")
    ax = axes[0, 0]
    b = ax.bar(x, [r["TokPerSecPerGPU"] for r in bars], color=colors)
    ax.bar_label(b, fmt="{:,.0f}", fontsize=8.5, padding=2)
    ax.axhline(b300[48]["TokPerSecPerGPU"], color="#8a8a8a", linestyle="--", linewidth=1)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    style(ax, "Total throughput / GPU (cached input tokens included)", "tokens/s/GPU")

    ax = axes[0, 1]
    b = ax.bar(x, [r["OutputTokPerSecPerGPU"] for r in bars], color=colors)
    ax.bar_label(b, fmt="{:.0f}", fontsize=8.5, padding=2)
    style(ax, "Output throughput / GPU", "tokens/s/GPU")

    ax = axes[1, 0]
    per_gpu = [r["ReqPerSec"] / r["GPUs"] for r in bars]
    b = ax.bar(x, per_gpu, color=colors)
    ax.bar_label(b, fmt="{:.3f}", fontsize=8.5, padding=2)
    style(ax, "Completed requests / s / GPU  (bars)   |   mean input tokens (line)", "requests/s/GPU")
    ax.set_ylim(0, max(per_gpu) * 1.75)
    twin = ax.twinx()
    inputs = [r["MeanInputK"] for r in bars]
    twin.plot(x, inputs, color="#3a8a3a", marker="o", linewidth=1.5)
    for i, v in enumerate(inputs):
        twin.annotate(f"{v:.0f}k", (i, v), xytext=(0, 6), textcoords="offset points", ha="center",
                      fontsize=8, color="#3a8a3a")
    spread = max(inputs) - min(inputs)
    twin.set_ylim(min(inputs) - 2.6 * spread, max(inputs) + 0.45 * spread)
    twin.set_ylabel("mean input (k tokens)", color="#3a8a3a")
    twin.spines[["top"]].set_visible(False)

    width = 0.38
    for ax, title, unit, p50, p90, fmt in (
        (axes[1, 1], "Full-response inter-token latency", "milliseconds", "ITLp50Ms", "ITLp90Ms", "{:.1f}"),
        (axes[2, 0], "Time to first token", "seconds", "TTFTp50S", "TTFTp90S", "{:.1f}"),
    ):
        b1 = ax.bar([i - width / 2 for i in x], [r[p50] for r in bars], width, color=colors, label="p50")
        b2 = ax.bar([i + width / 2 for i in x], [r[p90] for r in bars], width, color=colors, alpha=0.45,
                    hatch="//", label="p90")
        ax.bar_label(b1, fmt=fmt, fontsize=7.5, padding=1)
        ax.bar_label(b2, fmt=fmt, fontsize=7.5, padding=1)
        ax.legend(frameon=False, loc="upper left")
        style(ax, title + " (solid p50, hatched p90)", unit)

    ax = axes[2, 1]
    hit = [r["UsageHitPct"] for r in bars]
    b = ax.bar(x, hit, color=colors, label="measured")
    ax.bar_label(b, fmt="{:.1f}", fontsize=8, padding=2)
    ax.scatter(x, [r["TheoreticalHitPct"] for r in bars], marker="_", s=500, color="black", zorder=3,
               label="theoretical")
    ax.set_ylim(90, 99.5)
    ax.legend(frameon=False, loc="upper center", ncol=2)
    style(ax, "Prompt cache hit (MI355X: API usage; B300: server GPU + host)", "% of input tokens")

    ax = axes[3, 0]
    m_rows = [r for r in bars if r.get("EffConc") is not None]
    mx = [bars.index(r) for r in m_rows]
    first = [r["EffConcBeforeFirstToken"] for r in m_rows]
    decode = [r["EffConc"] - r["EffConcBeforeFirstToken"] for r in m_rows]
    ax.bar(mx, decode, color=[colors[i] for i in mx], label="decoding")
    ax.bar(mx, first, bottom=decode, color=[colors[i] for i in mx], alpha=0.4, hatch="//",
           label="waiting for first token")
    for i, d, w in zip(mx, decode, first):
        ax.annotate(f"{d:.0f} + {w:.0f}", (i, d + w), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.legend(frameon=False, loc="upper left")
    style(ax, "Client effective concurrency (B300: not published)", "concurrent requests")

    ax = axes[3, 1]
    e_rows = [r for r in bars if "RunningPerRankMean" in r]
    ex = [bars.index(r) for r in e_rows]
    for key, offset, alpha, hatch, label in (("Running", -width / 2, 1.0, None, "running / rank"),
                                             ("Queue", width / 2, 0.45, "//", "queued / rank")):
        mean = [r[f"{key}PerRankMean"] for r in e_rows]
        err = [[m - r[f"{key}PerRankMin"] for m, r in zip(mean, e_rows)],
               [r[f"{key}PerRankMax"] - m for m, r in zip(mean, e_rows)]]
        bb = ax.bar([i + offset for i in ex], mean, width, yerr=err, capsize=3, color=[colors[i] for i in ex],
                    alpha=alpha, hatch=hatch, label=label, error_kw={"elinewidth": 1})
        ax.bar_label(bb, fmt="{:.1f}", fontsize=7.5, padding=1, label_type="center")
    ax.legend(frameon=False, loc="upper left")
    style(ax, "Engine: requests per DP rank (mean, whiskers = min/max rank; B300: n/a)", "requests")

    fig.suptitle("GLM-5.2 MXFP4 aggregated (mixed) on MI355X vs B300 Dynamo C48 | AgentX, 3600 s per point",
                 fontsize=15, fontweight="bold")
    fig.supxlabel("Orange: a6b (best TP4). Gray: InferenceX B300 Dynamo + SGLang (4 GPUs, NVFP4). "
                  "TP8 rounds use 8 GPUs; per-GPU values divide by the GPUs used.\n"
                  "Run-to-run noise from the input-length mix is about ±3%: compare requests/s together "
                  "with tokens/s/GPU.", fontsize=10)
    save(fig, out / "agg_rounds", plt)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), layout="constrained")
    curve = sorted(r["Concurrency"] for r in rows if r["Series"] == "B300")
    dyn = sorted((r for r in json.loads(B300.read_text())["rows"] if r["framework"] == "dynamo-sglang"),
                 key=lambda r: r["conc"])
    def b300_values(key):
        out_ = []
        for r in dyn:
            m = r["metrics"]
            out_.append({"tps": m["tput_per_gpu"], "ttft": m["median_ttft"],
                         "dec": 1000 * (m["mean_e2el"] - m["mean_ttft"]) / m["mean_output_tokens_actual"],
                         "req": m["total_requests_completed"] / m["duration_seconds"] / r["num_decode_gpu"]}[key])
        return out_
    measured_rows = [r for r in rows if r["Series"] == "MI355X" and "TokPerSecPerGPU" in r]
    palette = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]
    markers = ["o", "s", "^", "v", "P", "X", "<", ">"]
    point_style = {}
    per_conc = {}
    for r in measured_rows:
        per_conc.setdefault(r["Concurrency"], []).append(r)
    others = [r for r in measured_rows if r["Round"] != "a6b"]
    for r in measured_rows:
        group = per_conc[r["Concurrency"]]
        k = group.index(r) - (len(group) - 1) / 2
        if r["Round"] == "a6b":
            color, marker = "#c66020", "D"
        else:
            i = others.index(r)
            color, marker = palette[i % len(palette)], markers[i % len(markers)]
        point_style[(r["Round"], r["Concurrency"])] = (color, marker, r["Concurrency"] * 1.035 ** k)
    for ax, key, field, title, unit in (
        (axes[0, 0], "tps", "TokPerSecPerGPU", "Total throughput / GPU", "tokens/s/GPU"),
        (axes[0, 1], "dec", "DecodeMsPerToken", "Decode time per output token\n"
                                                 "((mean E2E - mean TTFT) / mean output)", "ms/token"),
        (axes[1, 0], "ttft", "TTFTp50S", "TTFT p50", "seconds"),
        (axes[1, 1], "req", None, "Completed requests / s / GPU", "requests/s/GPU"),
    ):
        cx = [r["conc"] for r in dyn]
        vals = b300_values(key)
        ax.plot(cx, vals, color="#8a8a8a", linewidth=2, marker="o", label="B300 Dynamo (TP4; DPA4 from C24)")
        for c, v in zip(cx, vals):
            if c in curve:
                ax.annotate(f"{v:,.0f}" if key == "tps" else f"{v:.2f}" if key == "req" else f"{v:.1f}",
                            (c, v), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8,
                            color="#555555")
        for r in measured_rows:
            v = r[field] if field else r["ReqPerSec"] / r["GPUs"]
            color, marker, px = point_style[(r["Round"], r["Concurrency"])]
            hollow = r["Round"] == "a2"
            label = f"{r['Round']} C{r['Concurrency']}" + (" (8 GPUs)" if r["GPUs"] == 8 else "") + \
                (" (600 s debug)" if hollow else "")
            ax.scatter(px, v, s=70 if r["Round"] == "a6b" else 48, zorder=3, marker=marker,
                       facecolors="white" if hollow else color, edgecolors=color, label=label)
        ax.set_xscale("log", base=2)
        ax.set(title=title, ylabel=unit, xlabel="concurrency", xticks=[1, 8, 16, 24, 32, 48, 64, 128])
        ax.set_xticklabels(["1", "8", "16", "24", "32", "48", "64", "128"])
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        if key == "tps":
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.suptitle("Throughput and latency against concurrency: MI355X rounds vs B300 Dynamo",
                 fontsize=15, fontweight="bold")
    save(fig, out / "agg_vs_concurrency", plt)

    # InferenceX goal: high tokens/s/GPU at high interactivity (p90 = the slow tail of tok/s/user).
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), layout="constrained")
    for ax, key, api_key, title in ((axes[0], "IntvtyP90", "p90_intvty", "interactivity p90"),
                                    (axes[1], "IntvtyP50", "median_intvty", "interactivity p50")):
        bx = [r["metrics"][api_key] for r in dyn]
        by = [r["metrics"]["tput_per_gpu"] for r in dyn]
        ax.plot(bx, by, color="#8a8a8a", linewidth=2, marker="o", label="B300 Dynamo (4 GPUs)")
        for r, x_, y_ in zip(dyn, bx, by):
            ax.annotate(f"C{r['conc']}", (x_, y_), xytext=(5, 4), textcoords="offset points", fontsize=8,
                        color="#555555")
        pts = []
        for r in measured_rows:
            color, marker, _ = point_style[(r["Round"], r["Concurrency"])]
            hollow = r["Round"] == "a2"
            ax.scatter(r[key], r["TokPerSecPerGPU"], s=80 if r["Round"] in ("a6b", "a3") else 48, zorder=3,
                       marker=marker, facecolors="white" if hollow else color, edgecolors=color,
                       label=f"{r['Round']} C{r['Concurrency']}" + (" (8 GPUs)" if r["GPUs"] == 8 else "")
                       + (" (600 s debug)" if hollow else ""))
            ax.annotate(f"{r['Round']} C{r['Concurrency']}", (r[key], r["TokPerSecPerGPU"]), xytext=(6, -10),
                        textcoords="offset points", fontsize=7.5, color=color)
            if not hollow:
                pts.append((r[key], r["TokPerSecPerGPU"]))
        front = sorted(p for p in pts if not any(q[0] >= p[0] and q[1] >= p[1] and q != p for q in pts))
        ax.plot([p[0] for p in front], [p[1] for p in front], color="#c66020", linestyle="--", linewidth=1.2,
                label="MI355X frontier (scored rounds)")
        ax.set_xscale("log")
        ax.set(title=f"Total throughput / GPU vs {title}", xlabel=f"{title} (tokens/s/user, log scale)",
               ylabel="tokens/s/GPU")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(alpha=0.25, which="both")
        ax.spines[["top", "right"]].set_visible(False)
        ax.annotate("better", xy=(0.97, 0.97), xytext=(0.82, 0.86), xycoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "color": "#555555"}, fontsize=9, color="#555555")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle("InferenceX view: throughput per GPU against per-user interactivity (AgentX, 3600 s per point)",
                 fontsize=14, fontweight="bold")
    fig.supxlabel("interactivity p90 is the 90th-percentile slow tail of per-request tok/s/user (lower than p50); "
                  "points up and to the right are better.", fontsize=9.5)
    save(fig, out / "agg_pareto_interactivity", plt)

    p = profile_data()
    groups = [("attn_dsa", "attention / DSA", "#c66020"), ("moe", "MoE", "#2563a8"),
              ("gemm", "dense GEMM", "#3a8a3a"), ("comm", "communication", "#8a5ab0")]

    def shares(cats):
        total = sum(cats.values())
        s = [100 * cats.get(k, 0) / total for k, _, _ in groups]
        return s + [100 - sum(s)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.1, 1.6]})
    ax = axes[0]
    d = 100 * p["decode_share"]
    ax.bar([0], [d], color="#2563a8", width=0.5, label="decode steps (all ranks together)")
    ax.bar([0], [100 - d], bottom=[d], color="#c66020", width=0.5, alpha=0.75,
           label="eager steps (some rank prefilling, others idle)")
    ax.annotate(f"{d:.1f}%\n{p['decode_passes']:,.0f} steps x {p['decode_step_ms']:.1f} ms", (0, d / 2),
                ha="center", va="center", color="white", fontsize=10)
    ax.annotate(f"{100 - d:.1f}%", (0, d + (100 - d) / 2), ha="center", va="center", color="white", fontsize=11)
    ax.set(xlim=(-0.6, 0.6), ylim=(0, 100), xticks=[], ylabel="% of wall time",
           title=f"Wall time over {p['window_s'] / 60:.0f} min (C64 load)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=8.5)

    ax = axes[1]
    b = ax.bar(range(4), p["prefill_passes"], color="#c66020", alpha=0.75)
    ax.bar_label(b, fmt="{:.0f}", padding=2)
    ax.set(xticks=range(4), xticklabels=[f"DP{r}" for r in range(4)], ylabel="prefill steps",
           title=f"Prefill steps per rank in the same {p['window_s'] / 60:.0f} min\n"
                 f"(every rank: {p['decode_passes']:,.0f} decode steps)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    cols = [("decode step\n" + f"{p['decode_step_ms']:.1f} ms, bs 8-13 / rank", shares(p["decode_cats"])),
            ("prefill step, prefilling rank\n" + f"{p['prefill_step_ms']:.0f} ms (8192-token chunk)",
             shares(p["prefill_cats"])),
            ("same window, a waiting rank\n(kernel time outside its decode steps)", shares(p["wait_cats"]))]
    bottom = [0.0] * len(cols)
    for gi, (key, label, color) in enumerate(groups + [("other", "other", "#bbbbbb")]):
        vals = [c[1][gi] for c in cols]
        ax.bar(range(len(cols)), vals, bottom=bottom, color=color, label=label, width=0.6)
        for i, v in enumerate(vals):
            if v >= 4:
                ax.annotate(f"{v:.0f}%", (i, bottom[i] + v / 2), ha="center", va="center", color="white",
                            fontsize=9)
        bottom = [b_ + v for b_, v in zip(bottom, vals)]
    ax.set(xticks=range(len(cols)), ylim=(0, 100), ylabel="% of main-stream kernel time",
           title="Kernel time by category (a6b, torch profiler)")
    ax.set_xticklabels([c[0] for c in cols], fontsize=8.5)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Where the GPU time goes: a6b (TP4 + DPA4, C64) under AgentX load",
                 fontsize=14, fontweight="bold")
    fig.supxlabel("Decode share = decode CUDA-graph passes x mean verify-step time / window. A waiting rank "
                  "spends its time in collective kernels until the prefilling rank finishes the step.",
                  fontsize=9.5)
    save(fig, out / "gpu_time_breakdown", plt)


def save(fig, stem: Path, plt) -> None:
    for ext in ("png", "svg"):
        fig.savefig(stem.with_suffix("." + ext), dpi=160)
        print(stem.with_suffix("." + ext))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--table-only", action="store_true")
    mode.add_argument("--plot-only", action="store_true", help="draw from OUTPUT/agg_rounds.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "summary")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        rows = json.loads((args.output / "agg_rounds.json").read_text())
    else:
        rows = load_rows()
        write_table(rows, args.output)
    if not args.table_only:
        plot(rows, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
