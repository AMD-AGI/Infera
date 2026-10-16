#!/usr/bin/env python3
"""Summary of one or more published aggregated rounds (results/<variant>/c<CONC>/).

Client side from agentx_conc*.json and the AIPerf export; engine side from samples.jsonl
(agg_sampler.py), averaged over the last DURATION seconds of sampling (the measured window;
the client's warm-up lies before it).
Usage: agg_report.py RESULT_DIR [RESULT_DIR ...]
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def client(d: Path) -> dict:
    j = json.loads(next(d.glob("agentx_conc*.json")).read_text())
    m = j["request_metrics"]
    lat, thr = m["latency"], m["throughput"]
    cache = (j.get("server_metrics") or {}).get("cache") or {}
    r = {"tps": thr["per_gpu"]["total_tput_tps"], "out": thr["per_gpu"]["output_tput_tps"],
         "ok": j["num_requests_successful"], "inp": m["tokens"]["input"]["mean"],
         "itl50": lat["full_response_itl"]["p50"] * 1000, "itl90": lat["full_response_itl"]["p90"] * 1000,
         "t50": lat["ttft"]["p50"], "t90": lat["ttft"]["p90"],
         "ghit": cache.get("gpu_cache_hit_rate"), "chit": cache.get("cpu_cache_hit_rate"),
         "theo": m["cache"]["theoretical_cache_hit_rate"]}
    export = d / "profile_export_aiperf.json"
    if export.exists():
        e = json.loads(export.read_text())
        r["eff"] = e["effective_concurrency"]["avg"]
        r["eff_p"] = e["effective_prefill_concurrency"]["avg"]
    return r


def engine(d: Path) -> dict:
    path = d / "samples.jsonl"
    if not path.exists():
        return {}
    env = dict(line.split("=", 1) for line in (d / "round.env").read_text().splitlines() if "=" in line)
    duration = float(env.get("DURATION") or 3600)
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    rows = [x for x in rows if "m" in x]
    if len(rows) < 2:
        return {}
    end = rows[-1]["t"]
    win = [x for x in rows if x["t"] >= end - duration]
    gauges = defaultdict(lambda: defaultdict(list))
    for x in win:
        for key, value in x["m"].items():
            name, labels = (key.split("{", 1) + [""])[:2]
            rank = re.search(r'dp_rank="(\d+)"', labels)
            gauges[name][rank.group(1) if rank else "-"].append(value)
    first, last = win[0], win[-1]
    span = last["t"] - first["t"]

    def rate(name: str) -> float | None:
        a = sum(v for k, v in first["m"].items() if k.split("{")[0] == name)
        b = sum(v for k, v in last["m"].items() if k.split("{")[0] == name)
        return (b - a) / span if span > 0 and any(k.split("{")[0] == name for k in last["m"]) else None

    idle = {}
    for key, value in last["m"].items():
        if key.startswith("scheduler_idle_seconds_total") and key in first["m"] and span > 0:
            rank = re.search(r'dp_rank="(\d+)"', key)
            idle[rank.group(1) if rank else "-"] = (value - first["m"][key]) / span
    gauges["idle fraction"] = {r: [v] for r, v in idle.items()}
    cpu = {}
    for key, value in last["m"].items():
        if key.startswith(("process_cpu_seconds_total", "scheduler_process_cpu_seconds_total")) and key in first["m"] and span > 0:
            cpu[key] = (value - first["m"][key]) / span
    return {"gauges": {n: {r: sum(v) / len(v) for r, v in by.items()} for n, by in gauges.items()},
            "span": span, "cpu": cpu, "prompt_tps": rate("prompt_tokens_total"),
            "cached_tps": rate("cached_tokens_total"), "gen_tps": rate("generation_tokens_total")}


def fmt(v, spec="{:.2f}"):
    return "—" if v is None else spec.format(v)


def main() -> int:
    for arg in sys.argv[1:]:
        d = Path(arg)
        if not any(d.glob("agentx_conc*.json")):
            print(f"{d.parent.name} {d.name}: no client result (agentx_conc*.json missing)")
            continue
        c = client(d)
        print(f"{d.parent.name} {d.name}: {c['tps']:,.0f} tok/s/GPU (output {c['out']:.1f}), "
              f"{c['ok']:,} requests, mean input {c['inp'] / 1000:.1f}k, "
              f"ITL p50/p90 {c['itl50']:.1f}/{c['itl90']:.1f} ms, TTFT p50/p90 {c['t50']:.1f}/{c['t90']:.1f} s, "
              f"cache hit GPU/host {fmt(c['ghit'], '{:.1%}')}/{fmt(c['chit'], '{:.1%}')} "
              f"(theoretical {c['theo']:.1%}), effective conc {fmt(c.get('eff'), '{:.1f}')} "
              f"(before first token {fmt(c.get('eff_p'), '{:.1f}')})")
        e = engine(d)
        if not e:
            continue
        g = e["gauges"]
        ranks = sorted({r for by in g.values() for r in by if r != "-"}, key=int) or ["-"]
        cols = [("running", "num_running_reqs"), ("queue", "num_queue_reqs"), ("KV use", "token_usage"),
                ("gen tok/s", "gen_throughput"), ("hit rate", "cache_hit_rate"), ("accept len", "spec_accept_length"),
                ("util", "utilization"), ("idle", "idle fraction")]
        print(f"  engine, mean over {e['span'] / 60:.0f} min: prompt {fmt(e['prompt_tps'], '{:,.0f}')} tok/s, "
              f"of which cached {fmt(e['cached_tps'], '{:,.0f}')}; generated {fmt(e['gen_tps'], '{:,.0f}')} tok/s")
        print("  rank  " + "  ".join(f"{h:>10}" for h, _ in cols))
        for r in ranks:
            print(f"  {r:>4}  " + "  ".join(f"{fmt(g.get(n, {}).get(r)):>10}" for _, n in cols))
        if e["cpu"]:
            print("  CPU cores (window mean): " + ", ".join(
                f"{re.sub(r'.*(component|dp_rank)=\"([^\"]+)\".*', r'\2', k)}={v:.2f}" for k, v in sorted(e["cpu"].items())))
        extra = sorted(n for n in g if n.startswith(("hicache_", "num_retracted")))
        for n in extra:
            print(f"  {n}: " + ", ".join(f"{r}={v:.3g}" for r, v in sorted(g[n].items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
