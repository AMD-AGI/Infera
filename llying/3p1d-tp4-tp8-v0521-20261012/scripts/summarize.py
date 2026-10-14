#!/usr/bin/env python3
"""Summarize every AgentX point under results/<round>/c<CONC>/ into summary.csv and summary.md.

Usage: summarize.py ROUND [ROUND ...]   (run from anywhere; paths are relative to this kit)
Interactivity follows InferenceX / collect_agentx.py: P90 = 1 / full_response_itl.p90,
P50 = 1 / full_response_itl.p50. Per-GPU throughput is the aggregate's own per_gpu value
(total tokens / (prefill GPUs + decode GPUs), here 3x4 + 8 = 20).
"""

import csv
import json
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
FIELDS = ["round", "conc", "gpus", "tok_s_per_gpu", "total_tok_s", "out_tok_s_per_gpu",
          "p90_intv", "p50_intv", "itl_p50_ms", "itl_p90_ms", "ttft_p50_s", "ttft_p90_s",
          "ttft_mean_s", "tpot_mean_ms", "e2e_p50_s", "requests_ok", "error_dropped",
          "input_mean", "output_mean", "theoretical_hit", "gpu_cache_hit", "cpu_cache_hit",
          "duration_s"]


def row(round_name: str, path: Path) -> dict:
    j = json.loads(path.read_text())
    m = j["request_metrics"]
    lat, thr = m["latency"], m["throughput"]
    full = lat["full_response_itl"]
    cache = (j.get("server_metrics") or {}).get("cache") or {}
    gpus = int(j.get("num_prefill_gpu", 0)) + int(j.get("num_decode_gpu", 0))
    return {
        "round": round_name, "conc": j["conc"], "gpus": gpus,
        "tok_s_per_gpu": round(thr["per_gpu"]["total_tput_tps"], 1),
        "total_tok_s": round(thr["total"]["tokens_per_second"], 1),
        "out_tok_s_per_gpu": round(thr["per_gpu"]["output_tput_tps"], 2),
        "p90_intv": round(1 / full["p90"], 2), "p50_intv": round(1 / full["p50"], 2),
        "itl_p50_ms": round(full["p50"] * 1000, 2), "itl_p90_ms": round(full["p90"] * 1000, 2),
        "ttft_p50_s": round(lat["ttft"]["p50"], 3), "ttft_p90_s": round(lat["ttft"]["p90"], 3),
        "ttft_mean_s": round(lat["ttft"]["mean"], 3),
        "tpot_mean_ms": round(lat["tpot"]["mean"] * 1000, 2),
        "e2e_p50_s": round(lat["e2el"]["p50"], 3),
        "requests_ok": j["num_requests_successful"],
        "error_dropped": (j.get("request_accounting") or {}).get("records_error_dropped"),
        "input_mean": round(m["tokens"]["input"]["mean"]),
        "output_mean": round(m["tokens"]["output_actual"]["mean"]),
        "theoretical_hit": round(m["cache"]["theoretical_cache_hit_rate"], 4),
        "gpu_cache_hit": cache.get("gpu_cache_hit_rate"),
        "cpu_cache_hit": cache.get("cpu_cache_hit_rate"),
        "duration_s": round(thr["duration_seconds"], 1),
    }


def main() -> int:
    rounds = sys.argv[1:] or sorted(p.name for p in (KIT / "results").iterdir() if p.is_dir())
    rows = []
    for r in rounds:
        for path in sorted((KIT / "results" / r).glob("c*/agentx_conc*.json")):
            rows.append(row(r, path))
    rows.sort(key=lambda x: (x["round"], x["conc"]))
    out = KIT / "results"
    with open(out / "summary.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    cols = ["round", "conc", "tok_s_per_gpu", "out_tok_s_per_gpu", "total_tok_s", "p50_intv",
            "p90_intv", "itl_p50_ms", "ttft_p50_s", "ttft_p90_s", "tpot_mean_ms", "requests_ok",
            "error_dropped", "input_mean", "gpu_cache_hit", "cpu_cache_hit"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
