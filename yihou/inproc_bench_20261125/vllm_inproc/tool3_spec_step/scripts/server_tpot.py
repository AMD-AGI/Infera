#!/usr/bin/env python3
"""Summarize a server_reference.sh point and, optionally, compare it with an internal bench point.

Usage: server_tpot.py SERVER_POINT [--internal INTERNAL_POINT] [--local-batch N]
Server TPOT is the client's per-request (latency - TTFT) / (OSL - 1) (`vllm bench serve`). The server-side
steady-decode view uses the engine stat logger lines (`Engine 00d: ... Avg generation throughput: Y tokens/s,
Running: R reqs, Waiting: W reqs`, 10 s intervals): intervals where every DP engine runs exactly the local batch
with nothing waiting give TPOT_ms = R / Y * 1000. Internal TPOT is profile_decode's
effective_token_latency_ms_per_user = elapsed * 1000 * C / useful tokens (full-loop amortized). Different
estimators of the same per-user decode latency; confounds are listed in the output, not corrected.
"""
import argparse
import json
import re
import sys
from pathlib import Path

KEYS = ("completed", "max_concurrency", "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms", "mean_itl_ms",
        "median_itl_ms", "mean_ttft_ms", "median_ttft_ms", "output_throughput", "total_input_tokens",
        "total_output_tokens", "duration", "mean_e2el_ms", "request_throughput")

STAT_LINE = re.compile(r"Engine (?P<dp>\d+):.*?Avg prompt throughput: (?P<pthr>[0-9.]+) tokens/s, "
                       r"Avg generation throughput: (?P<thr>[0-9.]+) tokens/s, Running: (?P<run>\d+) reqs, "
                       r"Waiting: (?P<wait>\d+) reqs")
ITER_LINE = re.compile(r"Engine (?P<dp>\d+): Iteration\((?P<it>\d+)\): (?P<ctx>\d+) context requests, (?P<ctxt>\d+) context "
                       r"tokens, (?P<gen>\d+) generation requests, (?P<gent>\d+) generation tokens, iteration elapsed "
                       r"time: (?P<ms>[0-9.]+) ms(?P<dummy> \(dummy\))?")
ACCEPT_LINE = re.compile(r"Mean acceptance length: (?P<acc>[0-9.]+)")


def server_summary(point):
    point = Path(point)
    path = point / "client.json"
    summary = {}
    if path.is_file():
        record = json.loads(path.read_text())
        summary = {key: record.get(key) for key in KEYS if key in record}
    log = point / "server.log"
    if log.is_file():
        accepts = [float(m.group("acc")) for m in ACCEPT_LINE.finditer(log.read_text(errors="replace"))]
        if accepts:
            summary["server_log_accept_len_mean"] = sum(accepts) / len(accepts)
            summary["server_log_accept_len_samples"] = len(accepts)
    return summary


def steady_decode(point, local_batch):
    log = Path(point) / "server.log"
    if not log.is_file() or not local_batch:
        return None
    rows = [m.groupdict() for m in STAT_LINE.finditer(log.read_text(errors="replace"))]
    steady = [r for r in rows if int(r["run"]) == local_batch and int(r["wait"]) == 0 and float(r["pthr"]) == 0.0]
    if not steady:
        return {"intervals": 0, "stat_lines": len(rows)}
    thr = [float(r["thr"]) for r in steady if float(r["thr"]) > 0]
    tpots = sorted(local_batch / t * 1000.0 for t in thr)
    return {"intervals": len(steady), "dp_engines": sorted({r["dp"] for r in steady}),
            "gen_throughput_per_engine_mean": sum(thr) / len(thr) if thr else None,
            "tpot_ms_mean": sum(tpots) / len(tpots) if tpots else None,
            "tpot_ms_median": tpots[len(tpots) // 2] if tpots else None,
            "note": "10 s logger intervals with Running == local batch, Waiting == 0 and no prompt throughput"}


def steady_iterations(point, local_batch):
    """Per-iteration server step times (--enable-logging-iteration-details): decode-only iterations
    (0 context requests) whose generation requests == the local batch, per DP engine."""
    log = Path(point) / "server.log"
    if not log.is_file() or not local_batch:
        return None
    rows = [m.groupdict() for m in ITER_LINE.finditer(log.read_text(errors="replace"))]
    decode = [r for r in rows if r["dummy"] is None and int(r["ctx"]) == 0]
    full = [float(r["ms"]) for r in decode if int(r["gen"]) == local_batch]
    out = {"iteration_lines": len(rows), "decode_only_iterations": len(decode), "full_batch_iterations": len(full)}
    if full:
        ordered = sorted(full)
        out.update(step_ms_median=ordered[len(ordered) // 2], step_ms_mean=sum(full) / len(full),
                   step_ms_p10=ordered[len(ordered) // 10], step_ms_p90=ordered[(9 * len(ordered)) // 10])
    out["note"] = ("'iteration elapsed time' as logged by the engine (SchedulerIterationDetails.elapsed_ms); "
                   "decode-only iterations with generation requests == local batch")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("server_point")
    parser.add_argument("--internal")
    parser.add_argument("--local-batch", type=int, default=0, help="per-DP running requests for the steady-decode view")
    args = parser.parse_args()
    out = {"server": server_summary(args.server_point)}
    out["server_steady_decode"] = steady_decode(args.server_point, args.local_batch)
    out["server_steady_iterations"] = steady_iterations(args.server_point, args.local_batch)
    if args.internal:
        result = json.loads((Path(args.internal) / "result_yihou.json").read_text())
        internal = result["effective_token_latency_ms_per_user"]
        out["internal"] = {key: result.get(key) for key in (
            "batch_size", "input_len", "output_len", "effective_token_latency_ms_per_user", "step_seconds_p50",
            "output_tokens_per_second", "realized_accept_length", "verify_iterations", "kv_init")}
        for key in ("mean_tpot_ms", "median_tpot_ms"):
            server = out["server"].get(key)
            if server:
                out[f"delta_{key}_server_vs_internal_pct"] = (server - internal) / internal * 100.0
        steady = out.get("server_steady_decode") or {}
        for key in ("tpot_ms_mean", "tpot_ms_median"):
            if steady.get(key):
                out[f"delta_steady_{key}_server_vs_internal_pct"] = (steady[key] - internal) / internal * 100.0
        iters = out.get("server_steady_iterations") or {}
        if iters.get("step_ms_median") and result.get("step_seconds_p50"):
            out["delta_step_ms_median_server_vs_internal_pct"] = (
                (iters["step_ms_median"] - result["step_seconds_p50"] * 1000) / (result["step_seconds_p50"] * 1000) * 100.0)
        out["confounds"] = [
            "server: real Scheduler with async scheduling (CPU work of step i+1 overlaps the GPU of step i), DP "
            "balancer, admission of new requests between decode steps; the bench is synchronous, one fixed batch",
            "server requests start and finish at different times (effective concurrency < C at the tail and while "
            "the client refills); the bench decodes one fixed batch from exactly ISL",
            "client TPOT excludes the first token; internal TPOT amortizes over all useful tokens of the fixed batch",
            "prefix KV: server DecodeBenchConnector fill (constant 0.015 unless FILL says otherwise; packed layouts 0; "
            "circular buffers 0); bench --kv-init (compare with --kv-init decode-bench for the same contents)",
            "acceptance: same synthetic rates on both sides; per-request seeds differ (server: unseeded requests)",
        ]
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
