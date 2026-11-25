#!/usr/bin/env python3
"""Summarize a server_reference.sh point and, optionally, compare it with an internal bench point.

Usage: server_tpot.py SERVER_POINT [--internal INTERNAL_POINT]
Server TPOT is the client's per-request (latency - TTFT) / (OSL - 1), mean and median over requests
(sglang.benchmark.serving). Internal TPOT is profile_decode's effective_token_latency_ms_per_user
= elapsed * 1000 * C / useful tokens (full-loop amortized). They are different estimators of the
same per-user decode latency; confounds are listed in the output, not corrected.
"""
import argparse
import json
import re
import sys
from pathlib import Path

KEYS = ("completed", "max_concurrency", "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms", "mean_itl_ms",
        "median_itl_ms", "mean_ttft_ms", "median_ttft_ms", "output_throughput", "total_input_tokens",
        "total_output_tokens", "duration", "accept_length", "mean_e2e_latency_ms")


def server_summary(point):
    point = Path(point)
    lines = [line for line in (point / "client.jsonl").read_text().splitlines() if line.strip()]
    record = json.loads(lines[-1])
    summary = {key: record.get(key) for key in KEYS if key in record}
    log = point / "server.log"
    if log.is_file():
        accepts = re.findall(r"accept len: ([0-9.]+)", log.read_text(errors="replace"))
        if accepts:
            values = [float(v) for v in accepts]
            summary["server_log_accept_len_mean"] = sum(values) / len(values)
            summary["server_log_accept_len_samples"] = len(values)
    return summary


DECODE_LINE = re.compile(r"\[(?P<ts>[^\]]*?) DP(?P<dp>\d+)[^\]]*\] Decode batch, #running-req: (?P<run>\d+).*?"
                         r"accept len: (?P<acc>[0-9.]+).*?gen throughput \(token/s\): (?P<thr>[0-9.]+), #queue-req: (?P<q>\d+)")


def steady_decode(point, local_batch):
    """Server-side decode-only view: scheduler log intervals where every DP rank runs exactly the local
    batch with an empty queue (no prefill can interleave). TPOT_ms = running / gen throughput * 1000."""
    log = Path(point) / "server.log"
    if not log.is_file() or not local_batch:
        return None
    rows = [m.groupdict() for m in DECODE_LINE.finditer(log.read_text(errors="replace"))]
    steady = [r for r in rows if int(r["run"]) == local_batch and int(r["q"]) == 0]
    if not steady:
        return {"intervals": 0}
    thr = [float(r["thr"]) for r in steady]
    acc = [float(r["acc"]) for r in steady]
    tpots = sorted(local_batch / t * 1000.0 for t in thr if t > 0)
    return {"intervals": len(steady), "dp_ranks": sorted({r["dp"] for r in steady}),
            "gen_throughput_per_dp_mean": sum(thr) / len(thr), "accept_len_mean": sum(acc) / len(acc),
            "tpot_ms_mean": sum(tpots) / len(tpots), "tpot_ms_median": tpots[len(tpots) // 2],
            "first_interval_note": "the first interval after the last prefill still contains part of it; see intervals"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("server_point")
    parser.add_argument("--internal")
    parser.add_argument("--local-batch", type=int, default=0, help="per-DP running requests for the steady-decode view")
    args = parser.parse_args()
    out = {"server": server_summary(args.server_point)}
    out["server_steady_decode"] = steady_decode(args.server_point, args.local_batch)
    if args.internal:
        result = json.loads((Path(args.internal) / "result_yihou.json").read_text())
        internal = result["effective_token_latency_ms_per_user"]
        out["internal"] = {key: result.get(key) for key in (
            "batch_size", "input_len", "output_len", "effective_token_latency_ms_per_user",
            "output_tokens_per_second", "realized_accept_length", "verify_iterations")}
        for key in ("mean_tpot_ms", "median_tpot_ms"):
            server = out["server"].get(key)
            if server:
                out[f"delta_{key}_server_vs_internal_pct"] = (server - internal) / internal * 100.0
        steady = out.get("server_steady_decode") or {}
        for key in ("tpot_ms_mean", "tpot_ms_median"):
            if steady.get(key):
                out[f"delta_steady_{key}_server_vs_internal_pct"] = (steady[key] - internal) / internal * 100.0
        out["confounds"] = [
            "server runs the overlap scheduler, radix cache and chunked prefill; the bench is non-overlap with radix off",
            "server requests start and finish at different times (prefill interleaves with decode, effective concurrency < C at the tail); the bench decodes one fixed batch",
            "client TPOT excludes the first token; internal TPOT amortizes over all useful tokens of the fixed batch",
            "prefix KV: both come from real prefill, of different random token ids",
            "simulated acceptance on both sides (SGLANG_SIMULATE_ACC_LEN=3.51), drawn from different generator states",
        ]
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
