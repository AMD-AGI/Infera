#!/usr/bin/env python3
"""Offline latency/SLO analysis for the unprofiled concurrency sweep."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import statistics


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def distribution(values, prefix):
    return {f"{prefix}_{key}": func(values) for key, func in (
        ("mean", statistics.mean), ("p50", lambda x: percentile(x, .5)),
        ("p90", lambda x: percentile(x, .9)), ("p95", lambda x: percentile(x, .95)),
        ("p99", lambda x: percentile(x, .99)), ("max", max))} if values else {}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(data, meta):
    n = meta["num_requests"]
    if data["max_concurrency"] != meta["concurrency"]:
        raise ValueError("Client concurrency mismatch")
    for key in ("input_lens", "output_lens", "ttfts", "itls", "latencies", "start_times", "successes",
                "raw_chunk_gaps", "chunk_token_counts", "errors"):
        if len(data[key]) != n:
            raise ValueError(f"Wrong request count: {key}")
    requests = []
    all_itls, all_gaps, all_chunk_sizes = [], [], []
    for i in range(n):
        success = data["successes"][i]
        output = data["output_lens"][i]
        ttft, latency = data["ttfts"][i], data["latencies"][i]
        if success and (data["input_lens"][i] != 10000 or output != 500 or latency <= ttft):
            raise ValueError(f"Invalid successful request: {i}")
        tpot_ms = (latency - ttft) / (output - 1) * 1000 if success else None
        rate = 1000 / tpot_ms if success else 0
        e2e_rate = output / latency if success else 0
        itls = [value * 1000 for value in data["itls"][i]]
        gaps = [value * 1000 for value in data["raw_chunk_gaps"][i]]
        sizes = data["chunk_token_counts"][i]
        if len(gaps) != len(sizes) or sum(sizes) != len(itls):
            raise ValueError("SSE gap/token bookkeeping mismatch")
        if success:
            all_itls.extend(itls)
            all_gaps.extend(gaps)
            all_chunk_sizes.extend(sizes)
        requests.append({"request_index": i, "success": success, "input_tokens": data["input_lens"][i],
            "output_tokens": output, "start_time_monotonic_s": data["start_times"][i],
            "e2e_ms": latency * 1000, "ttft_ms": ttft * 1000, "tpot_ms": tpot_ms,
            "decode_tokens_s": rate, "e2e_tokens_s": e2e_rate,
            "pass_70": success and rate >= 70, "pass_80": success and rate >= 80,
            "itl_p50_ms": percentile(itls, .5), "itl_p90_ms": percentile(itls, .9),
            "itl_p99_ms": percentile(itls, .99), "raw_chunk_gap_p90_ms": percentile(gaps, .9),
            "raw_chunk_gap_p99_ms": percentile(gaps, .99), "itl_samples": len(itls)})
    good = [r for r in requests if r["success"]]
    if len(good) != data["completed"]:
        raise ValueError("Success count mismatch")
    tpot = [r["tpot_ms"] for r in good]
    # Cross-check the independent per-request calculation against native metrics.
    for key, q in (("median_tpot_ms", .5), ("p90_tpot_ms", .9), ("p99_tpot_ms", .99)):
        if not math.isclose(percentile(tpot, q), data[key], rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError(f"Native TPOT metric mismatch: {key}")
    for key, q in (("median_itl_ms", .5), ("p90_itl_ms", .9), ("p99_itl_ms", .99)):
        if not math.isclose(percentile(all_itls, q), data[key], rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError(f"Native ITL metric mismatch: {key}")
    row = {"point": meta["point"], "concurrency": meta["concurrency"], "repeat": meta["repeat"],
        "num_requests": n, "completed": len(good), "errors": n - len(good),
        "duration_s": data["duration"], "output_tokens_s": data["output_throughput"],
        "request_throughput": data["request_throughput"], "observed_client_concurrency": data["concurrency"],
        "native_accept_length_cumulative": data.get("accept_length")}
    row.update(distribution(tpot, "tpot_ms"))
    row.update(distribution(all_itls, "itl_ms"))
    row.update(distribution([r["ttft_ms"] for r in good], "ttft_ms"))
    row.update(distribution([r["e2e_ms"] for r in good], "e2e_ms"))
    row.update(distribution(all_gaps, "raw_chunk_gap_ms"))
    row["mean_tokens_per_chunk_after_first"] = statistics.mean(all_chunk_sizes)
    row["decode_tokens_s_p50"] = percentile([r["decode_tokens_s"] for r in good], .5)
    row["decode_tokens_s_p90"] = percentile([r["decode_tokens_s"] for r in good], .9)
    row["decode_tokens_s_p10"] = percentile([r["decode_tokens_s"] for r in good], .1)
    row["e2e_tokens_s_p50"] = percentile([r["e2e_tokens_s"] for r in good], .5)
    for target in (70, 80):
        passed = [r for r in requests if r[f"pass_{target}"]]
        row[f"slo_{target}_pass_requests"] = len(passed)
        row[f"slo_{target}_pass_fraction"] = len(passed) / n
        row[f"slo_{target}_goodput_tokens_s"] = sum(r["output_tokens"] for r in passed) / data["duration"]
        row[f"slo_{target}_goodput_requests_s"] = len(passed) / data["duration"]
        row[f"slo_{target}_p50_pass"] = row["tpot_ms_p50"] <= 1000 / target
        row[f"slo_{target}_p90_pass"] = row["tpot_ms_p90"] <= 1000 / target
        row[f"itl_{target}_p90_pass"] = row["itl_ms_p90"] <= 1000 / target
    return row, requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, help="Combined analysis destination; required with multiple experiments")
    args = parser.parse_args()
    runs = [path.resolve() for path in args.run_dirs]
    if len(runs) > 1 and args.output is None:
        parser.error("Use --output when combining experiments")
    out = args.output or runs[0] / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    environments = {}
    for run in runs:
        environment = json.loads((run / "state/environment.json").read_text())
        environments[run.name] = environment
        for directory in sorted((run / "rounds").iterdir()):
            if not (directory / "observations.json").exists():
                continue  # Exclude incomplete rounds and the server startup folder.
            path = directory / "benchmark.jsonl"
            if path.exists():
                data = json.loads(path.read_text())
            else:
                with gzip.open(path.with_suffix(".jsonl.gz"), "rt") as stream:
                    data = json.load(stream)
            row, requests = summarize(data, json.loads((directory / "run.json").read_text()))
            obs = json.loads((directory / "observations.json").read_text())
            row.update({"all_sampled_cuda_graph": obs["all_samples_cuda_graph"],
                        "max_sampled_retractions": max(obs["sampled_retractions"], default=0),
                        "experiment": run.name, "server_max_running": environment["max_running_requests"],
                        "mem_fraction_static": environment.get("mem_fraction_static", 0.85)})
            rows.append(row)
            # A combined report leaves each historical experiment's derived
            # files/checksums intact. Single-experiment analysis writes details.
            if len(runs) == 1:
                write_csv(directory / "requests.csv", requests)
                (directory / "metrics.json").write_text(json.dumps(row, indent=2) + "\n")
    rows.sort(key=lambda r: (r["concurrency"], r["point"]))
    write_csv(out / "metrics.csv", rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["concurrency"]].append(row)
    decisions = {}
    for target in (70, 80):
        decisions[str(target)] = {"target_tokens_s": target, "tpot_limit_ms": 1000 / target}
        for q in ("p50", "p90"):
            candidates = [c for c, rs in groups.items() if all(r[f"slo_{target}_{q}_pass"] and r["errors"] == 0 for r in rs)]
            decisions[str(target)][f"max_tested_concurrency_{q}_all_rounds"] = max(candidates, default=None)
        candidates = [c for c, rs in groups.items() if all(r[f"slo_{target}_pass_fraction"] >= .9 and r["errors"] == 0 for r in rs)]
        decisions[str(target)]["max_tested_concurrency_90pct_requests_all_rounds"] = max(candidates, default=None)
        candidates = [c for c, rs in groups.items() if all(r[f"itl_{target}_p90_pass"] and r["errors"] == 0 for r in rs)]
        decisions[str(target)]["max_tested_concurrency_itl_p90_all_rounds"] = max(candidates, default=None)
    result = {"definition": "Per-request decode rate=(output_tokens-1)/(request_latency-TTFT); SLA goodput counts only tokens in requests that meet this rate threshold.",
        "decode_rate_percentile_definition": "Decode rate P90 is the 90th percentile of successful per-request rates in ascending order, using linear interpolation. The rate met by at least 90% of requests is described by rate P10 (approximately 1000 / TPOT P90 in ms).",
        "itl_definition": "Native SGLang ITL distributes an SSE chunk gap equally among new tokens in that chunk; raw chunk gaps are separately retained.",
        "limits": ["Fake KV and simulated acceptance; no real prefill throughput or semantic correctness claim.",
                   "Server capacities and graph buckets are recorded per experiment; concurrency is client cap, not a constant batch guarantee.",
                   "Native accept_length is cumulative over server lifetime; it is not an independently differenced per-round acceptance rate.",
                   "Maxima are among tested points; P90 is one explicit selection rule, not an implicit user-specified SLA."],
        "experiments": environments, "points": rows, "decisions": decisions}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Unprofiled concurrency sweep", "", result["definition"], "", result["decode_rate_percentile_definition"], "", result["itl_definition"], "",
        "| Point | Conc | Requests | Output tok/s | TPOT P50/P90 ms | ITL P50/P90/P99 ms | Decode tok/s P50/P90 | ≥70 / ≥80 pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['point']} | {r['concurrency']} | {r['completed']}/{r['num_requests']} | {r['output_tokens_s']:.2f} | "
            f"{r['tpot_ms_p50']:.3f} / {r['tpot_ms_p90']:.3f} | {r['itl_ms_p50']:.3f} / {r['itl_ms_p90']:.3f} / {r['itl_ms_p99']:.3f} | "
            f"{r['decode_tokens_s_p50']:.2f} / {r['decode_tokens_s_p90']:.2f} | {r['slo_70_pass_fraction']:.1%} / {r['slo_80_pass_fraction']:.1%} |")
    lines += ["", "## Maximum tested concurrency", "", "All repeats must pass the selected criterion.", "",
              "| Target | TPOT limit | P50 criterion | P90 criterion | ≥90% requests meet target |", "|---|---:|---:|---:|---:|"]
    for target, d in decisions.items():
        lines.append(f"| {target} tok/s | {d['tpot_limit_ms']:.4f} ms | {d['max_tested_concurrency_p50_all_rounds']} | "
                     f"{d['max_tested_concurrency_p90_all_rounds']} | {d['max_tested_concurrency_90pct_requests_all_rounds']} |")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"completed_rounds": len(rows), "decisions": decisions}, indent=2))


if __name__ == "__main__":
    main()
