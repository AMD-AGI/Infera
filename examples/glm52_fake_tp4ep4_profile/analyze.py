#!/usr/bin/env python3
"""Summarize Chrome/Kineto traces without counting CPU launches as GPU work.

Category percentages use summed GPU kernel duration, not end-to-end latency.
Graph kernels are attributed to speculative stages only through correlation IDs
or external IDs. Unlinked events remain unattributed; GPU/CPU timestamp overlap
is deliberately not used to guess asynchronous execution ownership.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import re

RULES = [
    ("Communication / fused allreduce", r"allreduce|all_reduce|allgather|all_gather|reduce_scatter|reducescatter|cross_device_reduce|local_device_load_rmsnorm|nccl|rccl|msccl|quickreduce"),
    ("MoE expert GEMM", r"fmoe|fused_moe|moe[12]_|moe.*gemm|gemm.*moe"),
    ("MoE routing / sorting", r"moe|routing|router|softmax.*topk|topk.*softmax"),
    ("DSA index / top-k", r"mqa_logits|indexer|index_key|index_k|topk|top_k|top_k_per_row|lightning_index|dsa.*metadata|hadamard"),
    ("Attention / MLA", r"mla|fmha|flash_attn|flash_attention|attn|paged_attention|decode_attention|sparse_attention"),
    ("Dense GEMM / linear", r"gemm|matmul|gemv|cublas|hipblas|hgemm|sgemm|wgmma|mm_kernel|^Cijk_"),
    ("Norm / RoPE / KV write", r"rmsnorm|rms_norm|layernorm|layer_norm|qk_norm|rope|rotary|kvcache|kv_cache|cache_quant|store_kv|kn_entry.*OpCachedFwd"),
    ("Quantization / cast", r"quant|dequant|cast|convert|float_to|fp8|fp4"),
    ("Sampling / speculative bookkeeping", r"sample|sampling|speculat|eagle|accept|verify_tree|verifytree|build_tree|tree_spec|draft|renorm|logits|assign_extend|compute_position|alloc_extend|assign_req_to_token|_get_last_loc_safe"),
    ("Tensor / elementwise / indexing", r"elementwise|vectorized|index|gather|scatter|copy|fill|arange|cat_|concat|cumsum|softmax|reduce|scan|embedding|transpose|permute|silu|sigmoid|add|mul"),
]
STAGES = {"draft", "verify", "draft_extend"}


def stage_name(name):
    if name in STAGES:
        return name
    if re.fullmatch(r"step\[TARGET_VERIFY bs=\d+\]", name):
        return "verify"
    return None


def classify(name, aliases=None):
    if aliases and name in aliases:
        return aliases[name]
    for category, pattern in RULES:
        if re.search(pattern, name, re.I):
            return category
    return "Other / unclassified"


def union_duration(intervals):
    total = 0.0
    end = None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(groups):
    total = sum(sum(values) for values in groups.values())
    return [{"name": name, "count": len(values), "total_us": sum(values),
             "percent_kernel_time": sum(values) * 100 / total if total else 0,
             "mean_us": sum(values) / len(values), "p50_us": percentile(values, .5),
             "p90_us": percentile(values, .9), "max_us": max(values)}
            for name, values in sorted(groups.items(), key=lambda item: -sum(item[1]))]


def analyze_trace(path, aliases=None):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        data = json.load(stream)
    events = data["traceEvents"] if isinstance(data, dict) else data
    complete = [e for e in events if e.get("ph") == "X" and e.get("dur", 0) > 0]
    kernels = [e for e in complete if e.get("cat", "").lower() in ("kernel", "gpu_kernel")]
    if not kernels:
        raise ValueError(f"No GPU kernels in {path}; check HIP graph tracing and profiler activities")
    annotations = [e for e in complete if e.get("cat") == "user_annotation" and stage_name(e.get("name", ""))]
    spans = defaultdict(list)
    for event in annotations:
        spans[(event.get("pid"), event.get("tid"))].append(event)
    span_starts = {}
    for key in spans:
        spans[key].sort(key=lambda e: e["ts"])
        span_starts[key] = [e["ts"] for e in spans[key]]

    def stage_for_cpu(event):
        if event is None:
            return "unattributed"
        key = (event.get("pid"), event.get("tid"))
        index = bisect_right(span_starts.get(key, []), event["ts"]) - 1
        if index >= 0:
            span = spans[key][index]
            if event["ts"] < span["ts"] + span["dur"]:
                return stage_name(span["name"])
        return "unattributed"

    runtime_by_correlation = {}
    cpu_by_external_id = {}
    for event in complete:
        cat = event.get("cat", "").lower()
        args = event.get("args", {})
        if cat in ("cuda_runtime", "hip_runtime", "cuda_driver") and "correlation" in args:
            runtime_by_correlation[args["correlation"]] = event
        if cat == "cpu_op" and "External id" in args:
            cpu_by_external_id[args["External id"]] = event
    categories, names, stages = defaultdict(list), defaultdict(list), defaultdict(list)
    for event in kernels:
        name, duration = event["name"], event["dur"]
        categories[classify(name, aliases)].append(duration)
        names[name].append(duration)
        args = event.get("args", {})
        stage = stage_for_cpu(runtime_by_correlation.get(args.get("correlation")))
        if stage == "unattributed":
            stage = stage_for_cpu(cpu_by_external_id.get(args.get("External id")))
        stages[stage].append(duration)
    devices = defaultdict(list)
    for event in kernels:
        devices[event.get("args", {}).get("device", event.get("pid"))].append((event["ts"], event["ts"] + event["dur"]))
    device_stats = [{"device": device, "kernel_busy_us": union_duration(intervals),
                     "kernel_window_us": max(end for _, end in intervals) - min(start for start, _ in intervals)}
                    for device, intervals in devices.items()]
    rank = re.search(r"TP-(\d+)", path.name)
    kernel_correlations = Counter(e.get("args", {}).get("correlation") for e in kernels)
    graph_launches = [e for e in complete if e.get("name", "").lower() == "hipgraphlaunch"]
    graph_coverage = [{"correlation": e.get("args", {}).get("correlation"),
                       "stage": stage_for_cpu(e),
                       "kernel_count": kernel_correlations[e.get("args", {}).get("correlation")],
                       "ts": e["ts"]} for e in graph_launches]
    return {"file": str(path), "rank": int(rank[1]) if rank else None,
            "kernel_count": len(kernels), "kernel_sum_us": sum(e["dur"] for e in kernels),
            "devices": device_stats,
            "stage_annotation_counts": dict(Counter(stage_name(e["name"]) for e in annotations)),
            "verify_batch_sizes": dict(Counter(re.search(r"bs=(\d+)", e["name"])[1]
                for e in annotations if re.search(r"bs=(\d+)", e["name"]))),
            "graph_kernel_coverage": graph_coverage,
            "graph_launches_without_kernels": sum(e["kernel_count"] == 0 for e in graph_coverage),
            "graph_launch_count": sum("graphlaunch" in e.get("name", "").lower() for e in complete),
            "event_categories": dict(Counter(e.get("cat", "") for e in complete)),
            "categories": summarize(categories), "kernels": summarize(names), "stages": summarize(stages)}, (categories, names, stages)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Diagnostic only: permit graph launches without GPU kernels")
    parser.add_argument("--kernel-map", type=Path,
                        help="Model-specific aliases for opaque kernel names, with provenance")
    args = parser.parse_args()
    mapping = json.loads(args.kernel_map.read_text()) if args.kernel_map else {}
    aliases = mapping.get("aliases", {})
    files = []
    for path in args.traces:
        files.extend(sorted(path.rglob("*.trace.json*")) if path.is_dir() else [path])
    files = sorted(set(p.resolve() for p in files))
    if not files:
        parser.error("No trace files found")
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    aggregate = [defaultdict(list) for _ in range(3)]
    for path in files:
        report, groups = analyze_trace(path, aliases)
        if report["graph_launches_without_kernels"] and not args.allow_incomplete:
            raise RuntimeError(f"{path}: {report['graph_launches_without_kernels']} graph launches have no GPU kernels; "
                               "use traceable HIP graph mode, or --allow-incomplete for diagnosis only")
        reports.append(report)
        for target, source in zip(aggregate, groups):
            for name, durations in source.items():
                target[name].extend(durations)
        print(f"{path.name}: {report['kernel_count']} kernels, {report['kernel_sum_us']/1000:.1f} ms summed GPU time", flush=True)
    result = {"denominator": "Sum of GPU kernel durations across selected ranks and traces; not wall-clock latency",
              "classification": "Ordered kernel-name patterns; fused kernels stay in one category; audit kernels.csv",
              "rules": RULES, "kernel_alias_map": mapping, "traces": reports}
    for key, groups in zip(("categories", "kernels", "stages"), aggregate):
        result[key] = summarize(groups)
        if key == "kernels":
            for row in result[key]:
                row["category"] = classify(row["name"], aliases)
        write_csv(args.output / f"{key}.csv", result[key])
    rank_rows, category_rows, stage_rows = [], [], []
    round_totals, round_categories = defaultdict(float), defaultdict(lambda: defaultdict(float))
    for report in reports:
        capture = Path(report["file"]).parents[1].name
        steps = report["stage_annotation_counts"].get("verify", 0)
        rank_rows.append({"round": capture, "tp_rank": report["rank"], "mtp_steps": steps,
                          "kernel_count": report["kernel_count"],
                          "gpu_kernel_sum_ms": report["kernel_sum_us"] / 1000,
                          "gpu_kernel_ms_per_step": report["kernel_sum_us"] / steps / 1000 if steps else None,
                          "kernel_busy_union_ms": sum(d["kernel_busy_us"] for d in report["devices"]) / 1000,
                          "first_to_last_kernel_ms": max(d["kernel_window_us"] for d in report["devices"]) / 1000,
                          "missing_graph_launches": report["graph_launches_without_kernels"]})
        round_totals[capture] += report["kernel_sum_us"]
        for key, rows in (("categories", category_rows), ("stages", stage_rows)):
            for row in report[key]:
                rows.append({"round": capture, "tp_rank": report["rank"], **row})
                if key == "categories":
                    round_categories[capture][row["name"]] += row["total_us"]
    for key, rows in (("rank_summary", rank_rows), ("category_by_rank", category_rows), ("stage_by_rank", stage_rows)):
        write_csv(args.output / f"{key}.csv", rows)
    by_round = {capture: {name: duration / round_totals[capture] * 100 for name, duration in groups.items()}
                for capture, groups in round_categories.items()}
    (args.output / "category_by_round.json").write_text(json.dumps(by_round, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# GPU kernel distribution", "", result["denominator"] + ".", "",
             "| Category | GPU time (%) | Summed GPU ms | Calls |", "|---|---:|---:|---:|"]
    for row in result["categories"]:
        lines.append(f"| {row['name']} | {row['percent_kernel_time']:.2f} | {row['total_us']/1000:.2f} | {row['count']} |")
    lines += ["", "## Speculative stage attribution", "", "Only ID-linked GPU events are attributed. CPU ranges are not GPU durations.", "",
              "| Stage | GPU time (%) | Summed GPU ms |", "|---|---:|---:|"]
    for row in result["stages"]:
        lines.append(f"| {row['name']} | {row['percent_kernel_time']:.2f} | {row['total_us']/1000:.2f} |")
    lines += ["", "## Top kernels", "", "| Kernel | GPU time (%) | Mean us | Calls |", "|---|---:|---:|---:|"]
    for row in result["kernels"][:20]:
        # Full mangled/template names remain in CSV/JSON; keep the report readable.
        name = row['name'][:140].replace('|', '\\|')
        if len(row['name']) > 140:
            name += "…"
        lines.append(f"| `{name}` | {row['percent_kernel_time']:.2f} | {row['mean_us']:.2f} | {row['count']} |")
    (args.output / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
