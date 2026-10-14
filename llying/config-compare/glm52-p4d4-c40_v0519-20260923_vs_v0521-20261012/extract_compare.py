#!/usr/bin/env python3
"""Regenerate the machine-readable side of this comparison from the two pack-ups.

Inputs (repo-relative, read-only):
  yihou/glm52.samenode-triton-idxshare.c40-full.packup_20260923          (v0.5.19 baseline)
  yihou/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012          (v0.5.21, 22k)

Outputs (next to this script):
  server_args_diff.csv   every resolved ServerArgs field that differs per leg, or exists in one version only
  metrics.csv            AgentX result metrics of the baseline and the v0.5.21 control/headline runs
  harness.diff           unified diff of the vendored bench-harness (old -> new)

Usage: python3 extract_compare.py
"""
import ast
import csv
import difflib
import gzip
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OLD = REPO / "yihou/glm52.samenode-triton-idxshare.c40-full.packup_20260923"
NEW = REPO / "yihou/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012"

# Runtime-only or per-deployment identity fields: they always differ and carry no config meaning.
IDENTITY = {
    "host", "port", "random_seed", "disaggregation_bootstrap_port", "kv_events_config",
    "internal_states", "launch_command", "scheduler_pids", "startup_time", "status", "version",
}

# docker splits long log lines at 16 KiB and prefixes each fragment with its own timestamp.
DOCKER_TS = re.compile(r"\n?\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+Z ")


def server_args_from_log(path):
    with gzip.open(path, "rt", errors="replace") as f:
        text = f.read()
    start = text.index("server_args={") + len("server_args=")
    body = DOCKER_TS.sub("", text[start:start + 400_000])
    depth = 0
    for i, ch in enumerate(body):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            return ast.literal_eval(body[: i + 1])
    raise ValueError(f"unterminated server_args in {path}")


def render(value):
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)


def write_server_args_diff():
    legs = {
        "prefill": (json.loads((OLD / "results/server-info.29001.json").read_text()),
                    server_args_from_log(NEW / "logs/e2g/prefill-0.log.gz")),
        "decode": (json.loads((OLD / "results/server-info.29257.json").read_text()),
                   server_args_from_log(NEW / "logs/e2g/decode-0.a1.log.gz")),
    }
    rows = []
    for leg, (old, new) in legs.items():
        for key in sorted(set(old) | set(new)):
            if key in IDENTITY:
                continue
            if key not in new:
                rows.append((leg, key, render(old[key]), "<absent>", "only-v0519"))
            elif key not in old:
                rows.append((leg, key, "<absent>", render(new[key]), "only-v0521"))
            elif render(old[key]) != render(new[key]):
                rows.append((leg, key, render(old[key]), render(new[key]), "changed"))
    with open(HERE / "server_args_diff.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["leg", "key", "v0519_baseline", "v0521_e2g", "status"])
        w.writerows(rows)
    return len(rows)


RUNS = [
    ("v0519-baseline-fresh", OLD / "results/agentx_conc40.json"),
    ("v0521-e1c-fresh-relaxed", NEW / "results/other-rounds/e1c-agentx-c40-r2/agentx_conc40.json"),
    ("v0521-e2d-fresh-idxshare-true", NEW / "results/other-rounds/e2d-agentx-c40/agentx_conc40.json"),
    ("v0521-e2g-A1-fresh-target", NEW / "results/e2g/A1-first-run/agentx_conc40.json"),
    ("v0521-e2c-C40-warm-target", NEW / "results/e2c-C40-warm/agentx_conc40.json"),
    ("v0521-e2g-A2-warm-target", NEW / "results/e2g/A2-scored/agentx_conc40.json"),
]

METRICS = [
    ("total_tok_s", ("request_metrics", "throughput", "total", "tokens_per_second")),
    ("tok_s_per_chip", ("request_metrics", "throughput", "per_gpu", "total_tput_tps")),
    ("input_tok_s", ("request_metrics", "throughput", "input", "tokens_per_second")),
    ("output_tok_s", ("request_metrics", "throughput", "output", "tokens_per_second")),
    ("duration_s", ("request_metrics", "throughput", "duration_seconds")),
    ("requests_ok", ("num_requests_successful",)),
    ("error_dropped", ("request_accounting", "records_error_dropped")),
    ("ttft_mean_s", ("request_metrics", "latency", "ttft", "mean")),
    ("ttft_p50_s", ("request_metrics", "latency", "ttft", "p50")),
    ("ttft_p90_s", ("request_metrics", "latency", "ttft", "p90")),
    ("ttft_p95_s", ("request_metrics", "latency", "ttft", "p95")),
    ("tpot_mean_s", ("request_metrics", "latency", "tpot", "mean")),
    ("tpot_p50_s", ("request_metrics", "latency", "tpot", "p50")),
    ("tpot_p95_s", ("request_metrics", "latency", "tpot", "p95")),
    ("e2e_p50_s", ("request_metrics", "latency", "e2el", "p50")),
    ("e2e_p95_s", ("request_metrics", "latency", "e2el", "p95")),
    ("interactivity_p50", ("request_metrics", "latency", "intvty", "p50")),
    ("input_tokens_mean", ("request_metrics", "tokens", "input", "mean")),
    ("output_tokens_mean", ("request_metrics", "tokens", "output_actual", "mean")),
    ("theoretical_cache_hit", ("request_metrics", "cache", "theoretical_cache_hit_rate")),
    ("server_gpu_cache_hit", ("server_metrics", "cache", "gpu_cache_hit_rate")),
    ("server_cpu_cache_hit", ("server_metrics", "cache", "cpu_cache_hit_rate")),
    ("server_overall_cache_hit", ("server_metrics", "cache", "overall_cache_hit_rate")),
    ("prompt_tokens_computed", ("server_metrics", "tokens", "prompt_by_source", "computed")),
    ("kv_cache_pool_tokens", ("kv_cache_pool_tokens",)),
    ("hicache_host_tokens", ("server_metrics", "kv_cache", "cpu_total_tokens")),
]


def dig(obj, path):
    for key in path:
        obj = obj.get(key) if isinstance(obj, dict) else None
    return obj


def write_metrics():
    results = {name: json.loads(path.read_text()) for name, path in RUNS}
    with open(HERE / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + list(results))
        for metric, path in METRICS:
            w.writerow([metric] + [dig(j, path) for j in results.values()])


def write_harness_diff():
    old_root = OLD / "scripts/bench-harness"
    new_root = NEW / "scripts/harness/bench-harness"
    names = sorted({p.relative_to(old_root) for p in old_root.rglob("*") if p.is_file()} |
                   {p.relative_to(new_root) for p in new_root.rglob("*") if p.is_file()})
    chunks = []
    for name in names:
        a, b = old_root / name, new_root / name
        a_lines = a.read_text(errors="replace").splitlines(keepends=True) if a.exists() else []
        b_lines = b.read_text(errors="replace").splitlines(keepends=True) if b.exists() else []
        chunks.extend(difflib.unified_diff(a_lines, b_lines, f"v0519/bench-harness/{name}",
                                           f"v0521/bench-harness/{name}"))
    (HERE / "harness.diff").write_text("".join(chunks))


if __name__ == "__main__":
    print("server_args_diff.csv rows:", write_server_args_diff())
    write_metrics()
    write_harness_diff()
    print("wrote metrics.csv, harness.diff")
