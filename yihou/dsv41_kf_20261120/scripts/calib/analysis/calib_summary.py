#!/usr/bin/env python3
"""Gate summary of one calibrated-profiler server window: reads the calib_table.py JSON, the smoke
validate-window.json and the server config, writes summary.json with the numbers RUNPLAN's stop conditions
decide on.

Usage: calib_summary.py TABLE_JSON VALIDATE_WINDOW_JSON --config SERVER_CONFIG_JSON --out summary.json
Gates (RUNPLAN S1/S3 + the task's success bar):
  * expected_ranks = tensor-parallel-size x data-parallel-size of the server config; every one must have closed
    a window (ranks == expected_ranks) with >= CALIB_STEPS VERIFY steps;
  * (unattributed ms + outside-forward ms) / VERIFY kernel ms <= 5% per rank (durations, not counts);
  * causality violations == 0; no window dispatch without a kernel symbol; smoke mode window PASS.
"""
import argparse
import json
from pathlib import Path

MAX_UNASSIGNED_FRAC = 0.05


def summarize(ranks, smoke, config, config_name):
    args = config.get("args", {})
    expected = int(args.get("tensor-parallel-size", 1)) * int(args.get("data-parallel-size", 1))
    want_steps = int(config.get("env", {}).get("CALIB_STEPS", 20))
    per_rank, worst_frac, viol = [], 0.0, 0
    for r in ranks:
        if "unattributed_ms" not in r:
            raise SystemExit("table has no unattributed_ms: regenerate it with the current calib_table.py")
        v = r["stages"].get("VERIFY", {})
        verify_ms = v.get("kernel_ms_per_step", 0.0) * v.get("steps", 0)
        unassigned_ms = r["unattributed_ms"] + r["other_ms"]
        frac = unassigned_ms / verify_ms if verify_ms > 0 else float("inf")
        worst_frac = max(worst_frac, frac)
        viol += r["causality_violations"]
        d = r["stages"].get("DRAFT", {})
        per_rank.append({"pid": r["pid"], "tp_rank": r["tp_rank"], "window_ms": r["window_ms"],
                         "dispatches_in_window": r["dispatches_in_window"],
                         "unattributed_no_api": r["unattributed_no_api"], "unattributed_ms": r["unattributed_ms"],
                         "other_ms": r["other_ms"], "unassigned_frac_of_verify": frac,
                         "verify_steps": v.get("steps", 0), "verify_bs_median": v.get("bs_median"),
                         "verify_gpu_span_ms_per_step": v.get("gpu_span_ms_per_step"),
                         "verify_kernel_ms_per_step": v.get("kernel_ms_per_step"),
                         "verify_buckets_ms_per_step": v.get("buckets_ms_per_step", {}),
                         "draft_kernel_ms_per_step": d.get("kernel_ms_per_step")})
    min_steps = min((x["verify_steps"] for x in per_rank), default=0)
    smoke_pass = 1 if smoke.get("verdict") == "PASS" else 0
    return {
        "config": config_name,
        "expected_ranks": expected,
        "ranks": len(ranks),
        "rank_pids": [x["pid"] for x in per_rank],
        "configured_verify_steps": want_steps,
        "min_verify_steps": min_steps,
        "unattributed_total": sum(x["unattributed_no_api"] for x in per_rank),
        "causality_violations_total": viol,
        "missing_symbol_total": sum(r.get("missing_symbol_in_window", 0) for r in ranks),
        "worst_unattributed_or_other_frac": worst_frac if per_rank else float("inf"),
        "provisional": bool(len(ranks) != expected or min_steps < want_steps or worst_frac > MAX_UNASSIGNED_FRAC
                            or viol > 0 or not smoke_pass or not per_rank
                            or sum(r.get("missing_symbol_in_window", 0) for r in ranks) > 0),
        "smoke_window_pass": smoke_pass,
        "per_rank": per_rank,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("table_json", type=Path)
    p.add_argument("validate_window_json", type=Path)
    p.add_argument("--config", required=True, type=Path, help="server config JSON the window ran with")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    out = summarize(json.loads(a.table_json.read_text()), json.loads(a.validate_window_json.read_text()),
                    json.loads(a.config.read_text()), a.config.name)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_rank"}))


if __name__ == "__main__":
    main()
