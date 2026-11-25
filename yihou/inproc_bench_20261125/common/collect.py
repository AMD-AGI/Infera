"""Collect every point under an iterations dir into one CSV (one row per point, or per case for forward_bench).
Runs verify_point.check on each; exit 1 if any point fails its gates.
Usage: collect.py <iterations dir> --output summary.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_point import check  # noqa: E402

COLS = ["point", "engine", "tool", "gates", "batch", "isl", "osl", "num_gpus", "case", "prefill_tok_s", "tpot_ms",
        "tok_s", "tok_s_per_gpu", "median_us", "note"]


def rows_for(d: Path):
    res, fails, warns = check(d)
    gates = "PASS" if res is not None and not fails else "FAIL"
    note = "; ".join(fails + warns)
    if res is None:
        return [{"point": d.name, "gates": gates, "note": note}]
    tool = res.get("tool") or res.get("mode")
    base = {"point": d.name, "engine": res.get("engine", ""), "tool": tool, "gates": gates, "note": note}
    if tool == "forward_bench":
        return [{**base, "case": c["case"], "batch": c["num_requests"], "median_us": c["median_us"],
                 "tok_s": (c.get("decode_out_tok_s_per_replica_all_accepted") or c.get("forward_tok_s_per_replica")
                           or c.get("forward_tok_s_per_rank")),  # *_per_rank: pre-rename results
                 "tok_s_per_gpu": c.get("decode_out_tok_s_per_gpu_all_accepted") or c.get("forward_tok_s_per_gpu")}
                for c in res.get("cases", [])]
    if tool == "step_bench":
        return [{**base, "batch": res["batch_size_global"], "isl": res["input_len"], "osl": res["output_len"],
                 "num_gpus": res["num_gpus"], "prefill_tok_s": res["prefill_tok_s"], "tpot_ms": res["tpot_ms"],
                 "tok_s": res["decode_tok_s"], "tok_s_per_gpu": res["decode_tok_s_per_gpu"]}]
    if tool == "engine_step_bench":
        return [{**base, "batch": res["batch_size"], "isl": res["input_len"], "osl": res["output_len"],
                 "num_gpus": res["num_gpus"], "prefill_tok_s": res["prefill_tok_s"], "tpot_ms": res["tpot_ms"],
                 "tok_s": res["steady_tok_s"], "tok_s_per_gpu": res["steady_tok_s_per_gpu"]}]
    return [{**base, "batch": res.get("num"), "isl": res.get("isl"), "osl": res.get("osl"),
             "num_gpus": res.get("num_gpus"), "tok_s": res.get("output_tok_s"),
             "tok_s_per_gpu": res.get("output_tok_s_per_gpu")}]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("iterations")
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    rows = []
    for d in sorted(p for p in Path(a.iterations).iterdir() if p.is_dir()):
        rows += rows_for(d)
    with open(a.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLS})
    bad = [r for r in rows if r.get("gates") != "PASS"]
    print(f"{len(rows)} rows, {len(bad)} failing -> {a.output}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
