#!/usr/bin/env python3
"""Collect TP4/DPA high-concurrency points. A gate, not a formatter.

Every row must prove: the run completed, it emitted exactly C x OSL useful
tokens, the resolved topology is the requested one, and the launching process
exited 0. Anything else is reported with the reason and fails the exit code.
"""
import argparse
import csv
import json
import re
from pathlib import Path

# "_mrr" marks a run that passed --max-running-requests explicitly. Runs without it
# are the pre-fix attempts, kept as evidence of the 48-request speculative-decoding cap.
NAME = re.compile(
    r"^hc_ep(?P<ep>\d+)_c(?P<c>\d+)_mf(?P<mf>0p\d+)(?P<mrr>_mrr)?(?P<xs>_xs)?_yihou$")
INPUT_LEN = 70000
OUTPUT_LEN = 10000
TP_SIZE = 4


def classify(run, ep_size, concurrency):
    result_path = run / "result_yihou.json"
    if not result_path.exists():
        # Distinguish "never launched" from "launched and died before writing a result".
        launch = run / "launch_status.json"
        if launch.exists() and json.loads(launch.read_text())["exit_code"] != 0:
            return {"status": "process_failed_no_result",
                    "launch_wall_seconds": json.loads(launch.read_text())["launch_wall_seconds"]}
        return {"status": "missing"}
    result = json.loads(result_path.read_text())
    if not result["complete"] or result["useful_output_tokens"] != concurrency * OUTPUT_LEN:
        return {"status": "incomplete"}
    if (result["batch_size"], result["input_len"], result["output_len"]) != (
            concurrency, INPUT_LEN, OUTPUT_LEN):
        return {"status": "configuration_mismatch"}
    if (result.get("ep_size"), result.get("tp_size"), result.get("dp_size"),
            result.get("enable_dp_attention")) != (ep_size, TP_SIZE, TP_SIZE, True):
        return {"status": "topology_mismatch"}
    row = {
        "status": "pass",
        "local_batch": result["local_batch_size"],
        "tpot_ms": result["effective_token_latency_ms_per_user"],
        "output_tokens_per_second": result["output_tokens_per_second"],
        "decode_seconds": result["elapsed_seconds"],
        "realized_accept_length": result["realized_accept_length"],
        "verify_iterations": result["verify_iterations"],
        "kv_pool_tokens": result["memory"]["reserved_tokens"],
    }
    launch = run / "launch_status.json"
    if not launch.exists():
        row["status"] = "process_unconfirmed"
        return row
    status = json.loads(launch.read_text())
    row["launch_wall_seconds"] = status["launch_wall_seconds"]
    if status["exit_code"] != 0:
        row["status"] = "process_failed"
    return row


def collect(root, node):
    rows = []
    for run in sorted(root.iterdir()):
        match = NAME.match(run.name)
        if not run.is_dir() or not match:
            continue
        ep_size = int(match.group("ep"))
        concurrency = int(match.group("c"))
        row = {"node": node, "ep_size": ep_size, "concurrency": concurrency,
               "mem_fraction_static": float(match.group("mf").replace("p", ".")),
               "max_running_requests_set": bool(match.group("mrr")),
               "expandable_segments": bool(match.group("xs")),
               "iteration": run.name}
        row.update(classify(run, ep_size, concurrency))
        rows.append(row)
    rows.sort(key=lambda row: (row["ep_size"], row["concurrency"], row["mem_fraction_static"]))
    return rows


FIELDS = ["node", "ep_size", "concurrency", "mem_fraction_static",
          "max_running_requests_set", "expandable_segments", "status", "local_batch",
          "tpot_ms", "output_tokens_per_second", "decode_seconds", "launch_wall_seconds",
          "kv_pool_tokens", "realized_accept_length", "verify_iterations", "iteration"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("iterations", type=Path, nargs="+",
                        help="one or more iterations directories, named ..._<node>")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for root in args.iterations:
        rows.extend(collect(root, root.name.rsplit("_", 1)[-1]))
    if not rows:
        raise SystemExit("No high-concurrency iteration directories found")
    with args.output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    raise SystemExit(0 if all(row["status"] == "pass" for row in rows) else 1)
