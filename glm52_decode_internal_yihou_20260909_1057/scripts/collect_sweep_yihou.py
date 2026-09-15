#!/usr/bin/env python3
"""Collect complete TP4/EP4 sweep results without treating smoke as success."""
import argparse
import csv
import json
from pathlib import Path


def collect(root):
    rows = []
    for dpa in (False, True):
        for concurrency in (4, 8, 16, 20, 24):
            run = root / f"full_dpa_{'on' if dpa else 'off'}_c{concurrency}_yihou"
            row = {"dpa": "on" if dpa else "off", "concurrency": concurrency, "status": "missing"}
            result_path = run / "result_yihou.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                expected = concurrency * 10000
                if not result["complete"] or result["useful_output_tokens"] != expected:
                    row["status"] = "incomplete"
                elif result["batch_size"] != concurrency or result["input_len"] != 70000 or result["output_len"] != 10000:
                    row["status"] = "configuration_mismatch"
                elif result.get("ep_size") != 4 or result.get("tp_size") != 4 or result.get("dp_size") != (4 if dpa else 1):
                    row["status"] = "topology_mismatch"
                else:
                    row.update(status="pass", decode_seconds=result["elapsed_seconds"],
                               output_tokens_per_second=result["output_tokens_per_second"],
                               tpot_ms=result["effective_token_latency_ms_per_user"],
                               realized_accept_length=result["realized_accept_length"],
                               verify_iterations=result["verify_iterations"])
                    launch = run / "launch_status.json"
                    if launch.exists():
                        status = json.loads(launch.read_text())
                        row["launch_wall_seconds"] = status["launch_wall_seconds"]
                        if status["exit_code"] != 0:
                            row["status"] = "process_failed"
                    else:
                        row["status"] = "process_unconfirmed"
            rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("iterations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.iterations)
    fields = ["dpa", "concurrency", "status", "tpot_ms", "output_tokens_per_second", "decode_seconds", "launch_wall_seconds", "realized_accept_length", "verify_iterations"]
    with args.output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    raise SystemExit(0 if all(row["status"] == "pass" for row in rows) else 1)
