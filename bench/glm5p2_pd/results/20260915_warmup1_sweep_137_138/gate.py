#!/usr/bin/env python3
"""Fail-closed check that a launched stack is the point we intended to measure.

Every sweep point asserts conc == max_running == cuda_graph_max_bs on both
roles, the P/D GPU shape, mem_fraction_static, and that both workers run the
pinned image.  A point that silently ran with different limits would land in the
curve as a real measurement, which is how a contaminated image previously cost
this campaign several days.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOPOLOGY = {
    # topology -> (prefill_tp, prefill_dp, decode_tp, decode_dp)
    "p8d8": (8, 8, 8, 8),
    "p4d8": (4, 4, 8, 8),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch-dir", required=True)
    ap.add_argument("--conc", type=int, required=True)
    ap.add_argument("--mem", type=float, required=True)
    ap.add_argument("--topology", required=True, choices=sorted(TOPOLOGY))
    ap.add_argument("--expect-image", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    launch = Path(args.launch_dir)
    p_tp, p_dp, d_tp, d_dp = TOPOLOGY[args.topology]
    errors: list[str] = []
    observed: dict[str, dict] = {}

    for role, want_tp, want_dp in (("prefill", p_tp, p_dp), ("decode", d_tp, d_dp)):
        info_path = launch / "server-info" / f"{role}-0.json"
        if not info_path.is_file():
            errors.append(f"{role}: missing {info_path}")
            continue
        info = json.loads(info_path.read_text())
        got = {
            "tp_size": info.get("tp_size"),
            "dp_size": info.get("dp_size"),
            "max_running_requests": info.get("max_running_requests"),
            "cuda_graph_max_bs_decode": info.get("cuda_graph_max_bs_decode"),
            "mem_fraction_static": info.get("mem_fraction_static"),
            "status": info.get("status"),
        }
        observed[role] = got
        if got["tp_size"] != want_tp:
            errors.append(f"{role}: tp_size {got['tp_size']} != {want_tp}")
        if got["dp_size"] != want_dp:
            errors.append(f"{role}: dp_size {got['dp_size']} != {want_dp}")
        if got["max_running_requests"] != args.conc:
            errors.append(
                f"{role}: max_running_requests {got['max_running_requests']} != conc {args.conc}"
            )
        if got["cuda_graph_max_bs_decode"] != args.conc:
            errors.append(
                f"{role}: cuda_graph_max_bs_decode {got['cuda_graph_max_bs_decode']} != conc {args.conc}"
            )
        if got["mem_fraction_static"] != args.mem:
            errors.append(
                f"{role}: mem_fraction_static {got['mem_fraction_static']} != {args.mem}"
            )
        if got["status"] != "ready":
            errors.append(f"{role}: status {got['status']!r} != 'ready'")

    # launch.sh records the resolved image id per node; both must be the pin.
    log = launch.parent / "logs" / "launch.log"
    image_ids: list[str] = []
    if log.is_file():
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("[launch] image node=") and " id=" in line:
                image_ids.append(line.rsplit(" id=", 1)[1].strip())
    if not image_ids:
        errors.append("no '[launch] image node=... id=' lines found in launch.log")
    for got in image_ids:
        if got != args.expect_image:
            errors.append(f"image {got} != pinned {args.expect_image}")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "topology": args.topology,
        "expected": {
            "conc": args.conc,
            "mem_fraction_static": args.mem,
            "prefill_tp_dp": [p_tp, p_dp],
            "decode_tp_dp": [d_tp, d_dp],
            "image": args.expect_image,
        },
        "observed": observed,
        "observed_image_ids": image_ids,
        "errors": errors,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
