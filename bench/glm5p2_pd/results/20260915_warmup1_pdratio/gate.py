#!/usr/bin/env python3
"""Fail-closed check that a launched stack is the point we intended to measure.

Same contract as the 137/138 sweep gate, widened to N prefill instances: every
prefill worker -- not just prefill-0 -- must carry the topology's TP/DP shape,
conc == max_running == cuda_graph_max_bs, the pinned mem_fraction_static, and
the pinned image id.  A point that silently ran with one worker missing or with
different limits would land in the curve as a real measurement, which is how a
contaminated image previously cost this campaign several days.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOPOLOGY = {
    # topology -> (prefill_instances, prefill_tp, prefill_dp, decode_instances,
    #              decode_tp, decode_dp)
    "p8x2d8": (2, 8, 8, 1, 8, 8),
    "p8x3d8": (3, 8, 8, 1, 8, 8),
    "p8x2d8x2": (2, 8, 8, 2, 8, 8),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch-dir", required=True)
    ap.add_argument("--conc", type=int, required=True)
    # Prefill's running count is pinned at the DP width (8) at every point
    # measured, so its max-running and graph-max-bs above that buy nothing and
    # cost graph memory -- at conc 256 they cost all of it, and the prefill
    # worker died with the KV pool 10% full and HSA reporting 0 MB free.  A
    # point may therefore cap prefill below the concurrency, but it has to say
    # so here: the gate still fails closed, against the declared number, and
    # the number lands in the evidence file rather than in nobody's notes.
    ap.add_argument("--prefill-conc", type=int, default=None)
    ap.add_argument("--mem", type=float, required=True)
    ap.add_argument("--topology", required=True, choices=sorted(TOPOLOGY))
    ap.add_argument("--expect-image", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    launch = Path(args.launch_dir)
    p_n, p_tp, p_dp, d_n, d_tp, d_dp = TOPOLOGY[args.topology]
    errors: list[str] = []
    observed: dict[str, dict] = {}

    prefill_conc = args.prefill_conc if args.prefill_conc else args.conc
    wanted = [("prefill", i, p_tp, p_dp, prefill_conc) for i in range(p_n)]
    wanted += [("decode", i, d_tp, d_dp, args.conc) for i in range(d_n)]
    for role, index, want_tp, want_dp, want_conc in wanted:
        instance = f"{role}-{index}"
        info_path = launch / "server-info" / f"{instance}.json"
        if not info_path.is_file():
            errors.append(f"{instance}: missing {info_path}")
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
        observed[instance] = got
        if got["tp_size"] != want_tp:
            errors.append(f"{instance}: tp_size {got['tp_size']} != {want_tp}")
        if got["dp_size"] != want_dp:
            errors.append(f"{instance}: dp_size {got['dp_size']} != {want_dp}")
        if got["max_running_requests"] != want_conc:
            errors.append(
                f"{instance}: max_running_requests {got['max_running_requests']} != {want_conc}"
            )
        if got["cuda_graph_max_bs_decode"] != want_conc:
            errors.append(
                f"{instance}: cuda_graph_max_bs_decode {got['cuda_graph_max_bs_decode']} != {want_conc}"
            )
        if got["mem_fraction_static"] != args.mem:
            errors.append(
                f"{instance}: mem_fraction_static {got['mem_fraction_static']} != {args.mem}"
            )
        if got["status"] != "ready":
            errors.append(f"{instance}: status {got['status']!r} != 'ready'")

    # A stray extra worker is as wrong as a missing one: it would change the
    # GPU denominator the tok/s/GPU metric divides by.
    present = sorted(p.stem for p in (launch / "server-info").glob("*.json"))
    expected_names = sorted(f"{entry[0]}-{entry[1]}" for entry in wanted)
    if present != expected_names:
        errors.append(f"server-info holds {present}, expected {expected_names}")

    # launch.sh records the resolved image id per node; every one must be the pin.
    log = launch.parent / "logs" / "launch.log"
    image_ids: list[str] = []
    if log.is_file():
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("[launch] image node=") and " id=" in line:
                image_ids.append(line.rsplit(" id=", 1)[1].strip())
    if len(image_ids) != p_n + d_n:
        errors.append(
            f"launch.log has {len(image_ids)} image ids, expected {p_n + d_n}"
        )
    for got in image_ids:
        if got != args.expect_image:
            errors.append(f"image {got} != pinned {args.expect_image}")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "topology": args.topology,
        "expected": {
            "conc": args.conc,
            "prefill_conc": prefill_conc,
            "mem_fraction_static": args.mem,
            "prefill_instances_tp_dp": [p_n, p_tp, p_dp],
            "decode_instances_tp_dp": [d_n, d_tp, d_dp],
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
