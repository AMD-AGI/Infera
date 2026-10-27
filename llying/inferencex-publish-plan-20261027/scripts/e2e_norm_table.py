#!/usr/bin/env python3
"""PR #210 Pareto points under both interactivity definitions.

InferenceX's AgentX dashboard draws its canonical frontier against
E2E normalized interactivity P90 (1 / p90 of per-request E2EL / OSL), which
includes TTFT. The PR plotted decode interactivity P90. This prints both for
every point and marks the points that stay non-dominated under each.

Usage (from llying/): python3 inferencex-publish-plan-20261027/scripts/e2e_norm_table.py
"""

from __future__ import annotations

import glob
import json
import os

RESULTS = os.path.join(os.path.dirname(__file__), "../../2p1d-v0522-20261020/results")
POINTS = [
    ("2P1D C176", "pp4-ck8k-2p-c176", 16),
    ("1P + 4-GPU decode C80", "pp4-ck8k-d4-c80", 8),
    ("1P + 4-GPU decode C64", "pp4-ck8k-d4-c64", 8),
    ("1P1D balanced C88", "pp4-ck8k-rb-c88", 12),
    ("1P1D balanced C64", "ex-rb-c64", 12),
    ("1P1D balanced C48", "ex-rb-c48", 12),
    ("1P1D balanced C32", "ex-rb-c32", 12),
]


def frontier(rows: list[dict], x: str) -> set[str]:
    keep = set()
    for r in rows:
        if not any(
            o is not r and o[x] >= r[x] and o["tput"] >= r["tput"] and (o[x] > r[x] or o["tput"] > r["tput"])
            for o in rows
        ):
            keep.add(r["name"])
    return keep


def main() -> None:
    rows = []
    for name, d, gpus in POINTS:
        path = glob.glob(os.path.join(RESULTS, d, "agentx_conc*.json"))[0]
        j = json.load(open(path))
        lat = j["request_metrics"]["latency"]
        rows.append(
            {
                "name": name,
                "gpus": gpus,
                "tput": j["request_metrics"]["throughput"]["per_gpu"]["total_tput_tps"],
                "intvty": lat["intvty"]["p90"],
                "e2e": lat["e2e_norm_intvty"]["p90"],
                "ttft90": lat["ttft"]["p90"],
            }
        )
    on_old, on_new = frontier(rows, "intvty"), frontier(rows, "e2e")
    print(f"{'point':24s} {'GPUs':>4s} {'tok/s/GPU':>10s} {'intvty p90':>10s} {'E2E-norm p90':>12s} {'TTFT p90':>8s}  frontier(old/new)")
    for r in rows:
        print(
            f"{r['name']:24s} {r['gpus']:4d} {r['tput']:10,.0f} {r['intvty']:10.1f} {r['e2e']:12.1f} "
            f"{r['ttft90']:8.1f}  {'Y' if r['name'] in on_old else '-'}/{'Y' if r['name'] in on_new else '-'}"
        )


if __name__ == "__main__":
    main()
