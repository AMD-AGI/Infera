#!/usr/bin/env python3
"""Compare the P:D ratio points against the 1P1D baseline on one set of axes.

Throughput per GPU is the campaign's metric, but it cannot answer whether an
extra prefill worker helped: the denominator grows with the worker. So the
absolute total is carried alongside it, and the bottleneck columns come from
the run-time Prometheus samples rather than from reading the curve backwards.

Only samples with a decode actually running are counted, which is what drops
the warmup and drain phases (same rule as the first sweep's report).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1]
BASELINE = RESULTS / "20260915_warmup1_sweep_137_138" / "p8d8"
CAMPAIGN = Path(__file__).resolve().parent

# (label, point dir, prefill endpoint names, decode endpoint names, GPU counts)
BASE_POINTS = [("1P1D", BASELINE / f"c{c}", ["prefill"], ["decode"], 8, 8)
               for c in (56, 72, 80, 88, 96, 128)]
TOPOLOGIES = {
    "p8x2d8": ("2P1D", ["p0", "p1"], ["d0"], 16, 8),
    "p8x3d8": ("3P1D", ["p0", "p1", "p2"], ["d0"], 24, 8),
    "p8x2d8x2": ("2P2D", ["p0", "p1"], ["d0", "d1"], 16, 16),
}


def throughput(point: Path) -> dict | None:
    """Absolute and per-GPU throughput from the point's AgentX aggregate."""
    found = list(point.glob("bench/agentx_conc*.json")) or list(
        point.glob("agentx_conc*.json"))
    if not found:
        return None
    record = json.loads(found[0].read_text())
    rates = record.get("request_metrics", {}).get("throughput", {})
    # A point that died mid-run can still leave a partial aggregate. It has no
    # throughput to report and is not a row in the curve, but it must not take
    # the tool down with it -- the other points are why the tool was run.
    if "input" not in rates or "per_gpu" not in rates:
        return None
    return {
        "input": rates["input"]["tokens_per_second"],
        "output": rates["output"]["tokens_per_second"],
        "per_gpu": rates["per_gpu"]["total_tput_tps"],
    }


def samples(point: Path, prefill: list[str], decode: list[str]) -> list[dict]:
    """Prometheus samples taken while a decode was running.

    A truncated line is skipped rather than repaired: the sampler writes one
    line per scrape and a partial one means that scrape did not complete.
    """
    log = point / "metrics_sample.log"
    if not log.is_file():
        return []
    wanted = prefill + decode
    out = []
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("#") or "URLError" in line:
            continue
        record = {}
        for chunk in line.split("|"):
            fields = chunk.split()
            if not fields:
                continue
            name = fields[1] if fields[0][0].isdigit() else fields[0]
            values = {}
            for field in fields:
                key, sep, value = field.partition("=")
                if sep:
                    try:
                        values[key] = float(value)
                    except ValueError:
                        pass
            record[name] = values
        if not all(n in record and {"run", "kv"} <= record[n].keys()
                   for n in wanted):
            continue
        if sum(record[d]["run"] for d in decode) > 0:
            out.append(record)
    return out


def row(label: str, conc: int, point: Path, prefill: list[str],
        decode: list[str], p_gpu: int, d_gpu: int) -> str | None:
    rates = throughput(point)
    if rates is None:
        return None
    total = rates["input"] + rates["output"]
    rows = samples(point, prefill, decode)
    if rows:
        peak = lambda names, key: max(max(r[n][key] for n in names) for r in rows)
        total_median = lambda names, key: statistics.median(
            sum(r[n][key] for n in names) for r in rows)
        bottleneck = (f"{peak(prefill, 'q'):5.0f} | {peak(decode, 'kv'):.2f} | "
                      f"{total_median(decode, 'xfer'):5.0f} | "
                      f"{total_median(decode, 'run'):4.0f} | "
                      f"{statistics.median(max(r[p]['hit'] for p in prefill) for r in rows):.3f}")
    else:
        bottleneck = "    - |    - |     - |    - |     -"
    return (f"{label:6s} C{conc:<4d} {p_gpu + d_gpu:3d}  {total:10,.0f}  "
            f"{rates['per_gpu']:9,.1f}  {bottleneck}")


def main() -> None:
    print(f"{'点':6s} {'并发':5s} {'卡':>3s}  {'绝对 tok/s':>10s}  "
          f"{'tok/s/GPU':>9s}  {'P队列峰':>5s} | {'D_KV峰':>4s} | "
          f"{'xfer中位':>5s} | {'D_run':>4s} | {'命中':>5s}")
    print("-" * 96)
    for label, point, prefill, decode, p_gpu, d_gpu in BASE_POINTS:
        line = row(label, int(point.name[1:]), point, prefill, decode, p_gpu, d_gpu)
        if line:
            print(line)
    for topology, (label, prefill, decode, p_gpu, d_gpu) in TOPOLOGIES.items():
        for point in sorted((CAMPAIGN / topology).glob("c*"),
                            key=lambda p: int(p.name[1:].split(".")[0])):
            line = row(label, int(point.name[1:].split(".")[0]), point,
                       prefill, decode, p_gpu, d_gpu)
            if line:
                print(line)


if __name__ == "__main__":
    main()
