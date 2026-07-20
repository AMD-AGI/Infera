###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Preflight CLI: per-node probes → per-host JSON → combined HTML.

Installed as the ``infera-preflight`` console command; ``python -m
infera.tools.preflight`` is equivalent (the examples below use the module form).

    infera-preflight                                       # this host: collect + render
    python -m infera.tools.preflight                       # this host: collect + render
    python -m infera.tools.preflight --gpu --network        # only selected probes
    python -m infera.tools.preflight --collect-only         # write <host>.json, no HTML
    python -m infera.tools.preflight --render-only          # combine existing *.json → HTML

Multi-node (SLURM, one command): srun runs one task per node; each collects and
rank 0 renders the combined report into the SHARED --dump-path.

    srun --nodelist=nodeA,nodeB -N2 --ntasks-per-node=1 \
        python -m infera.tools.preflight --dump-path <shared dir>

Without SLURM it runs on the single local node. --collect-only / --render-only
remain for manual multi-node (collect on each node, then render once).

Run it INSIDE the engine container so container-only checks (ais-check, etc.)
are real; forward the identifying env vars into the container:
    PREFLIGHT_HOST   node name (else SLURMD_NODENAME/HOSTNAME; a container's own
                     hostname is just its container id)
    PREFLIGHT_IMAGE  the engine image tag, shown in the report's node list
Exit code: 0 if no failures, 2 otherwise.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime

from . import firmware, gpu, host, network, storage
from .finding import Finding, status_from
from .gpu import perf as gpu_perf
from .gpu import topology as gpu_topo
from .network import fabric, mooncakeperf, moriperf, netperf
from .report import render_html

# section name -> collector. Order defines report order.
PROBES: dict[str, Callable[[], list[Finding]]] = {
    "host": host.collect,
    "gpu": gpu.collect,
    "network": network.collect,
    "firmware": firmware.collect,
    "storage": storage.collect,
    "compute": gpu_perf.collect_gemm,
    "hbm": gpu_perf.collect_hbm,
    "p2p": gpu_perf.collect_p2p,
    "topology": gpu_topo.collect,
    "fabric": fabric.collect,
}

# Cross-node consistency: (section, detail key, label). A node whose value
# differs from the fleet majority is flagged. Values come from probe findings.
_CONSISTENCY = [
    ("firmware", "mec", "MEC firmware"),
    ("gpu", "driver", "GPU driver"),
    ("gpu", "gfx", "GPU gfx"),
    ("gpu", "gpu_count", "GPU count"),
]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="infera.tools.preflight", description=__doc__)
    for name in PROBES:
        p.add_argument(f"--{name}", action="store_true", help=f"run the {name} probe")
    p.add_argument(
        "--dump-path", default="output/preflight", help="output dir (share it across nodes)"
    )
    p.add_argument(
        "--report-file-name", default="infera_preflight_report", help="HTML report base name"
    )
    p.add_argument(
        "--storage-path",
        default="",
        help="dir for the storage throughput test (default: auto-pick largest local NVMe mount)",
    )
    p.add_argument(
        "--netperf",
        action="store_true",
        help="run the cross-node RoCE bandwidth matrix (coordinated; multi-node)",
    )
    p.add_argument(
        "--mooncake",
        action="store_true",
        help="run the cross-node Mooncake KV-transfer check, rdma + tcp (coordinated)",
    )
    p.add_argument(
        "--mori",
        action="store_true",
        help="run the cross-node Mori KV-transfer check, rdma only (coordinated)",
    )
    p.add_argument("--collect-only", action="store_true", help="write this host's JSON, skip HTML")
    p.add_argument(
        "--render-only", action="store_true", help="combine existing *.json, skip probing"
    )
    return p.parse_args(argv)


def _host_name() -> str:
    # Inside a container gethostname() is the container id, so prefer an explicit
    # node name passed by the launcher (SLURM sets SLURMD_NODENAME).
    return (
        os.environ.get("PREFLIGHT_HOST")
        or os.environ.get("SLURMD_NODENAME")
        or os.environ.get("HOSTNAME")
        or socket.gethostname()
    )


def _collect(selected: list[str], host: str) -> dict[str, list[Finding]]:
    sections: dict[str, list[Finding]] = {}
    for name in selected:
        print(f"[preflight] [{host}] running {name}...", file=sys.stderr)
        try:
            sections[name] = PROBES[name]()
        except Exception as e:  # noqa: BLE001 - a broken probe must not kill the run
            sections[name] = [Finding("fail", f"{name} probe crashed", {"error": str(e)})]
        print(f"[preflight] [{host}]   {name}: {status_from(sections[name])}", file=sys.stderr)
    return sections


def _write_json(dump_path: str, host_name: str, sections: dict[str, list[Finding]]) -> str:
    os.makedirs(dump_path, exist_ok=True)
    record = {
        "host": host_name,
        "image": os.environ.get("PREFLIGHT_IMAGE") or "unknown",
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sections": {k: [asdict(f) for f in fs] for k, fs in sections.items()},
    }
    path = os.path.join(dump_path, f"{host_name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=1)
    return path


def _load_results(dump_path: str) -> list[dict]:
    results = []
    for path in sorted(glob.glob(os.path.join(dump_path, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        rec["sections"] = {
            k: [Finding(**d) for d in fs] for k, fs in rec.get("sections", {}).items()
        }
        results.append(rec)
    return results


def _flag_version_mismatch(results: list[dict], section: str, key: str, label: str) -> None:
    """Cross-node consistency: warn on hosts whose ``details[key]`` (e.g. MEC
    firmware) differs from the fleet majority. A firmware version has no reliable
    absolute threshold, but a node out of step with its peers is a real signal."""
    vals = {
        r["host"]: f.details[key]
        for r in results
        for f in r["sections"].get(section, [])
        if key in f.details
    }
    if len(set(vals.values())) <= 1:
        return
    counts = Counter(vals.values())
    top = counts.most_common(1)[0][1]
    all_tied = len(set(counts.values())) == 1  # no clear majority (e.g. 2 nodes differ)
    dist = {str(k): v for k, v in sorted(counts.items())}
    for r in results:
        v = vals.get(r["host"])
        if v is not None and (all_tied or counts[v] < top):
            r["sections"][section].append(
                Finding("warn", f"{label} {v} inconsistent across fleet", {"fleet": dist})
            )


def _build_netperf_matrices(results: list[dict]) -> None:
    """Turn each client node's flat netperf results into per-(server,client)
    NIC matrices for the report (rows = server NIC, cols = client NIC)."""
    for r in results:
        fs = r["sections"].get("netperf")
        if not fs:
            continue
        tests = [t for f in fs for t in f.details.get("tests", [])]
        by_server: dict[str, list[dict]] = {}
        for t in tests:
            by_server.setdefault(t["server"], []).append(t)
        for server, ts in sorted(by_server.items()):
            nrows = max(t["s_nic"] for t in ts) + 1
            ncols = max(t["c_nic"] for t in ts) + 1
            matrix = [[None] * ncols for _ in range(nrows)]
            for t in ts:
                v = t.get("gb_s")
                matrix[t["s_nic"]][t["c_nic"]] = None if v is None else round(v, 1)
            r["sections"]["netperf"].append(
                Finding(
                    "info",
                    f"{server} (server) -> {ts[0]['client']} (client)",
                    {"matrix": matrix, "unit": "GB/s", "row_label": "srv", "corner": "srv\\cli"},
                )
            )


def _rank_world() -> tuple[int, int]:
    """This task's rank and the node count. Under SLURM (srun) these come from
    SLURM; otherwise it's a single node (rank 0 of 1)."""
    rank = int(os.environ.get("SLURM_PROCID") or 0)
    world = int(os.environ.get("SLURM_NNODES") or os.environ.get("SLURM_NTASKS") or 1)
    return rank, world


def _wait_for_json(dump_path: str, count: int, timeout: float = 300.0) -> int:
    """Block until `count` per-host JSON files exist in dump_path (or timeout)."""
    deadline = time.monotonic() + timeout
    while True:
        have = len(glob.glob(os.path.join(dump_path, "*.json")))
        if have >= count or time.monotonic() >= deadline:
            return have
        time.sleep(2.0)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    rank, world = _rank_world()
    node_fail = 0

    if not args.render_only:
        selected = [n for n in PROBES if getattr(args, n)]
        want_netperf = args.netperf
        want_mooncake = args.mooncake
        want_mori = args.mori
        if not selected and not (want_netperf or want_mooncake or want_mori):  # nothing -> all
            selected, want_netperf, want_mooncake, want_mori = list(PROBES), True, True, True
        if args.storage_path:  # let the zero-arg storage probe read it via env
            os.environ["INFERA_PREFLIGHT_STORAGE_PATH"] = args.storage_path
        host_name = _host_name()
        sections = _collect(selected, host_name)
        if want_netperf:
            print(f"[preflight] [{host_name}] running netperf...", file=sys.stderr)
            sections["netperf"] = netperf.run_matrix(args.dump_path, rank, world, host_name)
        if want_mooncake:
            print(f"[preflight] [{host_name}] running mooncake...", file=sys.stderr)
            sections["mooncake"] = mooncakeperf.run(args.dump_path, rank, world, host_name)
        if want_mori:
            print(f"[preflight] [{host_name}] running mori...", file=sys.stderr)
            sections["mori"] = moriperf.run(args.dump_path, rank, world, host_name)
        overall = status_from([f for fs in sections.values() for f in fs])
        print(
            f"[preflight] host={host_name} rank={rank}/{world} "
            f"sections={','.join(selected)} status={overall}",
            file=sys.stderr,
        )
        for name, findings in sections.items():
            for f in findings:
                if f.level in ("warn", "fail"):
                    print(f"[preflight] {f.level.upper()}: [{name}] {f.message}", file=sys.stderr)
        path = _write_json(args.dump_path, host_name, sections)
        print(f"[preflight] wrote {path}", file=sys.stderr)
        node_fail = sum(f.level == "fail" for fs in sections.values() for f in fs)

    # Only rank 0 renders the combined report; other SLURM ranks stop here.
    if args.collect_only or rank != 0:
        return 2 if node_fail else 0

    if world > 1 and not args.render_only:
        got = _wait_for_json(args.dump_path, world)
        if got < world:
            print(
                f"[preflight] WARN: only {got}/{world} node reports before timeout", file=sys.stderr
            )

    results = _load_results(args.dump_path)
    if not results:
        print(f"[preflight] no *.json under {args.dump_path}", file=sys.stderr)
        return 2
    for section, key, label in _CONSISTENCY:
        _flag_version_mismatch(results, section, key, label)
    _build_netperf_matrices(results)
    html_path = os.path.join(args.dump_path, f"{args.report_file_name}.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(render_html(results))
    print(f"[preflight] report ({len(results)} host(s)): {html_path}", file=sys.stderr)
    fleet_fail = sum(
        f.level == "fail" for r in results for fs in r["sections"].values() for f in fs
    )
    return 2 if fleet_fail else 0
