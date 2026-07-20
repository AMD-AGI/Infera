###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Render one or many hosts' findings to a self-contained HTML report.

Top of the report is a cluster matrix (rows = hosts, cols = sections), so a bad
node jumps out at a glance; below it, each host gets a detail block. One file,
inline CSS, no external assets.
"""

from __future__ import annotations

import html
import json
from datetime import datetime

from .finding import status_from

_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 2rem auto; max-width: 1100px; padding: 0 1rem; line-height: 1.5; }
h1 { margin-bottom: .3rem; }
.meta { color: #666; margin-bottom: 1.5rem; }
.meta code { background: rgba(127,127,127,.15); padding: .1rem .35rem; border-radius: 4px; }
h2 { margin-top: 2rem; border-bottom: 1px solid rgba(127,127,127,.3); padding-bottom: .3rem; }
h3 { margin-top: 1.4rem; margin-bottom: .2rem; }
h4 { margin: 1rem 0 .2rem; }
table { border-collapse: collapse; width: 100%; margin-top: .5rem; font-size: .92rem; }
th, td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid rgba(127,127,127,.2); vertical-align: top; }
th { background: rgba(127,127,127,.1); }
tr:hover td { background: rgba(127,127,127,.06); }
td.details { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .82rem;
             color: #555; word-break: break-word; }
table.matrix th, table.matrix td { text-align: center; }
table.matrix td:first-child, table.matrix th:first-child { text-align: left; }
.na { color: #999; }
.badge { display: inline-block; padding: .12rem .5rem; border-radius: 999px; font-size: .78rem;
         font-weight: 600; color: #fff; }
.ok   { background: #2e9e5b; }
.warn { background: #d99400; }
.fail { background: #d64545; }
.info { background: #8a8f98; }
a { color: #3b82f6; text-decoration: none; }
a:hover { text-decoration: underline; }
"""

# Friendly name + one-line description per section (shown in the report).
_SECTION_INFO = {
    "host": ("Host", "CPU, memory, free disk, NUMA, and the memlock limit RDMA needs."),
    "gpu": ("GPU", "GPU count, model, VRAM, temperature, driver version."),
    "network": (
        "Network",
        "Plain NICs, RDMA (ionic) devices and their link state, and each RDMA "
        "device's backing netdev and IP.",
    ),
    "firmware": (
        "Firmware",
        "Low-level firmware and GPU-direct config: GPU firmware version, whether "
        "GPU-direct (ais-check) is engaged, kernel P2PDMA setting.",
    ),
    "storage": (
        "NVMe / KV throughput",
        "Local NVMe inventory (count/size/model); NVMe<->DRAM throughput (O_DIRECT "
        "multi-threaded, the storage-substrate ceiling); NVMe<->HBM single-stream "
        "staged throughput (CPU-bounce via host, needs a GPU, runs in-container). "
        "Tests the largest local NVMe mount by default, overridable with "
        "--storage-path; FAILs when KV storage isn't on local NVMe (no disk or "
        "fell back to overlay).",
    ),
    "compute": ("GPU compute", "Per-GPU bf16 GEMM throughput (TFLOPs)."),
    "hbm": ("HBM bandwidth", "Per-GPU ~1GiB random-buffer copy read+write bandwidth (GB/s)."),
    "p2p": ("Intra-node interconnect", "Per-GPU-pair ~512MiB copy bandwidth (GB/s), as a matrix."),
    "topology": ("GPU topology", "Inter-GPU link type (xGMI/PCIe) and GPU<->NUMA mapping."),
    "fabric": (
        "RDMA fabric",
        "Per RDMA NIC: RoCE v2 GID, subnet, MTU (informational). Whether nodes "
        "can actually reach each other is decided by the inter-node interconnect test.",
    ),
    "netperf": (
        "Inter-node interconnect",
        "Measured cross-node RoCE bandwidth via ib_write_bw (GB/s): each node pair "
        "tests an NxN NIC matrix, rows = server NIC, cols = client NIC; blank cells "
        "are unreachable.",
    ),
    "mooncake": (
        "Mooncake KV transfer",
        "Measured cross-node KV-move bandwidth via Mooncake TransferEngine (GB/s): "
        "rdma (MC_GID_INDEX set, correct deployment), rdma-default (unset, simulates "
        "an operator who forgot the knob -> cross-node unreachable), tcp "
        "(MC_FORCE_TCP). Each row states the env it needed.",
    ),
    "mori": (
        "Mori KV transfer",
        "Measured cross-node KV-move bandwidth via Mori IOEngine (GB/s, RDMA only; "
        "Mori has no TCP). Mori selects a routable GID itself; only MORI_RDMA_DEVICES "
        "is needed.",
    ),
}

# Top-level categories → the sections they contain, in order.
_CATEGORIES: list[tuple[str, list[str]]] = [
    ("Base info", ["host", "gpu", "network", "firmware"]),
    ("Single-node perf & topology", ["compute", "hbm", "p2p", "topology"]),
    ("Multi-node / PD", ["fabric", "netperf", "mooncake", "mori"]),
    ("Storage", ["storage"]),
]


def _friendly(name: str) -> str:
    return _SECTION_INFO.get(name, (name, ""))[0]


def _desc(name: str) -> str:
    return _SECTION_INFO.get(name, (name, ""))[1]


def _badge(status: str) -> str:
    return f'<span class="badge {status.lower()}">{html.escape(status)}</span>'


def _matrix_html(caption: str, details: dict) -> str:
    """Render a bandwidth matrix. Rows/cols may differ; row label and corner
    header are configurable (``row_label`` default "gpu", ``corner`` default
    "src\\dst") so both the GPU P2P grid and the cross-node NIC grid reuse it."""
    m = details["matrix"]
    nrows = len(m)
    ncols = len(m[0]) if m else 0
    unit = details.get("unit", "")
    row_label = details.get("row_label", "gpu")
    corner = details.get("corner", "src\\dst")
    head = "".join(f"<th>{j}</th>" for j in range(ncols))
    body = []
    for i in range(nrows):
        cells = "".join(
            '<td class="na">—</td>' if m[i][j] is None else f"<td>{m[i][j]}</td>"
            for j in range(ncols)
        )
        body.append(f"<tr><td>{html.escape(row_label)}{i}</td>{cells}</tr>")
    cap = html.escape(f"{caption} ({unit})")
    return (
        f'<p class="meta">{cap}</p>'
        f'<table class="matrix"><thead><tr><th>{html.escape(corner)}</th>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _fmt_details(details: dict) -> str:
    if not details:
        return ""
    try:
        text = json.dumps(details, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        text = str(details)
    return html.escape(text)


def _global_groups(present: set[str]) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """Number the sections present in any host as category/section (1, 1.1, ...).

    Returns ``[(cat_num, cat_name, [(sec_num, sec_key), ...]), ...]``. Sections
    not listed in ``_CATEGORIES`` fall into a trailing "Other" category.
    """
    out: list[tuple[str, str, list[tuple[str, str]]]] = []
    used: set[str] = set()
    ci = 0
    for cat_name, keys in _CATEGORIES:
        ks = [k for k in keys if k in present]
        if not ks:
            continue
        ci += 1
        used.update(ks)
        out.append((str(ci), cat_name, [(f"{ci}.{i + 1}", k) for i, k in enumerate(ks)]))
    leftover = [k for k in sorted(present) if k not in used]
    if leftover:
        ci += 1
        out.append((str(ci), "Other", [(f"{ci}.{i + 1}", k) for i, k in enumerate(leftover)]))
    return out


def render_html(results: list[dict]) -> str:
    """``results``: list of ``{"host", "generated", "sections": {key: [Finding]}}``."""
    results = sorted(results, key=lambda r: r["host"])
    present: set[str] = set()
    for r in results:
        present |= set(r["sections"].keys())
    groups = _global_groups(present)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    p: list[str] = []
    p.append("<!doctype html>")
    p.append('<html lang="en"><head><meta charset="utf-8">')
    p.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    p.append("<title>Infera Preflight Report</title>")
    p.append(f"<style>{_CSS}</style></head><body>")

    p.append("<h1>Infera Preflight Report</h1>")
    p.append('<div class="meta">')
    p.append(f"Nodes <code>{len(results)}</code> &nbsp;·&nbsp; Generated <code>{now}</code>")
    p.append("</div>")

    _node_list(p, results)
    _overview(p, results, groups)
    _details(p, results, groups)

    p.append("</body></html>")
    return "\n".join(p) + "\n"


def _node_list(p: list[str], results: list[dict]) -> None:
    """One row per host (name / image / time) so the report records exactly what
    was tested and in which engine image."""
    p.append("<h2>Nodes</h2>")
    p.append("<table><thead><tr><th>Node</th><th>Image</th><th>Collected</th></tr></thead><tbody>")
    for r in results:
        p.append(
            f"<tr><td>{html.escape(r['host'])}</td>"
            f'<td class="details">{html.escape(str(r.get("image", "unknown")))}</td>'
            f"<td>{html.escape(str(r.get('generated', '')))}</td></tr>"
        )
    p.append("</tbody></table>")


def _mooncake_group(message: str) -> str | None:
    """Map a Mooncake finding to its overview column (rdma / rdma-default / tcp).

    The per-GPU VRAM variants (``rdma-gpu{g}``) are the ionic bare-VRAM RDMA
    registration path this whole check exists to catch, so they roll up UNDER the
    RDMA badge. Without this they matched no column and were dropped from the
    at-a-glance overview -- a failing GPU KV transfer stayed green there and only
    showed in the per-node detail table."""
    label = message.rsplit(" ", 1)[-1].lower()
    if label == "rdma" or label.startswith("rdma-gpu"):
        return "rdma"
    if label in ("rdma-default", "tcp"):
        return label
    return None


def _overview_cell(key: str, fs: list, anchor: str) -> str:
    """One overview cell. Fabric is informational only (reachability is decided by
    the netperf section), so it shows a neutral INFO badge rather than
    OK/WARN/FAIL. Mooncake splits its status per variant (RDMA / RDMA(default) /
    TCP), since they can differ; every other section rolls up to a single badge."""
    if key == "fabric":
        return f'<a href="#{anchor}">{_badge("INFO")}</a>'
    if key == "mooncake":
        labels = {"rdma": "RDMA", "rdma-default": "RDMA(default)", "tcp": "TCP"}
        by_proto: dict[str, list] = {}
        for f in fs:
            proto = _mooncake_group(f.message)
            if proto is not None:
                by_proto.setdefault(proto, []).append(f)
        if by_proto:
            parts = [
                f"{labels[p]}: {_badge(status_from(by_proto[p]))}"
                for p in ("rdma", "rdma-default", "tcp")
                if p in by_proto
            ]
            return f'<a href="#{anchor}">' + " &nbsp; ".join(parts) + "</a>"
    return f'<a href="#{anchor}">{_badge(status_from(fs))}</a>'


def _overview(p: list[str], results: list[dict], groups: list) -> None:
    """Cluster overview: one summary table per category (rows = hosts, cols = sections)."""
    p.append("<h2>0. Cluster overview</h2>")
    for i, (_, cat_name, secs) in enumerate(groups, start=1):
        p.append(f"<h3>0.{i} {html.escape(cat_name)}</h3>")
        head = "".join(
            f'<th title="{html.escape(_desc(key))}">{num} {html.escape(_friendly(key))}</th>'
            for num, key in secs
        )
        p.append(f'<table class="matrix"><thead><tr><th>Node</th>{head}</tr></thead><tbody>')
        for r in results:
            cells = [f"<td>{html.escape(r['host'])}</td>"]
            for _, key in secs:
                fs = r["sections"].get(key)
                if fs is None:
                    cells.append('<td class="na">—</td>')
                else:
                    anchor = f"{html.escape(key)}--{html.escape(r['host'])}"
                    cells.append(f"<td>{_overview_cell(key, fs, anchor)}</td>")
            p.append("<tr>" + "".join(cells) + "</tr>")
        p.append("</tbody></table>")


def _details(p: list[str], results: list[dict], groups: list) -> None:
    """Detail layout: category → section → host, so one section is easy to
    compare across all hosts."""
    for cat_num, cat_name, secs in groups:
        p.append(f"<h2>{cat_num}. {html.escape(cat_name)}</h2>")
        for num, key in secs:
            p.append(f"<h3>{num} {html.escape(_friendly(key))}</h3>")
            p.append(f'<p class="meta">{html.escape(_desc(key))}</p>')
            for r in results:
                fs = r["sections"].get(key)
                if fs is None:
                    continue
                host = html.escape(r["host"])
                # Fabric is informational only (reachability is decided by the
                # netperf section), so it shows a neutral INFO badge, not OK/WARN/FAIL.
                status = "INFO" if key == "fabric" else status_from(fs)
                p.append(f'<h4 id="{html.escape(key)}--{host}">{host} {_badge(status)}</h4>')
                # Findings carrying a "matrix" (e.g. P2P) render as their own grid;
                # the rest go in the normal level/check/details table.
                rows = [f for f in fs if "matrix" not in f.details]
                if rows:
                    p.append(
                        "<table><thead><tr><th>level</th><th>check</th><th>details</th>"
                        "</tr></thead><tbody>"
                    )
                    for f in rows:
                        p.append(
                            f"<tr><td>{_badge(f.level)}</td><td>{html.escape(f.message)}</td>"
                            f'<td class="details">{_fmt_details(f.details)}</td></tr>'
                        )
                    p.append("</tbody></table>")
                for f in fs:
                    if "matrix" in f.details:
                        p.append(_matrix_html(f.message, f.details))
