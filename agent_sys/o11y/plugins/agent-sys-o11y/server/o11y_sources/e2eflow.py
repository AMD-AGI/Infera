"""e2e-flow standard files (schemas under agent_sys/examples/llm_e2e_performance_optimization/e2e-flow/
assets/schemas/): kernel_table.json (hot ops, descending self time) and integration_report.json
(per-operator e2e verdict). Only the fields o11y reads are required (data2.md §B.4)."""

from __future__ import annotations

import glob
import os

from o11y_protocol.owned import owned_json

from . import Problems

KT_REQUIRED = ["schema_version", "totals.rows", "totals.pct_total_sum", "kernels"]
KT_KERNEL_REQUIRED = ["name", "pct_total", "self_us", "calls"]
IR_REQUIRED = [
    "patch.operator_id",
    "verdict.accepted",
    "verdict.reasons",
    "kernel_reconciliation.kernel_speedup",
    "kernel_reconciliation.observed_e2e_speedup",
    "kernel_reconciliation.agrees",
    "generated_at",
]


def _files(globs: list[str], name: str) -> list[str]:
    out = []
    for g in globs:
        out += [p for p in glob.glob(g, recursive=True) if os.path.basename(p) == name]
    return sorted(set(out), key=os.path.getmtime)


def hot_ops(globs: list[str], problems: Problems, top: int = 15) -> dict | None:
    files = _files(globs, "kernel_table.json")
    if not files:
        return None
    path = files[-1]  # newest profile wins
    kt, why = owned_json(path, external=True)
    if why:
        problems.add("e2e_flow", path, "(file)", why)
        return None
    if not problems.require("e2e_flow", path, kt, KT_REQUIRED):
        return None
    ops = []
    for k in kt["kernels"][:top]:
        if isinstance(k, dict) and problems.require("e2e_flow", path, k, KT_KERNEL_REQUIRED):
            ops.append(
                {
                    "name": k["name"],
                    "share_pct": k["pct_total"],
                    "self_us": k["self_us"],
                    "calls": k["calls"],
                }
            )
    return {"source": path, "ops": ops, "rows": kt["totals"]["rows"]}


def integration_reports(globs: list[str], problems: Problems) -> dict[str, dict]:
    """operator_id -> latest verdict."""
    out: dict[str, dict] = {}
    for path in _files(globs, "integration_report.json"):
        r, why = owned_json(path, external=True)
        if why:
            problems.add("e2e_flow", path, "(file)", why)
            continue
        if not problems.require("e2e_flow", path, r, IR_REQUIRED):
            continue
        kr = r["kernel_reconciliation"]
        out[r["patch"]["operator_id"]] = {
            "accepted": r["verdict"]["accepted"],
            "reasons": r["verdict"]["reasons"],
            "kernel_speedup": kr["kernel_speedup"],
            "observed_e2e_speedup": kr["observed_e2e_speedup"],
            "agrees": kr["agrees"],
            "generated_at": r["generated_at"],
            "source": path,
        }
    return out
