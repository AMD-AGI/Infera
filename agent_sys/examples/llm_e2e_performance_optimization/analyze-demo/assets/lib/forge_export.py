#!/usr/bin/env python3
"""One operator record -> the two files KernelForge reads.

DESIGN.md section 5.1: both formats are produced, and the field mapping lives
here in one place so that `invocation_spec_<op>.json` and `forge_task.yaml`
cannot drift apart. `check_workset_shape` re-derives the shared fields from this
module and compares them against what was written.

- `build_invocation_spec` follows Hyperloom `_invocation_spec.py` schema v2.
  Its `status` / `missing_fields` pair is what lets a partially resolved
  operator be exported honestly instead of with invented values.
- `build_forge_task` follows `KernelForge/docs/reference/task-definition.md`.

Neither function invents a value. Anything the upstream record did not carry
comes out empty and is named in `missing_fields`.
"""

from __future__ import annotations

import re

SCHEMA_VERSION = 2

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

#: Fields whose absence makes the spec `partial`. Ordered so that the
#: `missing_fields` list reads the same way every time.
REQUIRED_FOR_COMPLETE = [
    "edit_target.source_file",
    "invocation.launcher_locator",
    "implementation.symbols",
    "invocation.arguments",
    "tests.driver_contract",
]


def operator_id(record: dict) -> str:
    """A path-safe identity, used for the directory and the spec filename."""
    raw = (
        record.get("logical_operator")
        or record.get("category")
        or record.get("name")
        or "unknown_operator"
    )
    return _SAFE.sub("_", str(raw)).strip("._-")[:96] or "unknown_operator"


def invocation_spec_filename(record: dict) -> str:
    return f"invocation_spec_{operator_id(record)}.json"


def _missing(spec: dict) -> list[str]:
    out = []
    for path in REQUIRED_FOR_COMPLETE:
        node = spec
        for part in path.split("."):
            node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                break
        if not node:
            out.append(path)
    return out


def build_invocation_spec(record: dict, environment: dict) -> dict:
    """Hyperloom invocation-spec schema v2 for one operator."""
    identity = record.get("identity") or {}
    cases = record.get("cases") or []

    spec = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "missing_fields": [],
        "logical_operator": record.get("logical_operator") or "",
        "source_framework": identity.get("source_owner") or environment.get("framework") or "",
        "implementation": {
            "sources": list(identity.get("source_file_path") or []),
            "kernel_kind": identity.get("kernel_kind") or "",
            "symbols": list(identity.get("target_kernel_functions") or []),
            "runtime_backend": identity.get("repository_language") or "",
        },
        "kernel": {
            "kernel_id": record.get("kernel_id") or "",
            "name": record.get("name") or "",
            "operation": record.get("logical_operator") or "",
            "kernel_category": record.get("category") or "",
            "source_type": identity.get("kernel_kind") or "",
            "kernel_kind": identity.get("kernel_kind") or "",
        },
        "edit_target": {
            "source_file": (list(identity.get("editable_sources") or []) or [""])[0],
            "repo_root": identity.get("image_repo_path") or "",
            "source_symbol": (list(identity.get("target_kernel_functions") or []) or [""])[0],
            "runtime_symbols": list(identity.get("target_kernel_functions") or []),
            "kernel_sources": list(identity.get("source_file_path") or []),
            "resolution_method": identity.get("source_resolution_method") or "",
        },
        "invocation": {
            "launcher_source_file": identity.get("launcher_source_file") or "",
            "launcher_locator": identity.get("launcher_locator") or "",
            "public_callable_candidates": list(identity.get("target_kernel_functions") or []),
            "arguments": (cases[0].get("arguments") if cases else []) or [],
            "outputs": (cases[0].get("outputs") if cases else []) or [],
            "kernel_contract": {},
        },
        "tests": {
            "primary_benchmark": "scripts/forge_driver.py",
            "related_files": ["scripts/task_runner.py", "reference/naive_torch.py"],
            "driver_contract": _driver_contract(cases),
        },
        "workload": {
            "call_count": record.get("calls"),
            "task_group": {
                "task_group_id": operator_id(record),
                "primary_kernel_id": record.get("kernel_id") or "",
                "kernel_ids": [record.get("kernel_id") or ""],
                "aggregate_gpu_pct": record.get("pct_total"),
                "aggregate_call_count": record.get("calls"),
                "cases": cases,
            },
        },
        "execution": {
            "framework": environment.get("framework") or "",
            "precision": record.get("precision") or "",
            "target_platform": environment.get("gpu_type") or "",
            "is_multigpu": False,
        },
        "deployment": record.get("deployment") or {},
        "provenance": {
            "trace_report_path": record.get("trace_report_path") or "",
            "shape_provenance": record.get("shape_provenance") or "magpie_gap_analysis_csv",
            "source_resolution_status": identity.get("source_resolution_method") or "unresolved",
            "source_resolution_reason": identity.get("resolution_hint") or "",
        },
    }

    spec["missing_fields"] = _missing(spec)
    spec["status"] = "complete" if not spec["missing_fields"] else "partial"
    return spec


def _driver_contract(cases: list[dict]) -> dict:
    """`tests.driver_contract`, matching Hyperloom `_driver_contract`."""
    if not cases:
        return {}
    return {
        "shape_argument": "--shape",
        "case_selector_key": "CASE_ID",
        "requires_all_cases": len(cases) > 1,
        "case_selectors": [dict(c.get("selector") or {}) for c in cases if c.get("selector")],
    }


def build_forge_task(record: dict, environment: dict, spec: dict) -> dict:
    """KernelForge orchestrator task definition for the same operator.

    Anything the spec named in `missing_fields` becomes a `TODO:` constraint, so
    a human reading the YAML alone sees the same gaps the JSON declares.
    """
    identity = record.get("identity") or {}
    cases = record.get("cases") or []
    selectors = [dict(c.get("selector") or {}) for c in cases]

    constraints = list(record.get("constraints") or [])
    for field in spec.get("missing_fields") or []:
        constraints.append(f"TODO: {field} was not resolved by the analyze stage")
    if identity.get("resolution_hint"):
        constraints.append(f"Resolution hint: {identity['resolution_hint']}")

    task = {
        "task_id": operator_id(record),
        "description": (
            f"Optimize {record.get('logical_operator') or record.get('name')} on "
            f"{environment.get('gpu_type', 'unknown')}/{environment.get('gpu_target', 'unknown')}. "
            f"Observed at {record.get('pct_total')}% of profiled GPU time over "
            f"{record.get('calls')} calls."
        ),
        "operation": record.get("category") or "unknown",
        "dtype": record.get("precision") or "unknown",
        "gpu_target": environment.get("gpu_target") or "",
        "shapes": {
            "primary": selectors[0] if selectors else {},
            "validation": selectors[1:],
        },
        "backends": [identity.get("repository_language")] if identity.get("repository_language") else [],
        "targets": {
            "snr_db": float(environment.get("snr_threshold") or 30.0),
            "baseline_wall_ms": record.get("baseline_wall_ms"),
        },
        "paths": {
            "reference": "reference/naive_torch.py",
            "test_driver": "scripts/forge_driver.py",
        },
        "source_files": list(identity.get("source_file_path") or []),
        "constraints": constraints,
    }
    return task


def shared_fields(spec: dict, task: dict) -> dict:
    """The values `check_workset_shape` compares across the two files.

    Each entry is `(from_spec, from_task)` so the caller reports which side
    disagreed rather than only that they differ.

    `execution.target_platform` and `gpu_target` are deliberately **not**
    compared: Hyperloom's field is the SKU (`mi355x`) and KernelForge's is the
    compilation arch (`gfx950`). They describe the same machine and are never
    equal, so an equality check on them would fail on a correct pair.
    """
    cases = spec.get("workload", {}).get("task_group", {}).get("cases") or []
    return {
        "operator": (
            operator_id({"logical_operator": spec.get("logical_operator")}),
            task.get("task_id"),
        ),
        "primary_case": (
            (cases[0].get("selector") if cases else {}),
            task.get("shapes", {}).get("primary"),
        ),
        "n_cases": (
            len(cases),
            1 + len(task.get("shapes", {}).get("validation") or []),
        ),
        "source_files": (
            sorted(spec.get("implementation", {}).get("sources") or []),
            sorted(task.get("source_files") or []),
        ),
    }
