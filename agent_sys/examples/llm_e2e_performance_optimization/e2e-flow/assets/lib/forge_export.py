#!/usr/bin/env python3
"""The KernelForge add-on, generated from the base format and never beside it.

M3.7.6: *"kernel forge well format one line script 作为额外项目, 在基础格式上封装"* —
**layered over** the flashinfer-bench base, not an alternative to it. A consumer
that does not use KernelForge ignores `operators[].forge` entirely and loses
nothing, because the base carries the whole contract.

Everything here is derived from `definitions/<...>.json` plus
`workloads/<...>.jsonl` plus the operator's row in `workset.yaml`. Nothing is
authored. That is the property `check_workset_shape` checks, by comparing the
exported case list against the workload's — and the reason to check it is the
one the analysis stage gave for grading `invocation_spec.json` against
`forge_task.yaml`: two files produced from one record, so a disagreement can
only mean one of them was edited on its own.

Run it as a program, which is how the task readme's STEP 6 invokes it:

    python3 "$PKG/assets/lib/forge_export.py" --workset "$OUT/items/codes"
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import patchkit  # noqa: E402  -- sibling in assets/lib, path set above

__all__ = ["export_operator", "export_all", "main"]

#: forge-loop parses these off the driver's stdout and treats the driver as a
#: **protected file** its optimising agent may not modify. It therefore takes
#: what the driver prints entirely on trust, which is why `check_workset_runs`
#: exists one level up: something has to have run it.
_DRIVER = pathlib.Path(__file__).with_name("workset_driver.py.in").read_text(encoding="utf-8")

# The package alias materializes the skill-owned template beside this exporter.
_PROGRAM = pathlib.Path(__file__).with_name("workset_program.md").read_text(encoding="utf-8")

_ONE_LINE = pathlib.Path(__file__).with_name("workset_launcher.sh.in").read_text(encoding="utf-8")


def _driver_source(operator_id: str, case_ids: list[str], kernel_rel: str) -> str:
    """Render valid Python literals in the same style as the frozen driver."""

    def literal(value: str) -> str:
        # JSON string escapes are valid Python; retain Unicode code points and
        # escape unpaired surrogates so the generated file remains writable.
        return json.dumps(value, ensure_ascii=False).encode("utf-8", "backslashreplace").decode()

    cases = [literal(case_id) for case_id in case_ids]
    case_literals = "[" + ", ".join(cases) + "]"
    if len("CASE_IDS = " + case_literals) > 100:
        case_literals = "[\n" + "".join(f"    {case},\n" for case in cases) + "]"
    # A title is inside a docstring rather than a Python expression. Escape its
    # control characters, backslashes and quotes independently of the literals.
    title = operator_id.encode("unicode_escape").decode("ascii").replace('"', '\\"')
    return _DRIVER.format(
        operator_id=title,
        operator_literal=literal(operator_id),
        case_literals=case_literals,
        kernel_literal=literal(kernel_rel),
    )


def export_operator(root: pathlib.Path, operator: dict) -> dict:
    """Write one operator's add-on and return the `forge` block for `workset.yaml`."""
    operator_id = operator["operator_id"]
    directory = root / "operators" / operator_id
    (directory / "scripts").mkdir(parents=True, exist_ok=True)
    (directory / "tests").mkdir(parents=True, exist_ok=True)

    definition = json.loads((root / operator["definition"]).read_text(encoding="utf-8"))
    lines = [
        json.loads(line)
        for line in (root / operator["workload"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    case_ids = [s["case_id"] for s in operator["shapes"]]
    backend = _kernel_backend(operator, definition)

    # `tests/cases.json` — the CASE_ID-keyed view forge-loop's `--shape
    # CASE_ID=<id>` selector needs. Order follows the workload, because
    # `check_workset_shape` compares the two as sequences: a set comparison would
    # accept a re-sorted export, and the selector would then pick a different
    # case than a reader of `workset.yaml` expects.
    (directory / "tests" / "cases.json").write_text(
        json.dumps(
            {
                "operator_id": operator_id,
                "kernel_id": operator.get("kernel_id"),
                "device_symbol": operator.get("device_symbol"),
                "snr_threshold_db": operator["gates"]["snr_db"],
                "generated_from": ["definition", "workload"],
                "cases": [
                    {
                        "case_id": shape["case_id"],
                        "selector": {
                            "CASE_ID": shape["case_id"],
                            **{k.upper(): v for k, v in shape["axes"].items()},
                        },
                        "uuid": shape["uuid"],
                        "axes": {
                            **{
                                n: s["value"]
                                for n, s in definition["axes"].items()
                                if s["type"] == "const"
                            },
                            **shape["axes"],
                        },
                        "is_primary": bool(shape.get("is_primary")),
                        "synthetic": not shape.get("observed", True),
                        "role": shape["role"],
                    }
                    for shape, _ in zip(operator["shapes"], lines)
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (directory / "invocation_spec.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": operator["status"],
                "missing_fields": operator["missing_fields"],
                "logical_operator": operator_id,
                "kernel_id": operator.get("kernel_id"),
                "device_symbol": operator.get("device_symbol"),
                "description": definition.get("description", ""),
                "environment": {"gpu_target": operator.get("_gpu_arch", ""), "framework": "sglang"},
                "implementation": {
                    "source_owner": operator["edit_target"].get("source_owner", ""),
                    "image_repo_path": operator["edit_target"].get("repo_root_var", ""),
                    "sources": operator["edit_target"].get("editable_sources") or [],
                    "editable_sources": operator["edit_target"].get("editable_sources") or [],
                    "target_kernel_functions": [
                        f for f in [operator["edit_target"].get("entry_function")] if f
                    ],
                    "source_resolution_method": operator["edit_target"].get(
                        "source_resolution_method"
                    ),
                    "kernel_backend": backend,
                },
                "tests": {
                    "cases_file": "tests/cases.json",
                    "driver_contract": {
                        "driver": "scripts/forge_driver.py",
                        "protected": True,
                        "case_selectors": case_ids,
                        "selector_flag": "--shape CASE_ID=<id>",
                        "correctness_invocation": "python scripts/forge_driver.py",
                        "bench_invocation": "python scripts/forge_driver.py --bench-mode",
                        "stdout_keys": [
                            "SNR",
                            "allclose",
                            "case_snr",
                            "case_allclose",
                            "case_ms",
                            "mean_ms",
                        ],
                    },
                },
                "targets": {
                    "snr_db": operator["gates"]["snr_db"],
                    "correctness_gate": "the worst-case SNR over every case must be at or above the threshold",
                    # **Not a number.** The standalone baseline lives in
                    # `evidence/performance.json` and is the only figure a speedup may be
                    # computed against; Magpie's `Avg time (us)` is an in-service average
                    # across mixed batch sizes and contention, and the two can differ
                    # severalfold. Naming the file instead of copying a number is what
                    # stops the wrong one being used.
                    "speedup_baseline": "evidence/performance.json, impl=baseline",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (directory / "forge_task.yaml").write_text(
        _task_yaml(operator, definition, backend), encoding="utf-8"
    )
    (directory / "scripts" / "forge_driver.py").write_text(
        _driver_source(operator_id, case_ids, _kernel_rel(operator)),
        encoding="utf-8",
    )
    (directory / "forge_program.md").write_text(
        _PROGRAM.format(operator_id=operator_id, kernel_rel=_kernel_rel(operator), backend=backend),
        encoding="utf-8",
    )
    one_line = directory / "run_forge.sh"
    # Single-quoted for the shell, because both values come from the workset and
    # a path with a space would otherwise split into two arguments.
    one_line.write_text(
        _ONE_LINE.format(
            operator_id=operator_id,
            backend_arg=_sh_quote(backend),
            kernel_arg=_sh_quote(_kernel_rel(operator)),
        ),
        encoding="utf-8",
    )
    one_line.chmod(0o755)

    return {
        "one_line": f"operators/{operator_id}/run_forge.sh",
        "driver": f"operators/{operator_id}/scripts/forge_driver.py",
        "invocation_spec": f"operators/{operator_id}/invocation_spec.json",
        "task": f"operators/{operator_id}/forge_task.yaml",
        "program": f"operators/{operator_id}/forge_program.md",
        "cases": f"operators/{operator_id}/tests/cases.json",
        "generated_from": ["definition", "workload"],
        "kernel_backend": backend,
    }


#: Known identifiers from Hyperloom's kernel_backends/constants.py. The session
#: preflight validates its selected registry; unknown names must not fall back.
#: New worksets emit kernel_backend. Legacy -fellow tags and environment
#: overrides are accepted only while reading older input contracts.
_FORGE_BACKENDS = (
    "assembly",
    "ck",
    "flydsl",
    "triton",
    "gluon",
    "aiter",
    "hip",
    "hipblaslt",
    "fusion",
)


def _sh_quote(value: str) -> str:
    """Single-quote a workset value for the generated `sh`."""
    return "'" + str(value).replace("'", "'\\''") + "'"


def _kernel_rel(operator: dict) -> str:
    """`--kernel`, in the frame forge resolves it in: relative to `--workspace`.

        **Measured by m4 on the live tree, and the reason this is not a
        pass-through.** `edit_target.source_file` is repo-root-framed
        (`sglang/python/sglang/kernels/…`) while `--workspace` is a checkout of
        `SGLANG_ROOT` itself, under which the file is `kernels/…`. Handing the
        recorded string straight to forge names a path that does not exist there.
        Third time this frame has bitten this package.

        **A wrong value fails loudly.** An absolute `--kernel` does *not* silently
        ignore `--workspace`, edit outside the tree and leave `git diff` empty:
        `_relative_file` validates the value before the
        `(workspace / kernel_path)` join that suggested the hazard: it accepts
        either frame, resolves both sides, raises *"kernel must be inside
        workspace"* on a path outside it, and raises again if the path is not a
        file. **Both frames accepted, containment enforced, existence enforced.**

        **The checkout that decides is `<KernelForge checkout>`** —
    `30_run_forge.sh` installs from it per container, and the old Forge CLI is
        not in the image — where the guard is `campaign_config.py`, called for
        the kernel at `:527`. Cited because the apparent disagreement about this
        behaviour was two people reading two different copies: line numbers here
        are hyperloom's, not the `chaojhou` tree's, whose numbers differ by roughly
        two hundred and which nothing installs from.

        A claim that arrives through a message rather than through a tool or an
        artefact has no independent check behind it. It is written out here rather
        than deleted because the fix it justified is still the right one for a
        different reason: the wrapper's own check turns forge's
        exception, raised from inside a campaign that has already booked the node,
        into a refusal that names both frames before anything starts.

        The tail is derived from `container_roots.yaml` rather than trimmed as a
        literal, per m4: a literal `sglang/python/sglang/` is correct today and
        wrong the first time `SGLANG_ROOT` moves. A path already carrying its
        placeholder goes through `patchkit.split_placeholder`, which is the same
        conversion `apply.py` uses; the recorded form does not carry one, so the
        root's own trailing components are matched instead — longest first.
    """
    target = operator.get("edit_target") or {}
    source = str(target.get("source_file") or "").strip()
    if not source:
        return ""
    if source.startswith("@"):
        return patchkit.split_placeholder(source)[1]
    root_path = patchkit.CONTAINER_ROOTS.get(str(target.get("repo_root_var") or "").strip())
    if not root_path:
        return source
    parts = [p for p in root_path.strip("/").split("/") if p]
    for start in range(len(parts)):
        prefix = "/".join(parts[start:]) + "/"
        if source.startswith(prefix):
            return source[len(prefix) :]
    return source


def _kernel_backend(operator: dict, definition: dict) -> str:
    """Select a maintained backend, accepting legacy input tags without emitting them."""
    tags = [t for t in (definition.get("tags") or []) if isinstance(t, str)]
    named = [t[: -len("-fellow")] for t in tags if t.endswith("-fellow")]
    backends: list[str] = []
    for candidate in named + tags:
        if candidate in _FORGE_BACKENDS and candidate not in backends:
            backends.append(candidate)
    if len(backends) == 1:
        return backends[0]
    # **The override, and it is second rather than first on purpose.** One
    # `--var` cannot be the primary mechanism: a single real workset carries
    # Triton *and* HIP operators, so a launch-wide value would be correct for
    # some of them and a substitution for the rest — the defect this replaced.
    # It exists for the case the tags cannot settle: none, or two.
    override = (
        os.environ.get("E2E_FORGE_BACKEND") or os.environ.get("E2E_FORGE_FELLOW", "")
    ).strip()
    backend = override[: -len("-fellow")] if override.endswith("-fellow") else override
    return backend if backend in _FORGE_BACKENDS else ""


def _task_yaml(operator: dict, definition: dict, backend: str) -> str:
    import yaml

    primary = next((s for s in operator["shapes"] if s.get("is_primary")), operator["shapes"][0])
    return yaml.safe_dump(
        {
            "task_id": operator["operator_id"],
            "gpu_target": operator.get("_gpu_arch", ""),
            "kernel_backend": backend,
            "generated_from": ["definition", "workload"],
            "shapes": {
                "primary": {
                    "CASE_ID": primary["case_id"],
                    **{k.upper(): v for k, v in primary["axes"].items()},
                },
                "validation": [
                    {"CASE_ID": s["case_id"], **{k.upper(): v for k, v in s["axes"].items()}}
                    for s in operator["shapes"]
                    if s["case_id"] != primary["case_id"]
                ],
            },
            "targets": {
                "snr_db": operator["gates"]["snr_db"],
                "baseline_source": "evidence/performance.json, impl=baseline",
            },
            "edit_target": operator["edit_target"],
        },
        sort_keys=False,
    )


def export_all(root: pathlib.Path) -> int:
    import yaml

    path = root / "workset.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    arch = (
        ((document.get("ground_truth") or {}).get("environment") or {})
        .get("fixed", {})
        .get("gpu_arch", "")
    )
    for operator in document["operators"]:
        operator["_gpu_arch"] = arch
        operator["forge"] = export_operator(root, operator)
        del operator["_gpu_arch"]
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    print(f"forge_export: {len(document['operators'])} operator(s) exported")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workset", required=True, help="the workset root, i.e. <handoff>/items/codes")
    return export_all(pathlib.Path(ap.parse_args(argv).workset))


if __name__ == "__main__":
    sys.exit(main())
