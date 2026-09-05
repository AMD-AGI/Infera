#!/usr/bin/env python3
"""Generate everything about a workset that is not judgement, and stop there.

`build_workset` is the one AI task in this stage, and the mission is explicit
about how much freedom it should have: *"每一步都必须有可执行的脚本和验收标准。ai
只需要串起来工作，减少 ai 的自由度和认知负载"* (G4.2). This program is the
executable half. It writes:

* `workset.yaml`, complete, from `operator_identity` and `profiling_evidence`;
* `definitions/<op_type>/<name>.json`, complete **except** `reference` and
  `baseline`, which are left as the sentinel below;
* `workloads/<op_type>/<name>.jsonl`, one line per observed shape, in the order
  `workset.yaml` indexes them — `check_workset_shape` checks that
  correspondence, and generating both from one loop is why it holds;
* `run_correctness.sh`, `run_performance.sh` and `_common.py`, copied verbatim
  from `harness/`. **Never generated, never edited**: an agent that writes its
  own oracle controls its own result.
* `environment.yaml`, carried through from the input.

What is left for the agent is exactly two things per operator, and both are
things a rule table cannot do:

1. **`reference`** — an implementation that is obviously right, imported from
   the framework's own test suite wherever one exists. A reference written from
   a reading of the kernel is how a correctness gate comes to agree with a bug.
2. **`baseline`** — the incumbent fast implementation, the call the served
   engine actually makes today. Separate from the reference, and conflating the
   two is the single most common way a speedup number becomes meaningless.

`gates.extra` is a third, optional and usually empty.

Idempotent: run it twice and the second run overwrites the scaffold and
**leaves any Definition whose sentinel has already been replaced**, so an agent
that has written two of three references does not lose them by re-running.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_PACKAGE = Path(os.environ.get("AGENT_SYS_TASK_PACKAGE") or os.environ["AGENT_SYS_DEMO_PACKAGE"])
sys.path.insert(0, str(_PACKAGE / "assets" / "lib"))

import schema as schema_lib  # noqa: E402
import store  # noqa: E402
import workset_io as W  # noqa: E402

HARNESS = _PACKAGE / "assets" / "build_workset.task" / "harness"

#: What an unwritten Definition body says. Chosen so that **every** way of
#: getting it wrong is caught: `check_workset_shape` rejects it as a template
#: marker (`TODO`), the harness rejects it as defining no `run`, and a reader
#: sees the instruction rather than an empty string that looks finished.
SENTINEL = (
    "# TODO(build_workset): replace this whole string with the implementation.\n"
    "# It must end in `def run(*args, **kwargs)`. See the task readme, STEP 4.\n"
)

_OP_TYPE = re.compile(r"[^a-z0-9_]+")

#: One-line form of `SENTINEL`, for list fields where a two-line comment block
#: would be nonsense. Same `TODO(build_workset)` token, so the same three
#: checks catch it.
SENTINEL_LINE = "TODO(build_workset): STEP 4a — state an invariant a replacement may not break."


def _op_type(operator: dict) -> str:
    """flashinfer-bench's `op_type`, from the category the ranker assigned.

    The category is a taxonomy label and `op_type` is a directory name, so this
    normalises rather than trusting: `moe_gemm` is fine, `Elementwise ops` is
    not, and the two must not diverge silently into two directories.
    """
    raw = (operator.get("category") or "unknown").strip().lower()
    return _OP_TYPE.sub("_", raw).strip("_") or "unknown"


def _axes(cases: list[dict]) -> tuple[dict, list[dict]]:
    """`(definition axes, per-shape var axes)` from the worklist's cases.

    An axis that takes the same value in every case is `const` and lives in the
    Definition; one that varies is `var` and appears per line in the workload.
    That split is flashinfer-bench's and it is what keeps the JSONL short enough
    to read — and it is derived from the data rather than declared, so a
    "constant" that is not constant cannot be asserted into one.
    """
    keys: dict[str, set] = {}
    for case in cases:
        for name, value in (case.get("selector") or {}).items():
            if name == "CASE_ID" or not isinstance(value, int):
                continue
            keys.setdefault(name, set()).add(value)
    axes, per_shape = {}, []
    for name, values in sorted(keys.items()):
        if len(values) == 1:
            axes[name.lower()] = {"type": "const", "value": next(iter(values)),
                                  "description": f"{name}, constant across every observed shape"}
        else:
            axes[name.lower()] = {"type": "var", "description": f"{name}, varies across the observed shapes"}
    for case in cases:
        per_shape.append({
            name.lower(): value
            for name, value in (case.get("selector") or {}).items()
            if name != "CASE_ID" and isinstance(value, int) and axes.get(name.lower(), {}).get("type") == "var"
        })
    return axes, per_shape


def _uuid(operator_id: str, case_id: str) -> str:
    """A stable 16-hex id for one shape.

    Derived from the names rather than random, so re-running the scaffold does
    not invalidate `shapes[].uuid` in a `workset.yaml` an agent has since
    edited — the correspondence between the index and the JSONL is exactly what
    `check_workset_shape` checks, and a fresh random id would break it on the
    second run.
    """
    import hashlib

    return hashlib.sha256(f"{operator_id}/{case_id}".encode()).hexdigest()[:16]


def _magpie_row(operator: dict) -> dict:
    """Magpie's own row for this kernel (M3.7.3 — 尽量不自己搞).

    Reconstructed from the fields `rank` carried through under Magpie's own
    column names, so a rename upstream surfaces as a schema failure here rather
    than as a number that quietly means something else.
    """
    return {
        "Name": operator["name"],
        "Calls": operator.get("calls") or 0,
        "Self CUDA total (us)": operator.get("self_us") or 0.0,
        "Avg time (us)": operator.get("avg_us") or 0.0,
        "% Total": operator.get("pct_total") or 0.0,
        "Input Shapes": operator.get("input_shapes") or "",
    }


def _entrypoints(operator_id: str | None) -> dict:
    suffix = f" --operator {operator_id}" if operator_id else ""
    tag = operator_id or "all"
    return {
        "correctness": {"cmd": f"./run_correctness.sh{suffix}",
                        "report": f"evidence/correctness.{tag}.json", "protected": True},
        "performance": {"cmd": f"./run_performance.sh{suffix}",
                        "report": f"evidence/performance.{tag}.json", "protected": True,
                        "timeout_s": 1800,
                        # What `--impl` must be, stated rather than inferred
                        # from the harness. One authority in `assets/lib/`,
                        # copied into the artefact so a consumer holding only
                        # the workset has it.
                        "impl_contract": W.IMPL_CONTRACT},
    }


def _environment(staged: Path) -> tuple[dict, str]:
    for candidate in (staged / "items/env/environment.yaml",
                      staged / "items/codes/environment.yaml",
                      staged / "items/result/environment.yaml"):
        if candidate.is_file():
            import yaml

            text = candidate.read_text(encoding="utf-8")
            return yaml.safe_load(text) or {}, text
    raise SystemExit(f"{staged.name} carries no environment.yaml; CONTRACT.md 2 requires one on every kind")


def main() -> int:
    import yaml

    identity_dir = store.declared_dir("operator_identity", direction="INPUT")
    if identity_dir is None:
        raise SystemExit("AGENT_SYS_INPUT_OPERATOR_IDENTITY does not name a readable directory.")
    evidence_dir = store.declared_dir("profiling_evidence", direction="INPUT")
    if evidence_dir is None:
        raise SystemExit("AGENT_SYS_INPUT_PROFILING_EVIDENCE does not name a readable directory.")

    identity = json.loads((identity_dir / "items" / "text.json").read_text(encoding="utf-8"))
    environment, environment_text = _environment(identity_dir)

    dst = Path(os.environ.get("AGENT_SYS_OUTPUT_OPERATOR_WORKSET") or "")
    if not dst:
        raise SystemExit("AGENT_SYS_OUTPUT_OPERATOR_WORKSET is not set; this body has nowhere to write.")
    root = dst / "items" / "codes"
    root.mkdir(parents=True, exist_ok=True)

    for name in ("_common.py", "run_correctness.sh", "run_performance.sh"):
        shutil.copy2(HARNESS / name, root / name)
        if name.endswith(".sh"):
            (root / name).chmod(0o755)
    (root / "environment.yaml").write_text(environment_text, encoding="utf-8")

    operators, kept, written = [], 0, 0
    entry_dtypes: dict[str, str] = {}
    for entry in identity["operators"]:
        operator_id = entry["logical_operator"]
        op_type = _op_type(entry)
        cases = entry.get("cases") or []
        if len(cases) < 3:
            print(f"warning: {operator_id} has {len(cases)} observed shape(s); M3.7.4.1.2 needs 3. "
                  f"STEP 5 of the readme is where you add the missing ones and mark them observed: false",
                  file=sys.stderr)
        axes, per_shape = _axes(cases)

        definition_rel = f"definitions/{op_type}/{operator_id}.json"
        workload_rel = f"workloads/{op_type}/{operator_id}.jsonl"
        (root / definition_rel).parent.mkdir(parents=True, exist_ok=True)
        (root / workload_rel).parent.mkdir(parents=True, exist_ok=True)

        shapes, lines = [], []
        for case, var_axes in zip(cases, per_shape):
            uuid = _uuid(operator_id, case["case_id"])
            shapes.append({
                "case_id": case["case_id"], "uuid": uuid, "axes": var_axes or {"batch": 1},
                # Filled below, once every shape is known: the rule reads the
                # whole list. Keying it on `is_primary` alone — which is what
                # this line did until m4 found it — yields exactly one timed
                # shape per operator by construction, and m4's packup refuses at
                # three. `assign_roles` is shared with the validator so the
                # producer cannot be the looser of the two readers.
                "role": None,
                "is_primary": bool(case.get("is_primary")), "observed": True,
                "observed_shapes": case.get("shapes") or [],
                "calls": entry.get("calls") or 0,
            })
            lines.append(json.dumps({
                "definition": operator_id,
                "workload": {"axes": var_axes or {"batch": 1},
                             "inputs": {name: {"type": "random"} for name in axes if False} or {},
                             "uuid": uuid},
                "solution": None, "evaluation": None}))
        (root / workload_rel).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        for shape, role in zip(shapes, W.assign_roles(shapes)):
            shape["role"] = role
        timed = sum(1 for s in shapes if W.is_performance(s["role"]))
        if timed < W.PERFORMANCE_FLOOR:
            print(f"warning: {operator_id} has {timed} performance-measured shape(s); "
                  f"M3.7.4.1 needs {W.PERFORMANCE_FLOOR} and m4's packup refuses below it. "
                  f"STEP 5 of the readme is where you add the missing ones",
                  file=sys.stderr)

        # Idempotence: a Definition whose bodies the agent has already written is
        # left exactly as it is. Only the scaffolded half is refreshed.
        target = root / definition_rel
        existing = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}
        if existing.get("reference", SENTINEL) != SENTINEL and existing.get("baseline", SENTINEL) != SENTINEL:
            kept += 1
        else:
            target.write_text(json.dumps({
                "name": operator_id,
                "op_type": op_type,
                "axes": axes or {"batch": {"type": "var", "description": "batch"}},
                "inputs": {}, "outputs": {},
                "reference": existing.get("reference") or SENTINEL,
                "baseline": existing.get("baseline") or SENTINEL,
                "tags": [t for t in (op_type, entry.get("precision"), entry.get("fellow")) if t],
                "description": f"{operator_id}, from {entry['name'][:80]}",
            }, indent=2) + "\n", encoding="utf-8")
            written += 1

        entry_dtypes[operator_id] = (entry.get("dtypes") or {}).get("activation") or entry.get("precision") or ""
        operators.append({
            "operator_id": operator_id,
            "kernel_id": entry["kernel_id"],
            "device_symbol": entry["name"],
            "op_type": op_type,
            "status": "complete" if entry.get("target_kernel_functions") else "partial",
            "missing_fields": [] if entry.get("target_kernel_functions") else ["edit_target.entry_function"],
            "definition": definition_rel,
            "workload": workload_rel,
            "shapes": shapes,
            "entrypoints": _entrypoints(operator_id),
            "reference": {"kind": "written", "path": f"operators/{operator_id}/reference.md",
                          "rationale": "TODO(build_workset): STEP 4."},
            "baseline": {"kind": "written", "path": f"operators/{operator_id}/baseline.md",
                         "rationale": "TODO(build_workset): STEP 4."},
            "edit_target": {
                "source_owner": (entry.get("kernel_identity") or {}).get("source_owner", ""),
                "repo_root_var": entry.get("image_repo_path") or "",
                "source_file": (entry.get("source_file_path") or [""])[0],
                "editable_sources": entry.get("editable_sources") or [],
                "entry_function": (entry.get("target_kernel_functions") or [""])[0],
                "source_resolution_method": entry.get("source_resolution_method"),
                "resolution_evidence": entry.get("resolution_evidence") or entry.get("resolution_hint") or "",
                # **What the identity said, recorded beside what was derived
                # from it.** `source_file` above is `source_file_path[0]`
                # verbatim, and nothing could confirm it had stayed that way:
                # no phase in this graph stages the identity and the workset
                # together, so a validator comparing them binds to nothing
                # (measured 2026-09-04 — the two-kind validator was written and
                # selected nowhere). Recording the source turns a question that
                # needs a second handoff into one that needs a comparison,
                # which is what `base_sha256` already does for the image.
                #
                # It matters because at rung 3 this scaffold is not the author:
                # the agent runs it at STEP 2 and edits afterwards.
                "from_identity": {
                    "source_file_path": list(entry.get("source_file_path") or []),
                    "kernel_id": entry.get("kernel_id"),
                },
            },
            # M5.1.1. Scaffolded from `edit_target` because the *file* is
            # already known; `public_symbol` and `invariants` are the agent's,
            # and STEP 4a is where they are filled. They are deliberately left
            # as sentinels rather than guessed: an invariant nobody checked is
            # worse than a missing one, because m5 will rely on it.
            "integration": {
                "target_files": entry.get("editable_sources") or ([entry["source_file_path"][0]]
                                                                  if entry.get("source_file_path") else []),
                # **Both of these came from `target_kernel_functions[0]`**, so
                # the real path asserted a *method qualname* as the symbol an
                # overlay installs — `Sampler.forward` is not a module-level
                # function and cannot be swapped by name. Different surface from
                # the mock's contradiction, same wrong premise: that every
                # operator has a named substitution target.
                #
                # `identify` decides now, because it is the step that reads the
                # image. Absent, the kind is unstated rather than guessed, and
                # `check_workset_shape` says so.
                "substitution": entry.get("substitution"),
                "module_symbols": entry.get("module_symbols"),
                "public_symbol": entry.get("public_symbol"),
                "signature": "",
                "invariants": [SENTINEL_LINE],
                # **Carried from `identify`, never computed here.** The schema
                # is explicit that this is pinned at identification time: a
                # hash taken later is a hash of whatever the file had become.
                # `null` when identify could not read the image, which the
                # schema permits and which obliges m4 to hash and say it did —
                # a stated gap, not a silent one.
                "base_sha256": entry.get("base_sha256"),
                # Derived from `substitution`, not hardcoded: the schema
                # refuses `call_site_fragment` with `overlay_files`, so a
                # constant here would fail this task's own output
                # validation on a fragment operator. `W.apply_mode_for`
                # is shared with `mock_adapt.py`, the other producer.
                "apply_mode": W.apply_mode_for(entry.get("substitution")),
                "requires_restart": True,
                "build_step": None,
            },
            "gates": {"snr_db": float(os.environ.get("E2E_SNR_THRESHOLD") or 30.0)},
            # Transcribed from evidence/performance.json at STEP 8; 1.05 until
            # then, which is the previous round's rule of thumb and is a
            # placeholder rather than a measurement.
            "noise_floor": 1.05,
            # What must travel byte-identically for the entrypoints to run
            # outside this handoff. m4 re-runs from its own packup's copy.
            "apparatus": ["_common.py", "run_correctness.sh", "run_performance.sh",
                          "workset.yaml", definition_rel, workload_rel],
            "provenance": {
                "source": (identity.get("resolver") or {}).get("profile")
                or f"profiling_evidence {evidence_dir.name}",
                "rank": entry.get("rank"),
                "pct_total": entry.get("pct_total"),
                "in_service_avg_us": entry.get("avg_us"),
                "magpie_row": _magpie_row(entry),
            },
        })

    document = {
        "schema_version": 1,
        "workset_id": os.environ.get("E2E_WORKSET_ID") or "workset",
        "produced_by": {"package": "e2e-flow",
                        "commit": os.environ.get("E2E_PACKAGE_COMMIT") or "unknown",
                        "step": "build_workset",
                        "produced_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "ground_truth": {
            "abort_on_mismatch": ["gpu_arch", "gpu_count", "tp_size", "dtype"],
            "warn_on_mismatch": ["image_id", "rocm", "torch"],
            # A summary of the Definitions, lifted so a consumer holding only
            # `ground_truth` can compare it. `check_workset_shape` checks it
            # against `inputs[].dtype`, so it cannot drift.
            "dtypes": {o["operator_id"]: (entry_dtypes.get(o["operator_id"]) or "") for o in operators},
            # The workset-wide bar: the largest operator floor. A placeholder
            # until STEP 8 transcribes the measured figures.
            "noise_floor": max((o["noise_floor"] for o in operators), default=1.05),
            "environment": environment},
        "entrypoints": _entrypoints(None),
        "protocol": {"groups": 5, "iters_per_group": 10, "warmup": 3, "timing": "wall_clock_sync"},
        "operators": operators,
    }
    (root / "workset.yaml").write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    print(f"scaffold: {len(operators)} operator(s); {written} definition(s) scaffolded, {kept} left as written")
    print(f"          next: STEP 4 of the readme — fill `reference` and `baseline` in "
          f"{', '.join(o['definition'] for o in operators)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
