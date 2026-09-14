#!/usr/bin/env python3
"""What `packup` runs: assemble the stage's deliverable.

Reads all four producing handoffs and writes the layout
`temp/claude_code_skill_used_by_human/experiment-result-packup` defines, so a
colleague receives the same shape they receive from every other experiment in
this programme.

Everything written here is derived from the handoffs rather than recomputed, so
the deliverable cannot disagree with the evidence it summarises.
"""

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_PACKAGE = Path(
    os.environ.get("AGENT_SYS_TASK_PACKAGE") or os.environ["AGENT_SYS_DEMO_PACKAGE"]
)
sys.path.insert(0, str(_PACKAGE / "assets" / "lib"))

import store  # noqa: E402

RUN_COMMAND = """AGENT_SYS_NO_PERMISSIONS=1 agent-sys run \\
  --package agent_sys/examples/llm_e2e_performance_optimization/analyze-demo \\
  --var jobid=<slurm job id> \\
  --var gpu_node=<node holding the allocation> \\
  --var sglang_src=<sglang checkout> \\
  --var aiter_src=<aiter checkout>
"""


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set; this body has nowhere to write.")
    return value


def _read(kind: str) -> dict:
    staged = store.declared_dir(kind, direction="INPUT")
    if staged is None:
        raise SystemExit(f"AGENT_SYS_INPUT_{kind.upper()} does not name a readable directory")
    return json.loads((staged / "items" / "text.json").read_text(encoding="utf-8"))


def main() -> int:
    worklist = _read("kernel_worklist")
    identity = _read("operator_identity")
    evidence = _read("workset_evidence")
    workset_dir = store.declared_dir("operator_workset", direction="INPUT")

    selected = [k for k in worklist["kernels"] if k.get("selected")]
    buckets = worklist["buckets"]
    environment = worklist["environment"]
    measured = {o["operator_id"]: o for o in evidence["operators"]}

    dst = Path(_required("AGENT_SYS_OUTPUT_ANALYZE_PACKUP"))
    items = dst / "items"
    for name in ("results", "logs", "code"):
        (items / name).mkdir(parents=True, exist_ok=True)

    # The worksets themselves are the substance of the deliverable, so they are
    # carried rather than referenced.
    if workset_dir is not None and (workset_dir / "items" / "code").is_dir():
        shutil.copytree(
            workset_dir / "items" / "code", items / "code", dirs_exist_ok=True
        )

    (items / "results" / "worklist.json").write_text(
        json.dumps(worklist, indent=2), encoding="utf-8"
    )
    (items / "results" / "identity.json").write_text(
        json.dumps(identity, indent=2), encoding="utf-8"
    )
    (items / "results" / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )

    bucket_rows = "\n".join(
        f"| `{name}` | {entry['kernels']} | {entry['pct_total']:.2f} |"
        for name, entry in sorted(buckets.items(), key=lambda kv: -kv[1]["pct_total"])
    )
    operator_rows = "\n".join(
        "| {rank} | `{op}` | {pct:.2f} | {lang} | {method} | {status} |".format(
            rank=row["rank"],
            op=ident["logical_operator"],
            pct=row["pct_total"],
            lang=ident["repository_language"],
            method=ident["source_resolution_method"],
            status=(
                f"{measured[ident['logical_operator']]['bench']['weighted_mean_ms']:.4f} ms"
                if measured.get(ident["logical_operator"], {}).get("bench")
                else measured.get(ident["logical_operator"], {}).get("failure", "not measured")[:40]
            ),
        )
        for row, ident in zip(
            sorted(selected, key=lambda r: r["rank"]),
            sorted(identity["operators"], key=lambda o: o["rank"]),
        )
    )

    (items / "result").write_text(
        f"Selected {len(selected)} operators from {worklist['source']['rows']} profiled "
        f"kernels; resolved {identity['summary']['resolved']} of "
        f"{identity['summary']['operators']} to a source file by symbol search; "
        f"{evidence['summary']['passed']} of {evidence['summary']['operators']} worksets "
        f"ran and measured cleanly on {evidence['environment']['node'] or 'the target node'}.\n",
        encoding="utf-8",
    )
    (items / "env").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    # `agent/gate.py:EXECUTABLE_ITEMS` is `{script, command, entry}`: an item
    # under one of those keys must satisfy `os.access(path, os.X_OK)` or the
    # seal is refused with `output_not_executable`. Writing the file is not
    # enough, and the refusal arrives only after the body has returned — which
    # is how it cost a fourteen-minute AI task its whole delivery once.
    command = items / "command"
    command.write_text("#!/bin/sh\n" + RUN_COMMAND, encoding="utf-8")
    command.chmod(0o755)
    (items / "watchout").write_text(
        "The profile behind this run is a GLM-5.2 1P1D decode capture, not "
        "GLM-5.3-Flash: the shapes are real, the operator mix is not the target "
        "model's.\n\n"
        "Measured times are standalone single-operator figures and are not "
        "comparable to the profile's serving-time averages.\n\n"
        "Container roots travel as ${PLACEHOLDER} tokens because the seal refuses "
        "absolute paths; resolve them against the image named in env.\n",
        encoding="utf-8",
    )

    readme = f"""# analyze_packup

## Purpose

The deliverable of the analyze stage: from a kernel-level profile to a
KernelForge workset per operator, with each workset's driver executed on the
target GPU before hand-over.

## How to run

```sh
{RUN_COMMAND}```

## Result

{items.joinpath('result').read_text(encoding='utf-8').strip()}

Profiled kernels by bucket:

| bucket | kernels | % of profiled GPU time |
|---|---|---|
{bucket_rows}

Selected operators:

| rank | operator | % GPU | language | resolution | measured |
|---|---|---|---|---|---|
{operator_rows}

## Environment

Target: `{environment['gpu_type']}` / `{environment['gpu_target']}`, framework
`{environment['framework']}`, image `{environment['image']}`.
Measurements were taken on `{evidence['environment']['node'] or 'the allocated node'}`
under the protocol in `results/evidence.json`: {evidence['protocol']['groups']} groups
of {evidence['protocol']['iters']} iterations after {evidence['protocol']['warmup']} warmup
iterations, combined by a weighted mean.

## Watch out

{items.joinpath('watchout').read_text(encoding='utf-8').strip()}
"""
    (dst / "README.md").write_text(readme, encoding="utf-8")

    reproduce = f"""# REPRODUCE

Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}.

## 1. Environment

Run `agent-sys` on Python 3.12 or newer. Below that the CLI does not import —
see `temp/bugs/001-requires-python-3.10-but-fails-below-3.12.md`.

```sh
python3.13 -m venv <venv> && <venv>/bin/pip install -e agent_sys
<venv>/bin/pip install -e "agent_sys[claude]"
```

## 2. Source trees

`identify` resolves a device symbol to the source that declares it by searching
indexed repositories. Extract them from the serving image so that what is
searched is what runs. The container roots are the `path` values in the
producing package's `assets/lib/container_roots.yaml`; they are not repeated
here because the seal refuses a handoff that names an absolute path outside a
portable allow-list.

```sh
CID=$(docker create {environment['image']} true)
docker cp "$CID:${{AITER_ROOT}}" <repos>/aiter
docker cp "$CID:${{SGLANG_ROOT}}" <repos>/sglang
docker rm -f "$CID"
```

## 3. Run

```sh
{RUN_COMMAND}```

## 4. What to expect

Six leaves in order: `seed_table`, `rank`, `identify`, `build_workset`,
`verify_workset`, `packup`. Six handoffs, each with one validator. The run is
complete when all six report `succeeded` and every handoff slot reads `valid`.

`verify_workset` is the only one needing a GPU, and it needs the allocation the
`--var jobid` names to still be held.
"""
    (items / "REPRODUCE.md").write_text(reproduce, encoding="utf-8")
    (items / "environment.md").write_text(
        "# Environment\n\n"
        + json.dumps(environment, indent=2)
        + "\n\nMeasured on: "
        + (evidence["environment"]["node"] or "(not recorded)")
        + "\nImage: "
        + environment["image"]
        + "\n\nThe repositories searched by `identify` were extracted from that image, so\n"
        "the source a symbol resolves to is the source the profiled binary was built\n"
        "from. Container roots appear as ${PLACEHOLDER} tokens throughout, because\n"
        "the handoff seal refuses absolute paths that are not portable.\n",
        encoding="utf-8",
    )
    (items / "notes.md").write_text(
        "# Notes\n\n"
        "## What this stage produces\n\n"
        "The measuring apparatus, not the thing measured. Kernel source stays in the\n"
        "framework and forge-loop edits it there; what it cannot make for itself is the\n"
        "driver that decides whether an edited kernel is still correct, and it treats\n"
        "that driver as a protected file.\n\n"
        "## Two findings worth carrying forward\n\n"
        "A device symbol resolves to its source when its compound name is kept\n"
        "contiguous and shortened from the right. Splitting it into tokens does not\n"
        "work: `quant` matches half of aiter. Details in `assets/lib/symbols.py`.\n\n"
        "Handoff content may not name an absolute path outside a small allow-list, so\n"
        "container roots travel as placeholders. The mechanism meant for this exists in\n"
        "the framework and is not wired up — see\n"
        "`temp/bugs/002-handoff-dependencies-never-reach-locality-check.md`.\n",
        encoding="utf-8",
    )

    for name in ("worklist.json", "identity.json", "evidence.json"):
        shutil.copy2(items / "results" / name, items / "logs" / name)

    print(
        f"packup: {len(selected)} operators, {evidence['summary']['passed']} measured, "
        f"-> {dst}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
