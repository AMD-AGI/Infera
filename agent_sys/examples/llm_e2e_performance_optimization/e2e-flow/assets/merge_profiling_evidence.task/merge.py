#!/usr/bin/env python3
"""Fold stage 2's four artefacts into its one export.

Mission M2.9: *"bench_result、profiling result、magpie standardized output
result，三者整合进一个大的统一 handoff"*. m3 and m5 consume `profiling_evidence`
rather than the four pieces, so this is the boundary of stage 2.

**It carries both benches.** m5's stock arm has to reproduce m2's numbers
(M5.1.3.1), and the number it must reproduce is the profiler-detached one —
comparing a served deployment against a profiled one would be comparing two
different machines.

The layout, which `check_profiling_evidence` reads and which is the whole
contract of this task::

    items/result/bench_profiling_mode_off/{result,env}/…
    items/result/bench_profiling_mode_on/{result,env}/…
    items/result/trace/{manifest.json,traces/,stacks/,stacks_manifest.json}
    items/result/kernel_table/{text.json,table.csv,schema}
    items/env/environment.yaml     one record, inherited
    items/env/parts.json           which handoff each part came from
    items/command                  how to rebuild it

`parts.json` is the piece that is not a copy. Each part's own environment record
is lifted into it, so the merge can be asked afterwards whether the four parts
describe one deployment — which is a question only this task's inputs can
answer, and which none of the four lines' own validators can see.

**There is no mock path.** The four inputs are mocked upstream when the stage is
mocked, so this runs for real in every mode. That is deliberate: a hand-shaped
`profiling_evidence` would make the cross-part rules grade a stand-in, and those
rules are the only reason this handoff exists rather than four.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import schema as schema_lib  # noqa: E402 — the path insert above is what makes it importable

#: part name in the export -> (input environment variable, what to lift out of it)
#:
#: The part names are `check_profiling_evidence`'s `require_parts` and m3's read
#: path. They are spelled here once.
PARTS = {
    "bench_profiling_mode_off": ("AGENT_SYS_INPUT_PROFILING_MODE_OFF_BENCH_RESULT", "reproducible"),
    "bench_profiling_mode_on": ("AGENT_SYS_INPUT_PROFILING_MODE_ON_BENCH_RESULT", "reproducible"),
    "trace": ("AGENT_SYS_INPUT_PROFILING_MODE_ON_PROFILE_RESULT", "reproducible"),
    "kernel_table": ("AGENT_SYS_INPUT_PROFILING_MODE_ON_KERNEL_TABLE", "structured_text"),
}

#: What a part contributes to `items/result/<part>/`, and **the shape is one rule
#: for all four**: a part's content lands directly under its own directory, with
#: its environment beside it as `env/`.
#:
#:     items/result/bench_profiling_mode_off/{summary.json,profile_export_*,env/}
#:     items/result/trace/{manifest.json,traces/,stacks/,env/}
#:     items/result/kernel_table/{text.json,table.csv,schema,env/}
#:
#: A `reproducible` part's `result/` is therefore **flattened** rather than
#: nested. Nesting it was the first shape and `check_profiling_evidence` caught
#: it: the validator looked for `trace/manifest.json` and the merge had written
#: `trace/result/manifest.json`. One rule for four parts is what stops a
#: consumer having to know which content type each part used to be.
#:
#: `env/` travels for every part because the load configuration in it is what
#: makes the two benches comparable, and is not recoverable from the numbers.
LIFT = {
    "reproducible": [("result", "."), ("env", "env")],
    "structured_text": [("text.json", "text.json"), ("table.csv", "table.csv"),
                        ("schema", "schema"), ("env", "env")],
}

ENV_REL = {
    "reproducible": "items/env/environment.yaml",
    "structured_text": "items/env/environment.yaml",
    "code": "items/codes/environment.yaml",
}

README = """# profiling_evidence

## Purpose

Stage 2's single export, and the boundary of the stage: both benches, the
profiler trace, and the kernel ranking, under one environment record. Stages 3
and 5 read this rather than the four artefacts it folds, so that neither can be
handed three of the four and not notice.

The two benches are the same load against the same deployment with one switch
flipped. `bench_profiling_mode_off` ran with decode CUDA graphs on and no
profiler attached, and **its numbers are the ones worth quoting** — stage 5's
stock arm has to reproduce them. `bench_profiling_mode_on` ran with graphs off
and the profiler attached, which is what makes individual kernels attributable
and is also why its throughput is not a control for anything.

## How to run

`items/command` rebuilds this handoff from the four it folds. It is a merge, not
a measurement: re-running it does not re-measure anything, and the two lines
above are what produce the inputs.

## Result

`items/result/bench_profiling_mode_off/` and `.../bench_profiling_mode_on/` each
carry that line's AIPerf exports under `result/` and the load's own
configuration under `env/`. `items/result/trace/` carries the per-rank profiler
traces and the manifest that describes them. `items/result/kernel_table/` carries
every kernel ranked by the CUDA time it owns, as `text.json`, with Magpie's own
export beside it as `table.csv` and the schema it validates against as `schema`.

## Environment

`items/env/environment.yaml` is the one record, inherited from stage 1 unchanged
except for the runtime half. `items/env/parts.json` records which handoff each
part came from and what environment that part was taken in, which is what makes
"these four describe one deployment" a checkable claim rather than an assumption
of the merge.

## Watch out

See `items/watchout`. In short: the two benches are not two samples of one
quantity, the trace's own window is seconds out of a minutes-long load, and the
ranking pools prefill and decode.
"""

WATCHOUT = """The two benches are NOT two samples of one quantity. They differ by
one deliberate switch and measured 8x apart on the reference run (15.65 ms mean
inter-token latency against 124.98 ms). Averaging them, or quoting the profiled
one as a throughput, describes nothing that exists.

The trace is a window of seconds cut out of a load of minutes. It is
representative of the steady state and it is not the whole run, so a share in
the kernel ranking is a share of the window and not of the benchmark.

This is a MIX deployment: prefill and decode run in one process and the ranking
pools both kinds of kernel. They cannot be separated by role after the fact.

`items/result/trace/stacks/` is a second, shorter capture taken with Python
stacks on, and it exists only so the ranking can name the frame that launched
each kernel. It is not a measurement and must not be aggregated with the
measurement window.

Every environment record here is inherited from stage 1. If a later stage finds
the deployment differs, the difference belongs in that record's `warnings`, not
in a second record — one flow, one environment document.
"""


def die(message: str) -> None:
    print(f"merge: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_env(content: Path, content_type: str) -> dict:
    """The identity `check_profiling_evidence` compares parts by."""
    import yaml

    path = content / ENV_REL[content_type]
    if not path.is_file():
        die(f"{content} carries no environment record at {ENV_REL[content_type]}")
    doc = yaml.safe_load(path.read_text())
    if not isinstance(doc, dict):
        die(f"{path} is not an environment record")
    fixed, runtime = doc.get("fixed") or {}, doc.get("runtime") or {}
    return {
        "node": fixed.get("node"),
        "image_id": fixed.get("image_id"),
        "container": runtime.get("container"),
        "endpoint": runtime.get("endpoint"),
        "started_at": runtime.get("started_at"),
    }


def main() -> int:
    out = Path(os.environ.get("AGENT_SYS_OUTPUT_PROFILING_EVIDENCE") or die("no output slot"))
    items = out / "items"
    (items / "result").mkdir(parents=True, exist_ok=True)
    (items / "env").mkdir(parents=True, exist_ok=True)

    rows = []
    env_source: Path | None = None
    env_source_type = "reproducible"

    for part, (var, content_type) in PARTS.items():
        src = os.environ.get(var)
        if not src:
            die(f"{var} is unset — the closure does not declare this kind as an input")
        content = Path(src)
        if not content.is_dir():
            die(f"{var} names {src}, which is not a directory")

        dst = items / "result" / part
        dst.mkdir(parents=True, exist_ok=True)
        lifted = []
        for name, where in LIFT[content_type]:
            source = content / "items" / name
            if not source.exists():
                die(f"{part}: items/{name} is missing from {src}")
            target = dst if where == "." else dst / where
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            lifted.append(name)

        rows.append({
            "name": part,
            "source_kind": var.removeprefix("AGENT_SYS_INPUT_").lower(),
            "content_type": content_type,
            # Which of the two bring-ups this part came from. It is not derivable
            # from the part name — `trace` and `kernel_table` are both the
            # profiler-attached line — and `check_profiling_evidence` needs it to
            # know which parts must share a container and which must not.
            "line": "profiling_mode_off" if part.endswith("profiling_mode_off") else "profiling_mode_on",
            "lifted": lifted,
            "environment": load_env(content, content_type),
        })
        # Any part's record will do — `check_profiling_evidence` is what proves
        # they agree, and taking one arbitrarily is what gives that rule
        # something to disagree with. Taking the *profiler-detached* bench is
        # deliberate: it is the deployment stage 5 must reproduce.
        if part == "bench_profiling_mode_off":
            env_source, env_source_type = content / ENV_REL[content_type], content_type

    (items / "env" / "parts.json").write_text(
        json.dumps({"schema_version": 1, "parts": rows}, indent=2), encoding="utf-8"
    )

    # The environment record, inherited. `env_render.py` validates before it
    # writes, so a merge that produced a malformed record fails here rather than
    # shipping something that reads like one.
    pkg = os.environ.get("AGENT_SYS_TASK_PACKAGE") or os.environ.get("AGENT_SYS_DEMO_PACKAGE")
    if env_source is None:
        die("no part carried an environment record to inherit")
    rc = subprocess.run(
        [sys.executable, f"{pkg}/assets/lib/env_render.py", "--inherit", str(env_source),
         "--content-type", "reproducible", "--out", str(out)]
    ).returncode
    if rc:
        die("env_render refused the inherited environment record")

    (items / "command").write_text(
        "#!/usr/bin/env bash\n"
        "# Rebuild this handoff from the four it folds. A merge, not a measurement:\n"
        "# re-running it re-copies and re-checks, and measures nothing.\n"
        "#\n"
        "# Executable because agent.gate requires it of a 'command' item, and written\n"
        "# with shell variables rather than absolute paths so the locality seal has\n"
        "# nothing to reject.\n"
        "set -eu\n"
        ': "${SCRIPTS:?export SCRIPTS=<the package assets directory>}"\n'
        + "".join(
            f': "${{{var}:?export {var}=<the {part} handoff content directory>}}"\n'
            for part, (var, _) in PARTS.items()
        )
        + ': "${AGENT_SYS_OUTPUT_PROFILING_EVIDENCE:?export it to the target directory}"\n'
        'python3 "$SCRIPTS/merge_profiling_evidence.task/merge.py"\n',
        encoding="utf-8",
    )
    (items / "command").chmod(0o755)
    (items / "watchout").write_text(WATCHOUT, encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")

    # The record m3 reads must validate here, not three tasks away. Cheap, and
    # the merge is the last point at which the producer can still be blamed.
    table = items / "result" / "kernel_table" / "text.json"
    try:
        schema_lib.validate("kernel_table", json.loads(table.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        die(f"the folded kernel table is not readable: {exc}")
    except schema_lib.SchemaError as exc:
        die(f"the folded kernel table does not validate:\n{exc}")

    print(f"merge: {len(rows)} part(s) -> {out}")
    for row in rows:
        print(f"  {row['name']:<26} {row['environment'].get('container')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
