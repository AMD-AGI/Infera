#!/usr/bin/env python3
"""Compose one arm's `{stock,patched}.measurement` from the pieces that produced it.

**Six evidence kinds became two** (CONTRACT.md §7, a consequence of M5.2). In
`integration-demo` one arm was three handoffs — `deployment_<arm>`,
`acceptance_<arm>`, `bench_<arm>` — produced by two tasks. Here one task produces
one handoff per arm, and this module is the join: it takes the directories the
bring-up and the measurement scripts wrote and lays them out as a single
`reproducible` content tree.

**The union is mechanical, and exactly three things are not.**

*The collision rule.* Three files exist in more than one source — `README.md`,
`items/command` and `items/watchout` — because each script wrote a complete
handoff of its own. A plain `cp -a` of one directory over another would silently
keep the last. Everything else is copied, and a genuine collision (same path,
different bytes) is a failure rather than a last-writer-wins.

*`README.md` and `items/watchout` are concatenated* with a header naming where
each half came from. Both are prose; two halves of prose are prose.

*`items/command` is rewritten, not concatenated*, and that was a correction.
Three `command` scripts joined end to end describe bringing the arm up,
measuring it, bringing it up again and measuring it again — which is not what
happened and not something anybody should run. It also inherits every syntax
fault in every half: **measured, two of the three sealed stage-5 halves do not
parse**, so the concatenation did not either, and the package's
`command`-parses validator would have refused both arms. The merged handoff now
carries one script that reproduces the whole arm, and each half's original
verbatim under `items/logs/command/` — rewriting somebody else's script to make
it parse would be inventing history rather than recording it.

*The step order.* `env/steps.json` is the record `check_measurement_order` reads,
and no source carries the merged version: the measurement script knows about its
own five steps and nothing about the bring-up that preceded them. **The bring-up
has to be in it**, because the ordering guarantee that used to be the graph edge
`serve_patched ← measure_stock` is precisely "the patched arm's bring-up did not
start until the stock arm had finished measuring" — and a steps record that omits
bring-up cannot express it. So `--serve-started` and `--serve-seconds` are
required, and there is no default: a guessed bring-up window would make the
disjointness check pass by construction.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

ARMS = ("stock", "patched")

#: Written by every source because each was a whole handoff once. Concatenated
#: rather than overwritten.
#:
#: **`items/command` is deliberately not in this list**, and taking it out was a
#: correction. Concatenating three `command` scripts produces something that is
#: not a command: run it and it brings the arm up, measures it, then brings it up
#: and measures it again. Worse, it inherits every syntax fault in every half —
#: measured, two of the three sealed stage-5 halves do not parse (an apostrophe
#: inside a `${VAR:?word}` message opens a string that runs to end of file), so
#: the merged product did not parse either. `items/command` is now written as one
#: real script for the merged arm and the halves are preserved verbatim beside
#: the logs.
MERGED_TEXT = ("README.md", "items/watchout")

#: Where each source's original `command` goes, since it is no longer merged into
#: one. Kept verbatim: it is the record of how that half was actually invoked,
#: and rewriting somebody else's script to make it parse would be inventing
#: history rather than recording it.
COMMAND_ARCHIVE = "items/logs/command"


def copy_tree(src: Path, dest: Path, skip: set[str], clashes: list[str]) -> int:
    copied = 0
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        rel = path.relative_to(src).as_posix()
        if rel in skip:
            continue
        target = dest / rel
        if target.exists():
            if target.read_bytes() == path.read_bytes():
                continue
            clashes.append(f"{rel} exists in two sources with different bytes")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        shutil.copymode(path, target)
        copied += 1
    return copied


def write_command(out: Path, sources: list[Path], arm: str) -> None:
    """One runnable `items/command` for the merged arm, and the halves archived.

    **Not a concatenation, and that was a real defect rather than a style
    choice.** Three `command` scripts joined end to end describe bringing the arm
    up, measuring it, bringing it up again and measuring it again — which is not
    what happened and not something anybody should run. And it inherits every
    syntax fault in every half: measured, two of the three sealed stage-5 halves
    do not parse, so the merged product did not either, and the package's
    `command`-parses validator would refuse both arms.

    So this writes the invocation that reproduces *this* arm, in the order the
    STEPS readme ran it, and keeps each source's original beside the logs. The
    originals are kept **verbatim** — rewriting somebody else's script to make it
    parse would be inventing history rather than recording it.

    Every site path arrives as a shell variable. That is what makes the script
    portable, and it is checked: a `${VAR:?...}` message here must contain no
    apostrophe, because bash parses that message with quoting active and a lone
    `'` opens a string that runs to end of file — the same fault that broke the
    halves.
    """
    archive = out / COMMAND_ARCHIVE
    archive.mkdir(parents=True, exist_ok=True)
    kept = []
    for src in sources:
        path = src / "items" / "command"
        if path.is_file():
            (archive / f"{src.name}.command.sh").write_text(
                path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            kept.append(src.name)

    target = out / "items" / "command"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f"""#!/usr/bin/env bash
# Reproduce the {arm} arm of the integration comparison, in the order it ran.
#
# One script and not three. The bring-up and the measurement were one task
# (M5.2 -- bring-up and use may not be split across agents), so one arm has one
# invocation. Each producing script's own original is kept verbatim under
# {COMMAND_ARCHIVE}/ ({', '.join(kept) or 'none were carried'}).
#
# Every site path is a variable, so this runs somewhere other than the machine
# that produced it. PKG is the task package; the rest are the allocation.
set -eu
: "${{PKG:?set PKG to the e2e-flow package directory}}"
: "${{E2E_NODE:?set E2E_NODE to the node holding the allocation}}"
: "${{E2E_NODE_IP:?set E2E_NODE_IP to that node IP}}"
: "${{E2E_JOBID:?set E2E_JOBID to the allocation id}}"
: "${{E2E_MODEL_PATH:?set E2E_MODEL_PATH to the checkpoint directory}}"
: "${{E2E_IMAGE:?set E2E_IMAGE to the engine image}}"
: "${{E2E_WORK_ROOT:?set E2E_WORK_ROOT to a node-local scratch directory}}"
: "${{E2E_CONTAINER:?set E2E_CONTAINER to a container name you own}}"
: "${{OUT:?set OUT to a directory to write this arm into}}"

# 1. bring the arm up. The {arm} arm mounts {'the overlay' if arm == 'patched' else 'nothing'}.
E2E_ARM={arm} \\
E2E_OUTPUT_DIR="$OUT/deployment" \\
  bash "$PKG/assets/serve/round.sh"

# 2. measure it: smoke, needle, probe, lm_eval, then the replay rounds, in that
#    order and not overlapping. The order is part of the measurement -- "round 1
#    is cold against this trace" is only true if the same things preceded it on
#    both arms.
E2E_ARM={arm} \\
E2E_OUTPUT_ACCEPT="$OUT/accept" \\
E2E_OUTPUT_BENCH="$OUT/bench" \\
  bash "$PKG/assets/accept/measure.sh"

# 3. compose this handoff. The bring-up window has no default on purpose: a
#    guessed one makes check_measurement_order pass by construction.
python3 "$PKG/assets/lib/merge_arm.py" --arm {arm} \\
  --from "$OUT/deployment" --from "$OUT/accept" --from "$OUT/bench" \\
  --out "$OUT/handoff" \\
  --serve-started "${{SERVE_STARTED:?ISO 8601 time the bring-up began}}" \\
  --serve-seconds "${{SERVE_SECONDS:?how long the bring-up took}}" \\
  --package "$PKG" --environment "${{ENVIRONMENT_YAML:?m1 environment.yaml}}"
""",
        encoding="utf-8",
    )
    # `agent/gate.py` requires `script` / `command` / `entry` to carry the
    # executable bit.
    target.chmod(0o755)


def merged_steps(sources: list[Path], arm: str, started: str, seconds: float) -> dict:
    """The bring-up, then whatever the measurement recorded, in one list."""
    found: list[dict] = []
    for src in sources:
        path = src / "items" / "env" / "steps.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        declared = payload.get("arm")
        if declared and declared != arm:
            raise SystemExit(f"merge_arm: {path} records arm={declared!r} and this is the {arm} arm")
        for step in payload.get("steps") or []:
            if step not in found:
                found.append(step)
    if not found:
        raise SystemExit(
            "merge_arm: no source carried items/env/steps.json. Without it "
            "check_measurement_order has nothing to read, and the ordering guarantee that "
            "used to be a graph edge would be an assertion in a readme and nothing more."
        )
    serve = {"step": "serve", "rc": "0", "seconds": float(seconds), "started": started}
    return {"arm": arm, "steps": [serve] + found}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--from", dest="sources", action="append", required=True, metavar="DIR",
                    help="a directory a bring-up or measurement script wrote, in the order they ran")
    ap.add_argument("--out", required=True, help="the handoff's content directory")
    ap.add_argument("--serve-started", required=True, metavar="ISO8601",
                    help="when this arm's bring-up began; no default, see the module docstring")
    ap.add_argument("--serve-seconds", required=True, type=float)
    ap.add_argument("--package", required=True)
    ap.add_argument("--environment", required=True,
                    help="m1's environment.yaml, carried forward rather than re-derived (G5)")
    a = ap.parse_args(argv)

    out = Path(a.out)
    sources = [Path(s) for s in a.sources]
    for src in sources:
        if not src.is_dir():
            raise SystemExit(f"merge_arm: {src} is not a directory")

    (out / "items").mkdir(parents=True, exist_ok=True)
    clashes: list[str] = []
    # `items/command` joins the skip set even though it is no longer merged:
    # without it `copy_tree` lays the FIRST source's command down verbatim, which
    # is one third of the arm and, on two of the three sealed halves, does not
    # parse. `write_command` replaces it with one script for the whole arm.
    skip = set(MERGED_TEXT) | {"items/command"}
    total = sum(copy_tree(src, out, skip, clashes) for src in sources)
    if clashes:
        for line in clashes:
            print(f"merge_arm: {line}", file=sys.stderr)
        raise SystemExit(
            "merge_arm: two sources disagree about a file. Overwriting one with the other "
            "would lose evidence silently, which is the failure this refusal exists for."
        )

    for rel in MERGED_TEXT:
        parts = []
        for src in sources:
            path = src / rel
            if path.is_file():
                parts.append(f"# ---- from {src.name} ----\n" + path.read_text(encoding="utf-8"))
        if not parts:
            continue
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(parts), encoding="utf-8")

    write_command(out, sources, a.arm)

    steps = merged_steps(sources, a.arm, a.serve_started, a.serve_seconds)
    (out / "items" / "env").mkdir(parents=True, exist_ok=True)
    (out / "items" / "env" / "steps.json").write_text(json.dumps(steps, indent=2), encoding="utf-8")

    names = " -> ".join(str(s.get("step")) for s in steps["steps"])
    (out / "README.md").write_text(
        f"""# {a.arm}.measurement

## Purpose

Everything measured on the **{a.arm}** arm of the integration comparison, in one
record: how the deployment was brought up, what it proved about correctness, and
how it performed under the replay trace.

One handoff and not three. The bring-up and the measurement may not be split
across agents (M5.2), so one task produces both — and the evidence they produce
is one arm's evidence.

## How to run

`items/command` is the concatenation of each producing script's own invocation,
in the order they ran. Site paths are written as `@NAME@`; `@WORK_ROOT@` was the
node-local work area on the machine that produced this.

## Result

`items/result/` holds the correctness evidence — `smoke.json`, `needle.json`,
`probe.json`, `lm_eval/` and, when the run generated them, `adhoc.json` — beside
one `r<N>/` directory per replay round.

## Environment

`items/env/environment.yaml` is the flow's environment record, carried forward
from m1 unchanged. `items/env/steps.json` is this arm's step order with
timestamps:

    {names}

That record is what `check_measurement_order` reads. It exists because the
ordering it describes used to be a graph edge and is now a numbered list in a
readme, which is a weaker guarantee — so the guarantee moved from the scheduler
to the evidence.

## Watch out

See `items/watchout`. The one that matters most on this arm: a deployment being
healthy is not the same as it being *comparable* to the other arm's. The design
controls for session, node, trace, order and image and not for node load at
measurement time, which on a shared node is the dominant term.
""",
        encoding="utf-8",
    )

    source_env = Path(a.environment)
    subprocess_argv = [
        sys.executable,
        str(Path(a.package) / "assets" / "lib" / "env_render.py"),
        "--inherit", str(source_env),
        "--content-type", "reproducible",
        "--out", str(out),
    ]
    import subprocess

    subprocess.run(subprocess_argv, check=True)

    print(f"merge_arm: {a.arm}: {total} file(s) from {len(sources)} source(s); steps: {names}")
    print(f"merge_arm: composed at {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
