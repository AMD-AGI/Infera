#!/usr/bin/env python3
"""Compose one arm's `{stock,patched}.measurement` from the pieces that produced it.

**Six evidence kinds became two** (CONTRACT.md §7, a consequence of M5.2). In
`integration-demo` one arm was three handoffs — `deployment_<arm>`,
`acceptance_<arm>`, `bench_<arm>` — produced by two tasks. Here one task produces
one handoff per arm, and this module is the join: it takes the directories the
bring-up and the measurement scripts wrote and lays them out as a single
`reproducible` content tree.

**The union is mechanical, and exactly two things are not.**

*The collision rule.* Three files exist in more than one source — `README.md`,
`items/command` and `items/watchout` — because each script wrote a complete
handoff of its own. They are concatenated with a header naming where each half
came from, rather than one overwriting the other, which is what a plain
`cp -a` of one directory over another would have done silently. Everything else
is copied, and a genuine collision (same path, different bytes) is a failure
rather than a last-writer-wins.

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
MERGED_TEXT = ("README.md", "items/command", "items/watchout")


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
    total = sum(copy_tree(src, out, set(MERGED_TEXT), clashes) for src in sources)
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
        if rel == "items/command":
            # `agent/gate.py` requires `script`/`command`/`entry` to carry the
            # executable bit, and a concatenation of two scripts needs its own
            # shebang or it is a text file with commands in it.
            target.write_text("#!/usr/bin/env bash\nset -eu\n" + target.read_text(encoding="utf-8"),
                              encoding="utf-8")
            target.chmod(0o755)

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
