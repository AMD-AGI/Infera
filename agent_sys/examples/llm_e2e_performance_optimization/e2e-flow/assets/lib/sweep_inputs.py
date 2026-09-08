#!/usr/bin/env python3
"""Resolve one real content tree per kind, so layer C2 can feed a task body.

Why this exists — CONTRACT §4.4. `interpreter_sweep.py`'s C2 handed each task
body its **output** slots and nothing else, so every non-zero row there was
ambiguous between "correctly refuses without an input" and "broken". A fixture
more convenient than production tests the fixture: the graph never dispatches a
task without staging its declared inputs, and `run_profiling_mode_off.task`'s
*mock* branch already reads `$AGENT_SYS_INPUT_DEPLOY_KIT/items/codes/
environment.yaml` — so a sweep that withholds inputs cannot see anything past
that line.

**The shape a body is handed.** `env_mgr/grants.py:396-431` — the value of
`AGENT_SYS_INPUT_<KIND>` is the *staged copy* at `<zone>/handoffs/<hid>/v<N>`,
and since `stage` narrowed it copies `v<N>/content` **to** `<into>/<hid>/v<N>`.
So a body finds `README.md` and `items/` directly at the end of the variable,
with no `content/` hop. Every path this module returns has that shape, and
`normalise()` is where the two on-disk layouts are folded into it.

**Provenance is printed, not assumed.** Three sources, in preference order:

  1. a **run tree** under `runroot/runs/` — the artefact this package's own
     bodies produced, already adapted (m1's `environment.yaml`, m2's reshaped
     `kernel_table`). Closest to what the graph stages;
  2. a **sealed handoff** under `cheat_for_mock/` — real bytes from the
     2026-09-02 cluster run, but predating this package's adaptations;
  3. the package's **own m5 mock producer**, `assets/lib/mock_m5.sh`, for the
     kinds MOCK-MAP (D)/(F) say have no one-to-one sealed source.

A kind that resolves to none of the three is reported `MISSING` rather than
stubbed. A stub would put the ambiguity back one level down.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

#: Every run root a run tree may be under, **newest root first**.
#:
#: Two, because the root moved on 2026-09-04: `/shared_nfs` is mounted `ro` on
#: the login node and `rw` on node 108891 — same volume, two mounts — so the
#: leader moved runs to `--demo-root /home/yihou/agent_sys_runroot`. The old
#: root stays in the list and is still *read*: it holds every run this package
#: has produced, including `20260903T172821-6a3c24`, which is where eight of
#: the fourteen kinds below come from. Dropping it would have silently demoted
#: those eight to their sealed sources and changed what the sweep measures
#: without changing a line of the sweep.
RUN_ROOTS = [
    Path("/home/yihou/agent_sys_runroot/runs"),
    Path("/shared_nfs/yihou/agent_sys/ws_handoff_refine/runroot/runs"),
]
MOCK = Path("/shared_nfs/yihou/agent_sys/cheat_for_mock")

#: MOCK-MAP.md's table, for kinds no run tree carries. `profiling_evidence` is
#: deliberately absent — (H) says the merge is real in every mode, so its four
#: parts are what gets mocked and it is never a copy.
SEALED = {
    "deploy_kit": "stage1-deploy/deploy_kit",
    "profiling_mode_off.bench_result": "stage2-profiling/aiperf_baseline",
    "profiling_mode_on.bench_result": "stage2-profiling/aiperf_profiled",
    "profiling_mode_on.profile_result": "stage2-profiling/torch_trace",
    "profiling_mode_on.kernel_table": "stage2-profiling/kernel_table",
    "kernel_worklist": "stage3-analyze/kernel_worklist",
    "operator_identity": "stage3-analyze/operator_identity",
    "operator_workset": "stage3-analyze/operator_workset",
    "kernel_optimization": "stage4-kernel-opt/kernel_optimization",
    "patch_overlay": "stage5-integration/patch_overlay",
    "integration_report": "stage5-integration/integration_report",
}

#: MOCK-MAP (D)/(F): `mock_m5.sh <what>` writes these into the output slots it
#: reads from `AGENT_SYS_OUTPUT_<KIND>`. Kind -> the subcommand that produces it.
BY_MOCK_M5 = {
    "stock.measurement": "arms",
    "patched.measurement": "arms",
    "e2e_packup": "packup",
}


#: Kinds whose **sealed** source is documented not to be the kind, so a run
#: tree is preferred even when the run never sealed it `valid`.
#:
#: `operator_workset` only, and MOCK-MAP (C) is the citation: the sealed
#: `stage3-analyze/operator_workset` is `items/code/` where the merged kind is
#: `items/codes/`, and behind that first mismatch are five more. Feeding it to
#: `optimize_kernel` or `apply_patch` produces a refusal on the fixture's
#: directory name — a §4.4 face-2 false failure, planted by the harness. The
#: run tree's `v1` carries the merged shape (`items/codes/workset.yaml`,
#: `environment.yaml`) because m3's `mock_adapt.py` had already run; it is
#: `generating` only because the run was cut short after it.
PREFER_RUN_TREE = {"operator_workset"}


def env_name(kind: str) -> str:
    """`env_mgr/grants.py:450` verbatim — uppercase, non-alphanumerics to `_`."""
    return "".join(c if c.isalnum() else "_" for c in kind).upper()


#: Nothing outside this prefix is ever removed. The scratch tree is the only
#: thing this module owns; the run trees and `cheat_for_mock` it reads are
#: other people's evidence and are opened read-only.
OWNED = Path(os.environ.get("E2E_SWEEP_SCRATCH",
                            str(Path.home() / "ws_handoff_refine_m2" / "sweep"))).parent


def _reset(d: Path) -> None:
    import shutil
    if not str(d.resolve()).startswith(str(OWNED)):
        raise SystemExit(f"sweep_inputs: refusing to clear {d}, which is outside {OWNED}")
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)


def _files(d: Path) -> int:
    return sum(len(fs) for _, _, fs in os.walk(d))


def normalise(d: Path) -> Path:
    """A version directory, in the shape `input_env` hands a body.

    Two layouts exist on disk and they differ by one hop: the store keeps
    `handoffs/<hid>/v<N>/content/...` while a staged copy in a zone is
    `handoffs/<hid>/v<N>/...`. `grants.input_env`'s docstring names exactly this
    and says the staged form is the one a body sees.
    """
    inner = d / "content"
    if inner.is_dir() and not (d / "README.md").exists():
        return inner
    return d


def shape_problem(d: Path) -> str | None:
    """Is this the shape a body finds at the end of `AGENT_SYS_INPUT_<KIND>`?

    **Prove the probe can fail before believing it passed** (§4.4). Provenance
    alone says where a tree came from, not whether it is an artefact: the
    partial `deploy_kit` that motivated the `valid` filter above was 38 real
    files and would have read as a healthy input on file count. Every handoff in
    this package carries `README.md` and `items/`, so a resolved input missing
    either is reported next to the row it feeds rather than left to surface as
    somebody's body being blamed for refusing.
    """
    if not d.is_dir():
        return "not a directory"
    missing = [n for n in ("README.md", "items") if not (d / n).exists()]
    return ("no " + ", ".join(missing)) if missing else None


def from_runs() -> dict[str, tuple[Path, str]]:
    """kind -> (best non-empty version tree, provenance), newest run first.

    **Largest wins within a run, not lowest-numbered.** Measured on
    `20260903T172821-6a3c24`: `deploy_kit`'s `v0/content` is empty while a `v1`
    the store record never lists carries the whole 237 KB kit. Taking `v0`
    because it is the version the record names would have handed every consumer
    an empty directory and called it a real input — face 2 of §4.4, built in.
    """
    out: dict[str, tuple[Path, str]] = {}
    # Roots in order, and runs within a root newest first. A run directory is
    # named for its start time, so ordering across roots by name would interleave
    # them; the roots are ordered by which one the flow launches into now.
    runs = [r for root in RUN_ROOTS if root.is_dir()
            for r in sorted(root.iterdir(), reverse=True)]
    for run in runs:
        store = run / "store" / "handoff"
        if not store.is_dir():
            continue
        for rec in sorted(store.glob("*.json")):
            try:
                d = json.loads(rec.read_text())
            except Exception:
                continue
            kind, hid = d.get("type"), d.get("id")
            if not kind or not hid or kind in out:
                continue
            # **Only a handoff the graph accepted.** Newest-run-wins alone took
            # `deploy_kit` from `20260903T174638-046322`, a run aborted while
            # that kit was still `generating`: 38 files, no
            # `items/codes/environment.yaml`, because MOCK-MAP (A)'s render had
            # not run yet. Every consumer would then have been handed a partial
            # kit and its refusal read as a body defect — §4.4 face 2, and the
            # harness would have built it in. The status is per version and the
            # populated tree is not always the version the record names (run
            # `…172821` lists v0 valid while v1 holds the bytes), so the filter
            # is at run granularity: did this kind reach `valid` in this run.
            sealed_valid = any(v.get("status") == "valid" for v in d.get("versions") or [])
            if not sealed_valid and kind not in PREFER_RUN_TREE:
                continue
            # Every `<hid>/v<N>` anywhere in this run: the store copy and every
            # staged copy in a zone. They are the same artefact; whichever is
            # populated is the one to take.
            best, best_n = None, 0
            for v in run.glob(f"**/{hid}/v*"):
                if not v.is_dir():
                    continue
                cand = normalise(v)
                n = _files(cand)
                if n > best_n:
                    best, best_n = cand, n
            if best is not None:
                note = "" if sealed_valid else ", NOT valid"
                # The root, not just the run: two roots are live since the move
                # and a run directory is named only for its start time, so
                # `run 20260903T172821-6a3c24` alone no longer says where from.
                root = "home" if str(run).startswith("/home/") else "nfs"
                out[kind] = (best, f"run[{root}] {run.name} ({best_n} files{note})")
    return out


def from_sealed(kind: str) -> tuple[Path, str] | None:
    rel = SEALED.get(kind)
    if not rel:
        return None
    d = MOCK / rel / "content"
    if d.is_dir() and _files(d):
        return d, f"sealed {rel}"
    return None


def from_mock_m5(kinds: list[str], pool: Path, pkg: Path, envyaml: Path) -> dict[str, tuple[Path, str]]:
    """Run the package's own m5 mock producer into `pool`.

    Not a stand-in: `mock_m5.sh` is the body a mocked `integrate_and_verify` and
    `packup` run, so what it writes here is byte-for-byte what the graph stages
    downstream of them. It refuses unless m5 is in `$E2E_MOCK_STAGES`, which is
    the gate that keeps it out of a real run.
    """
    got: dict[str, tuple[Path, str]] = {}
    wanted = sorted({BY_MOCK_M5[k] for k in kinds if k in BY_MOCK_M5})
    for what in wanted:
        env = dict(os.environ)
        env.update({
            "AGENT_SYS_TASK_PACKAGE": str(pkg),
            "E2E_MOCK_ROOT": str(MOCK),
            "E2E_MOCK_STAGES": "all",
        })
        produced = [k for k, w in BY_MOCK_M5.items() if w == what]
        for k in produced:
            d = pool / k
            # **An output slot the graph hands a body is empty.** Left populated
            # from a previous invocation, `merge_arm.py` sees its own last
            # output as a second source and refuses with "two sources disagree
            # about a file" — a refusal that is correct about the directory it
            # was given and says nothing about the mock. Measured here on the
            # second run of this module, which is exactly when nobody looks.
            _reset(d)
            env["AGENT_SYS_OUTPUT_" + env_name(k)] = str(d)
        r = subprocess.run(["bash", str(pkg / "assets" / "lib" / "mock_m5.sh"), what, str(envyaml)],
                           env=env, capture_output=True, text=True)
        for k in produced:
            d = pool / k
            if r.returncode == 0 and _files(d):
                got[k] = (d, f"mock_m5.sh {what}")
            else:
                tail = [l for l in (r.stderr or r.stdout or "").strip().splitlines() if l.strip()]
                sys.stderr.write(f"sweep_inputs: mock_m5.sh {what} did not produce {k}: "
                                 f"rc={r.returncode} {tail[-1][:140] if tail else ''}\n")
    return got


def resolve(kinds: list[str], pkg: Path, pool: Path) -> dict[str, tuple[Path, str] | None]:
    """kind -> (content tree, provenance) or None if no real source exists."""
    runs = from_runs()
    out: dict[str, tuple[Path, str] | None] = {}
    for k in kinds:
        out[k] = runs.get(k) or from_sealed(k)
    # `mock_m5.sh` needs an `environment.yaml` to carry forward, out of whatever
    # deploy_kit we just resolved. Without one its subcommands cannot run, and
    # saying so is better than producing an arm with no environment record.
    kit = out.get("deploy_kit")
    envyaml = (kit[0] / "items" / "codes" / "environment.yaml") if kit else None
    todo = [k for k in kinds if out.get(k) is None and k in BY_MOCK_M5]
    if todo:
        if envyaml and envyaml.is_file():
            out.update(from_mock_m5(todo, pool, pkg, envyaml))
        else:
            sys.stderr.write("sweep_inputs: no deploy_kit carrying items/codes/environment.yaml, "
                             f"so mock_m5.sh cannot run; {', '.join(todo)} stay MISSING\n")
    return out


def declared_inputs(pkg: Path) -> dict[str, list[str]]:
    """task name -> its declared input kinds, from the step files."""
    import yaml
    out: dict[str, list[str]] = {}
    for f in sorted((pkg / "steps").glob("*.yaml")):
        for doc in yaml.safe_load_all(f.read_text()):
            for e in (doc if isinstance(doc, list) else [doc]):
                if isinstance(e, dict) and e.get("module") == "task":
                    out[e["name"]] = list((e.get("task") or {}).get("inputs") or [])
    return out


def main() -> int:
    pkg = Path(__file__).resolve().parents[2]
    pool = Path(os.environ.get("E2E_SWEEP_SCRATCH",
                               str(Path.home() / "ws_handoff_refine_m2" / "sweep"))) / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    kinds = sorted({k for ks in declared_inputs(pkg).values() for k in ks})
    got = resolve(kinds, pkg, pool)
    for k in kinds:
        v = got[k]
        bad = "" if v is None else (shape_problem(v[0]) or "")
        print(f"  {k:36s} {'MISSING' if v is None else v[1]:44s} "
              f"{('** ' + bad + ' ** ') if bad else ''}{'' if v is None else v[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
