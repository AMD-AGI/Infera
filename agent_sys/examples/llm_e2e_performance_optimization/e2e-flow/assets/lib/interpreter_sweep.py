#!/usr/bin/env python3
"""Drive every body in the package under the shell and interpreter a run creates.

Not part of the deliverable. Asked for by the leader after three
interpreter-shaped bugs, each invisible until something ran:

  1. bodies are `#!/usr/bin/env bash` + `set -o pipefail`, but agent_sys invokes
     one as `["/bin/sh", entry]` and `/bin/sh` here is dash, which exits 2 on
     line 1 (CONTRACT §3.2a);
  2. a validator body's zone gets a policy-derived PATH on which `python3` is
     `/usr/bin/python3`, which cannot import `assets/lib/schema.py`'s deps;
  3. the same again in a *task* body, which never sees `AGENT_SYS_DEMO_PYTHON`.

Three layers, cheapest first, because each catches a different thing:

  A. `sh -n` on every body            — can the shell even parse it
  B. `import` every python module     — can the interpreter load it
     under /usr/bin/python3              (catches (2) and (3) at the seam
                                          regardless of code path)
  C. run every body for real          — what actually happens, and for a
     under PATH=/usr/bin:/bin            validator, does it leave a verdict

Layer C's verdict column is the one the leader cares about. A validator that
refuses is healthy; a validator that **dies without writing `verdict.json`** is
read by the phase as a broken validator rather than as a refused handoff, so the
graph reports the wrong thing about the wrong artefact.

**It does not judge PASS/FAIL.** Every body here is given deliberately thin
inputs, so refusals are expected and uninteresting. The question is only whether
the body got far enough to have an opinion.

C2 supplies inputs — 2026-09-04
------------------------------
C2 used to hand a task body its **output** slots and nothing else. Every
non-zero row there was therefore ambiguous between "correctly refuses without an
input" and "broken", which is CONTRACT §4.4 in the harness's own instrument: the
graph never dispatches a task without staging its declared inputs, and
`run_profiling_mode_off.task`'s *mock* branch reads
`$AGENT_SYS_INPUT_DEPLOY_KIT/items/codes/environment.yaml` on its second line —
so the old C2 could not see past it.

`sweep_inputs.py` now resolves one real content tree per kind and this module
stages a fresh copy of each declared input into the task's own zone, at the
shape `grants.input_env` hands a body. `--no-inputs` restores the old behaviour,
which is how the change is shown to make a difference rather than asserted to:
**prove the probe can fail before believing that it passed.**

C1 grades real artefacts too — 2026-09-04
-----------------------------------------
**A gate whose fixtures are simpler than reality certifies the fixtures.**

C1 handed every validator one stub of empty `items/` directories. Against that,
`check_workset_shape` returned rc=0 *with* a verdict; against the real
`operator_workset` rung 0 produced, the same body **died writing no verdict at
all** — `ModuleNotFoundError: No module named 'referencing'`, the category this
sweep exists to catch, missed by the sweep. A validator that never reaches its
schema loader cannot fail in a way a stub can show.

So C1 now feeds each validator one real artefact per kind it declares in its own
`inputs:`, out of the same pool C2 stages task inputs from.

**Shown to fire, not assumed to.** Same artefact, same interpreter, two versions
of that body:

    check_workset_shape @ 4b4c9ce^   rc=1  verdict=False  ModuleNotFoundError: referencing
    check_workset_shape @ HEAD       rc=0  verdict=True   1/1 passed

The gate catches the class, and it passes today because m3 fixed the cause. Both
halves were needed: a gate that has only ever passed is not evidence.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_inputs  # noqa: E402  — sits beside this file

#: **Derived, not written down.** This file now lives inside the package it
#: sweeps (`assets/lib/`), so the package root is two directories up. A hard
#: path here was correct exactly once — in the worktree it was written in — and
#: would have made the gate silently sweep the wrong tree for anyone else.
PKG = Path(__file__).resolve().parents[2]
MOCK = Path("/shared_nfs/yihou/agent_sys/cheat_for_mock")
#: **Local disk, not `/shared_nfs`.** Measured 2026-09-04: the export is mounted
#: `ro` on this login node (`mount | grep shared_nfs`), so the scratch tree this
#: harness has always used cannot be written. Reads are unaffected and every
#: input still comes off `/shared_nfs`. Move this back when the mount returns.
#: **Scratch, never beside the script.** `assets/lib/` is the tree every body is
#: staged from, so a sweep writing zones and staged inputs here would grow every
#: zone in every run. Local disk rather than `/shared_nfs`, which is mounted
#: `ro` on the login node. Override with `E2E_SWEEP_SCRATCH`.
SCRATCH = Path(os.environ.get("E2E_SWEEP_SCRATCH",
                              str(Path.home() / "ws_handoff_refine_m2" / "sweep")))

#: The PATH a body actually gets. `/usr/bin/python3` is jsonschema 4.10.3 with
#: no `referencing`; the login shell's python is miniconda's, which has both —
#: which is exactly why nobody saw (2) or (3) in testing.
RUN_PATH = "/usr/bin:/bin"

#: Which module owns which body, from CONTRACT §8a. Only used to address the
#: report; nothing here edits another owner's file.
OWNER = [
    (re.compile(r"deploy|packup_shape|deploy_kit|deploy_serves"), "m1"),
    (re.compile(r"bench_result|trace_coverage|kernel_table|profiling_evidence|"
                r"profiling_mode|merge_profiling"), "m2"),
    (re.compile(r"rank|identify|worklist|workset|build_workset|operator"), "m3"),
    (re.compile(r"optimize_kernel|optimization_shape"), "m4"),
    (re.compile(r"patch|measurement|regression|speedup|acceptance|integrate|"
                r"bench_report|command_parses|environment"), "m5/common"),
]


def owner_of(name: str) -> str:
    for pattern, who in OWNER:
        if pattern.search(name):
            return who
    return "?"


#: `${x:-d}` -> `d`, and the result stays a **string** — which is what
#: `spec_loader.variables.substitute` does. Measured: a substituted numeric arg
#: reaches `args.json` as `"50"`, while a literal `80.0` stays a float. So a
#: validator that compares `count < args["min_requests"]` without `int()` raises
#: `TypeError` only for the substituted half, which is why it survives review.
_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
#: For `${x}` with no default. Values chosen to be plausible rather than right:
#: this sweep asks whether a body *runs*, not whether it passes.
_NO_DEFAULT = {"tp": "8", "jobid": "1", "node": "n", "node_ip": "127.0.0.1",
               "model_name": "m/n", "model_path": "/p", "image": "i:t"}


def _subst(v):
    if isinstance(v, str):
        return _REF.sub(lambda m: _NO_DEFAULT.get(m.group(1), m.group(2) if m.group(2) is not None else "1"), v)
    if isinstance(v, list):
        return [_subst(x) for x in v]
    if isinstance(v, dict):
        return {k: _subst(x) for k, x in v.items()}
    return v


_ARGS: dict | None = None
_VKINDS: dict[str, list[str]] = {}


def args_of(validator: str) -> dict:
    """Every validator's declared `args`, substituted the way a run substitutes.

    **Not `{}`.** The first version of this sweep handed every body empty args,
    so every `args.get(k, default)` returned the *typed* default and a missing
    coercion was invisible — the same structural blindness layer B had against
    `schema.validate()`. CONTRACT §4.2's arithmetic half only bites on the
    substituted, string-typed form, so a gate that does not deliver strings
    cannot gate it.
    """
    global _ARGS
    if _ARGS is None:
        import yaml
        _ARGS = {}
        for f in sorted((PKG / "steps").glob("*.yaml")):
            for doc in yaml.safe_load_all(f.read_text()):
                for e in (doc if isinstance(doc, list) else [doc]):
                    if isinstance(e, dict) and e.get("module") == "validator":
                        _ARGS[e["name"]] = _subst(e.get("args") or {})
                        _VKINDS[e["name"]] = list(e.get("inputs") or [])
    return _ARGS.get(validator, {})


def kinds_of(validator: str) -> list[str]:
    """The kinds a validator declares grading, from its own `inputs:`.

    Read rather than guessed, and it is what makes C1's real-artefact fixture
    possible at all: `check_measurement_order` declares **both** arms, and its
    thin-fixture complaint was literally *"this phase staged no arm(s) and both
    are needed"* — a validator refusing correctly about the fixture.
    """
    args_of("")          # populate
    return _VKINDS.get(validator, [])


_AGENT_ENV: dict | None = None


def agent_env() -> dict[str, str]:
    """`shared.yaml`'s `runner` env block, substituted the way a run substitutes.

    The third slot of the same §4.4 hole as the inputs, and it is what
    `deploy_and_prove`'s row was actually about: `mock_adapt.sh:70` branches on
    `$E2E_IMAGE`, which the package declares here and nowhere else —
    `agent.schema.json` makes an agent's `env` block the only route by which a
    package may name a variable. Withheld, m1's body refused on an unset image
    and the row read as m1's.

    **`runner`'s, for every task, and that is the mocked configuration rather
    than a simplification.** Four closures name `${m<N>_agent:-<an AI agent>}`
    and a mock run passes `--var m1_agent=runner …` (MOCK-MAP), so under the
    mode this layer runs in, all eleven bodies run under `runner`.
    """
    global _AGENT_ENV
    if _AGENT_ENV is None:
        import yaml
        _AGENT_ENV = {}
        for doc in yaml.safe_load_all((PKG / "shared.yaml").read_text()):
            for e in (doc if isinstance(doc, list) else [doc]):
                if isinstance(e, dict) and e.get("module") == "agent" and e.get("name") == "runner":
                    _AGENT_ENV = {k: str(_subst(v)) for k, v in (e.get("env") or {}).items()}
    return _AGENT_ENV


def sh_parses(path: Path) -> tuple[bool, str]:
    """Layer A. The shell named by the shebang, and `/bin/sh` regardless.

    Both, because agent_sys ignores the shebang and runs `/bin/sh`, while a
    human reproducing by hand runs the shebang's shell. A body has to survive
    the one that actually starts it; a mismatch between the two is worth seeing.
    """
    first = path.read_text(errors="replace").splitlines()[:1]
    declared = "bash" if first and "bash" in first[0] else "sh"
    out = []
    for shell in dict.fromkeys(("/bin/sh", f"/bin/{declared}")):
        r = subprocess.run([shell, "-n", str(path)], capture_output=True, text=True)
        if r.returncode:
            out.append(f"{shell}: {r.stderr.strip().splitlines()[0] if r.stderr.strip() else 'rc=%d' % r.returncode}")
    return (not out), "; ".join(out)


def imports(path: Path) -> tuple[bool, str]:
    """Layer B. Import the module under /usr/bin/python3, deps and all.

    Imported rather than executed: a module whose import raises cannot be run at
    all, and import is where `ModuleNotFoundError` for `referencing`, `yaml` or
    `jsonschema` lands. `sys.path` gets the module's own directory and
    `assets/lib`, which is what every body does for itself.
    """
    code = (
        "import sys, importlib.util as u\n"
        f"sys.path.insert(0, {str(PKG / 'assets' / 'lib')!r})\n"
        f"sys.path.insert(0, {str(path.parent)!r})\n"
        f"spec = u.spec_from_file_location('probe', {str(path)!r})\n"
        "m = u.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
    )
    r = subprocess.run(["/usr/bin/python3", "-c", code], capture_output=True, text=True,
                       env={"PATH": RUN_PATH, "AGENT_SYS_TASK_PACKAGE": str(PKG),
                            "HOME": os.environ.get("HOME", "/tmp")})
    if r.returncode == 0:
        return True, ""
    tail = [l for l in r.stderr.strip().splitlines() if l.strip()]
    return False, tail[-1] if tail else f"rc={r.returncode}"


def run_validator(name: str, entry: Path, materials: dict[str, Path], args: dict) -> dict:
    """Layer C for a validator: a real zone, and did it leave a verdict.

    **`materials` is one real artefact per declared kind**, not one thin
    directory. Until 2026-09-04 this handed every validator the same stub of
    empty `items/` subdirectories, and the fixture was thinner than the
    artefacts these bodies meet — so `check_workset_shape` returned rc=0 *with*
    a verdict here and, against the real `operator_workset` rung 0 produced,
    **died writing no verdict at all** on `ModuleNotFoundError: referencing`.
    The gate certified the fixture. A validator that never reaches its schema
    loader cannot fail in a way a stub can show.
    """
    with tempfile.TemporaryDirectory(dir=SCRATCH) as zone:
        z = Path(zone)
        (z / "args.json").write_text(json.dumps(args))
        (z / "inputs.json").write_text(json.dumps(list(materials)))
        (z / "materials.json").write_text(json.dumps({h: str(p) for h, p in materials.items()}))
        env = {
            "PATH": RUN_PATH,
            "HOME": os.environ.get("HOME", "/tmp"),
            # The PRODUCER row. A validator's *input* phase gets the GLOBAL row
            # and only `AGENT_SYS_DEMO_PACKAGE`; both are set so a body reading
            # either resolves, which is the friendlier of the two cases.
            "AGENT_SYS_TASK_PACKAGE": str(PKG),
            "AGENT_SYS_DEMO_PACKAGE": str(PKG),
            # Exported for a validation zone (`cli/main.py:668`), so a validator
            # legitimately has it. Pointed at the system interpreter on purpose:
            # this sweep asks what happens on the interpreter without the deps.
            "AGENT_SYS_DEMO_PYTHON": "/usr/bin/python3",
        }
        r = subprocess.run(["/bin/sh", str(entry)], cwd=z, env=env,
                           capture_output=True, text=True, timeout=300)
        verdict = (z / "verdict.json").is_file()
        return {"rc": r.returncode, "verdict": verdict,
                "err": _first_error(r.stderr or r.stdout)}


def run_task(name: str, entry: Path, outputs: dict[str, Path],
             inputs: dict[str, Path]) -> dict:
    """Layer C for a task body, in mock mode — the path a mock run takes.

    **No `AGENT_SYS_DEMO_PYTHON`.** `cli/main.py` puts it in `validation_env`
    only, so a task body never sees it; that is the whole of failure (3) and
    withholding it here is the point of the test.

    **`AGENT_SYS_INPUT_<KIND>`, though, yes.** Withholding those was not a test
    of anything — see the module docstring.
    """
    env = {
        "PATH": RUN_PATH,
        "HOME": os.environ.get("HOME", "/tmp"),
        "AGENT_SYS_TASK_PACKAGE": str(PKG),
    }
    # The package's own declared variables first, so a slot path can never be
    # shadowed by one of them.
    env.update(agent_env())
    env.update({
        "E2E_MOCK_ROOT": str(MOCK),
        "E2E_MOCK_STAGES": "all",
    })
    env.update({k: str(v) for k, v in outputs.items()})
    env.update({k: str(v) for k, v in inputs.items()})
    r = subprocess.run(["/bin/sh", str(entry)], cwd=str(SCRATCH), env=env,
                       capture_output=True, text=True, timeout=600)
    return {"rc": r.returncode, "verdict": None, "err": _first_error(r.stderr or r.stdout)}


def _stage_inputs(task: str, zone: Path, pool: dict) -> tuple[dict[str, Path], list[str]]:
    """A fresh staged copy of each declared input, at production's path shape.

    `grants.input_env` hands a body `<zone>/handoffs/<hid>/v<N>` holding the
    artefact's own files, so that is the shape built here; the pool's trees are
    already normalised to it by `sweep_inputs.normalise`.

    **Copied per task rather than shared.** A body that writes into an input it
    was handed would otherwise change what the next body in this sweep reads,
    and the second reader's row would then be about the first reader.

    **The `<hid>` level is a uuid, not the kind's name**, and that is not
    cosmetic. A readable name there was the first version and it cost a wrong
    reading immediately: `optimize_kernel` derives `workset_ref.version` from
    that component, so the schema complaint came back naming `OPERATOR_WORKSET`
    and read like the harness's own spelling. `uuid5` off the task and kind
    keeps it deterministic — the same run stages the same paths twice — while
    making the component the same *shape* the runner mints, so a body that
    misreads it says so in a form nobody can mistake for the fixture.
    """
    import uuid
    got: dict[str, Path] = {}
    unmet: list[str] = []
    for kind in sweep_inputs.declared_inputs(PKG).get(task, []):
        src = pool.get(kind)
        if src is None:
            unmet.append(kind)
            continue
        hid = uuid.uuid5(uuid.NAMESPACE_URL, f"sweep/{task}/{kind}")
        dst = zone / "handoffs" / str(hid) / "v0"
        shutil.copytree(src[0], dst, dirs_exist_ok=True, symlinks=True)
        got["AGENT_SYS_INPUT_" + sweep_inputs.env_name(kind)] = dst
    return got, unmet


NOISE = re.compile(r"^(mock:|m2_reshape:|wrote |Traceback|\s+File |\s+\w+ = |\s*\^)")


def _first_error(text: str) -> str:
    lines = [l.rstrip() for l in (text or "").strip().splitlines() if l.strip()]
    if not lines:
        return ""
    for l in reversed(lines):
        if not NOISE.match(l):
            return l[:160]
    return lines[-1][:160]


def main() -> int:
    with_inputs = "--no-inputs" not in sys.argv
    unmet_by_task: dict[str, list[str]] = {}
    SCRATCH.mkdir(parents=True, exist_ok=True)
    validators = sorted(PKG.glob("assets/*.validator/entry.sh"))
    tasks = sorted(PKG.glob("assets/*.task/entry.sh"))
    pymods = sorted(PKG.glob("assets/lib/*.py")) + sorted(PKG.glob("assets/*.validator/*.py"))

    print(f"package: {PKG}")
    print(f"PATH under test: {RUN_PATH}   python3 -> "
          f"{subprocess.run(['sh','-c','command -v python3'],env={'PATH':RUN_PATH},capture_output=True,text=True).stdout.strip()}")

    # ---- Layer A -----------------------------------------------------------
    print(f"\n{'='*100}\nA. shell parse — every body, under /bin/sh and under its own shebang's shell\n{'='*100}")
    bad = 0
    for entry in validators + tasks:
        ok, why = sh_parses(entry)
        if not ok:
            bad += 1
            print(f"  FAIL  [{owner_of(entry.parent.name):9s}] {entry.parent.name:38s} {why}")
    print(f"  {len(validators)+len(tasks)} bodies, {bad} fail to parse")

    # ---- Layer B -----------------------------------------------------------
    print(f"\n{'='*100}\nB. python import under /usr/bin/python3 — every lib module and every check.py\n{'='*100}")
    bad_b = 0
    for mod in pymods:
        if mod.name == "__init__.py":
            continue
        ok, why = imports(mod)
        if not ok:
            bad_b += 1
            rel = mod.relative_to(PKG)
            print(f"  FAIL  [{owner_of(mod.parent.name):9s}] {str(rel):58s} {why}")
    print(f"  {len(pymods)} modules, {bad_b} fail to import")

    # ---- Layer C: validators ----------------------------------------------
    print(f"\n{'='*100}\nC1. validator bodies in a real zone — rc, and did it leave a verdict\n{'='*100}")
    # **Real artefacts, one per declared kind.** See `run_validator`. The pool is
    # the same one C2 stages task inputs from, so a kind is resolved once and
    # both layers grade the same bytes.
    kinds_all = sorted({k for v in validators
                        for k in kinds_of(v.parent.name.replace(".validator", ""))})
    vpool = {k: x for k, x in sweep_inputs.resolve(kinds_all, PKG, SCRATCH / "pool").items() if x}
    missing_kinds = [k for k in kinds_all if k not in vpool]
    print(f"  materials: {len(vpool)}/{len(kinds_all)} kinds resolved to a real artefact"
          + (f"; NO SOURCE for {', '.join(missing_kinds)}" if missing_kinds else ""))
    print(f"  {'owner':<10} {'validator':<32} {'mat':>5} {'rc':>4}  {'verdict':<8} first error")
    print(f"  {'-'*10} {'-'*32} {'-'*5} {'-'*4}  {'-'*8} {'-'*38}")
    novidict = []
    thinned = []
    for entry in validators:
        name = entry.parent.name.replace(".validator", "")
        want = kinds_of(name)
        mats = {f"h-{sweep_inputs.env_name(k)}": vpool[k][0] for k in want if k in vpool}
        if len(mats) < len(want):
            # A validator graded on fewer artefacts than it declares is exactly
            # the old fixture's failure mode, so the row says so rather than
            # letting a thin pass read as a pass.
            thinned.append((name, [k for k in want if k not in vpool]))
        try:
            got = run_validator(name, entry, mats, args_of(name))
        except subprocess.TimeoutExpired:
            got = {"rc": -1, "verdict": False, "err": "TIMEOUT after 300s"}
        mark = "yes" if got["verdict"] else "**NO**"
        if not got["verdict"]:
            novidict.append((owner_of(name), name, got["err"]))
        print(f"  {owner_of(name):<10} {name:<32} {len(mats)}/{len(want):<3} {got['rc']:>4}  "
              f"{mark:<8} {got['err']}")

    # ---- Layer C: task bodies ---------------------------------------------
    print(f"\n{'='*100}\nC2. task bodies in mock mode, WITHOUT AGENT_SYS_DEMO_PYTHON\n{'='*100}")
    pool: dict = {}
    if with_inputs:
        kinds = sorted({k for ks in sweep_inputs.declared_inputs(PKG).values() for k in ks})
        resolved = sweep_inputs.resolve(kinds, PKG, SCRATCH / "pool")
        pool = {k: v for k, v in resolved.items() if v is not None}
        print("  inputs, one real content tree per kind — source, and is it handoff-shaped")
        for k in kinds:
            v = resolved.get(k)
            bad = "" if v is None else (sweep_inputs.shape_problem(v[0]) or "")
            print(f"    {k:36s} {'** MISSING **' if v is None else v[1]:46s}"
                  f"{'  ** ' + bad + ' **' if bad else ''}")
    else:
        print("  --no-inputs: output slots only, the way this layer ran before 2026-09-04")
    if not str(SCRATCH).startswith("/shared_nfs"):
        # **Name the instrument's condition before reading any row as a verdict**
        # (§4.4). A body whose output slot has to be visible from the GPU node —
        # `build_workset` runs its two entrypoints in a container there — refuses
        # correctly when the slot is on local disk, and that refusal is about
        # this scratch directory, not about the body.
        print(f"\n  ** scratch is {SCRATCH}, not under /shared_nfs, because the export is")
        print("     mounted ro on this login node. A row refusing because the node cannot")
        print("     see its output slot is the instrument saying so, not the body. **")
    print()
    print(f"  {'owner':<10} {'task':<34} {'in':>5} {'rc':>4}  first error")
    print(f"  {'-'*10} {'-'*34} {'-'*5} {'-'*4}  {'-'*60}")
    for entry in tasks:
        name = entry.parent.name.replace(".task", "")
        out = SCRATCH / "out" / name
        shutil.rmtree(out, ignore_errors=True)
        slots = _slots_for(name, out)
        declared = sweep_inputs.declared_inputs(PKG).get(name, [])
        zone = SCRATCH / "zone" / name
        shutil.rmtree(zone, ignore_errors=True)
        staged, unmet = _stage_inputs(name, zone, pool) if with_inputs else ({}, list(declared))
        try:
            got = run_task(name, entry, slots, staged)
        except subprocess.TimeoutExpired:
            got = {"rc": -1, "err": "TIMEOUT after 600s"}
        # `2/3` reads as: two of the three declared inputs were supplied. Any row
        # short of its declared count is ambiguous again and says so in place,
        # rather than being counted as evidence.
        supplied = f"{len(declared) - len(unmet)}/{len(declared)}"
        if unmet:
            unmet_by_task[name] = unmet
        print(f"  {owner_of(name):<10} {name:<34} {supplied:>5} {got['rc']:>4}  {got['err']}")

    print(f"\n{'='*100}\nvalidators that died WITHOUT writing a verdict — the dangerous category\n{'='*100}")
    if novidict:
        for who, name, err in novidict:
            print(f"  [{who}] {name}: {err}")
    else:
        print("  none")

    print(f"\n{'='*100}\nC1 validators graded on fewer artefacts than they declare\n{'='*100}")
    if thinned:
        for name, miss in thinned:
            print(f"  [{owner_of(name)}] {name}: no source for {', '.join(miss)}")
        print("  A pass on a partial fixture is not a pass — these rows are the old failure mode.")
    else:
        print("  none — every validator was graded on a real artefact of every kind it declares")

    print(f"\n{'='*100}\nC2 rows that are still ambiguous — a declared input nothing could supply\n{'='*100}")
    if unmet_by_task:
        for name, kinds in sorted(unmet_by_task.items()):
            print(f"  [{owner_of(name)}] {name}: {', '.join(kinds)}")
    else:
        print("  none — every task body was handed every input the graph declares for it")
    return 0


#: Output slots per task, so a mock body has somewhere to write. Derived from
#: the kind names in the step files rather than guessed.
def _slots_for(task: str, out: Path) -> dict[str, Path]:
    kinds = _outputs_of(task)
    slots = {}
    for kind in kinds:
        var = "AGENT_SYS_OUTPUT_" + re.sub(r"[^0-9A-Za-z]", "_", kind).upper()
        d = out / kind
        d.mkdir(parents=True, exist_ok=True)
        slots[var] = d
    return slots


_OUTPUTS: dict[str, list[str]] | None = None


def _outputs_of(task: str) -> list[str]:
    global _OUTPUTS
    if _OUTPUTS is None:
        import yaml
        _OUTPUTS = {}
        for f in sorted((PKG / "steps").glob("*.yaml")):
            for doc in yaml.safe_load_all(f.read_text()):
                for e in (doc if isinstance(doc, list) else [doc]):
                    if isinstance(e, dict) and e.get("module") == "task":
                        _OUTPUTS[e["name"]] = list((e.get("task") or {}).get("outputs") or [])
    return _OUTPUTS.get(task, [])


if __name__ == "__main__":
    sys.exit(main())
