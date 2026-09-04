#!/usr/bin/env python3
"""Reading an `operator_workset`, once, for everyone who reads one.

Three parties read this artefact and they must not read it differently:
`build_workset` writes it, `check_workset_shape` grades its shape,
`check_workset_runs` re-measures against it, and m4 takes its ground truth from
it. Four hand-rolled readers of one layout is four chances for the workset's
"three shapes" to mean three different things.

The arithmetic lives here for the reason `analyze-demo` put `bench_stats` in
`assets/lib/`: the producer and the validator import the same
`weighted_mean`, so a stored figure that disagrees with the raw per-group
numbers **cannot** come from a different formula. It can only come from the
record having been edited after it was measured, which is exactly the finding
worth making.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

__all__ = [
    "ABSOLUTE_PATH_ALLOW_LIST",
    "PERFORMANCE_FLOOR",
    "PERFORMANCE_ROLES",
    "IMPL_CONTRACT",
    "CRASH_MARKER",
    "VALIDATOR_REPORT",
    "write_report",
    "absolute_paths_in",
    "arg_num",
    "assign_roles",
    "is_performance",
    "load_definition",
    "load_report",
    "load_workload",
    "load_workset",
    "rsd",
    "weighted_mean",
    "workset_root",
]

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: `handoff/locality.py`'s allow-list, duplicated.
#:
#: **Not because the seal enforces it — it does not.** `handoff/store.py:447`
#: and `:494` decline to call `locality.check`, user-ruled 2026-08-31 after the
#: shape heuristic read an HTTP access-log line as a filesystem path and refused
#: a correct artefact; measured 97% false positive on a real kit. The module is
#: kept intact and tested, so re-wiring it is one line, and `ROADMAP.md` §6.4
#: carries the rebuild. Anything in this package claiming "the seal refuses this"
#: is repeating a premise that stopped being true.
#:
#: What survives is the *portability* half, which was always the real point: a
#: script carrying one host's directory does not run on the next host. Callers
#: scope it to executable and generated content and leave the environment record
#: alone, whose absolute paths are its whole purpose.
ABSOLUTE_PATH_ALLOW_LIST = (
    "/usr/", "/bin/", "/sbin/", "/lib/", "/lib64/", "/etc/", "/opt/",
    "/proc/", "/sys/", "/dev/", "/var/lib/", "/var/log/", "/run/", "/srv/",
    "/workspace/", "/app/", "/tmp/",
)

_ABS = re.compile(r"(?<![A-Za-z0-9._~@+-])(?:[A-Za-z]:\\[^\s\"'<>|]*|(?:/[A-Za-z0-9._+@-]+){2,}/?)")
_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S*")

#: A path continuing from a variable expansion is **not** hard-coded, and this is
#: the largest false-positive class after URLs.
#:
#: `"$HERE"/scripts/forge_driver.py` leaves `/scripts/forge_driver.py` as a fresh
#: two-segment candidate, because the framework's lookbehind excludes
#: `[A-Za-z0-9._~@+-]` and a closing quote is in none of those. Measured: it
#: flagged three lines of a correct generated `run_forge.sh` on the first run of
#: the full chain.
#:
#: `analyze-demo` works around this by building script paths in two steps and
#: telling the reader not to "simplify" them back. That contortion existed to
#: satisfy the seal, and the seal does not run. The rule here is *no hard-coded
#: host path*, so stripping the expansion before scanning is not a loosening —
#: it is the rule stated correctly. A `$`-prefixed path is a parameterised one,
#: which is exactly what the check is asking for.
#: The trailing `["']?` is load-bearing and was found by running it: the shell
#: idiom is `"$HERE"/scripts/x.py`, so consuming `$HERE` alone leaves `"` as the
#: character before `/scripts` — and `"` is not in the lookbehind's exclusion
#: class either, so the candidate still fires. The closing quote has to go with
#: the expansion.
_EXPANSION = re.compile(r"""(?:\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*|@[A-Z_][A-Z0-9_]*@)["']?""")


def arg_num(args: dict, name: str, default, cast=float):
    """One numeric `args` entry, where an explicit **0 means 0**.

    `int(args.get("reverify_shapes") or 1)` reads 1 when the spec says 0, because
    `0 or 1` is 1. Measured here, not reasoned about: the guard that refuses
    `reverify_shapes: 0` — the setting that would turn `check_workset_runs` into
    a reader of the producer's own claim — was itself unreachable, so the one
    knob that can dismantle the package's trust chain could be set and would be
    silently ignored.

    A step yaml writes `'${workset_max_rsd:-0.10}'`, so the value arrives as a
    string and is cast rather than compared.
    """
    value = args.get(name)
    if value is None or value == "":
        return cast(default)
    return cast(value)


#: The two `role` values that mean "this shape gets timed".
PERFORMANCE_ROLES = frozenset({"performance", "correctness-and-performance"})

#: How many shapes must be **performance-measured**, not merely present.
#:
#: `minItems: 3` on `shapes` counts every shape, correctness-only ones included,
#: so a workset can satisfy it while timing exactly one — and `build_workset`'s
#: scaffold did precisely that, keying `role` on `is_primary`, of which there is
#: exactly one per operator by construction. m4 found it: STEP 1 of its packup
#: refuses with `1 performance shapes, the workset contract requires >= 3`, and
#: `check_speedup_substantiated` refuses again at the output boundary under
#: `min_shapes_measured: 3`. So the floor belongs on the *role* count as well as
#: on the shape count, and M3.7.4.1 is where both come from: 必须提供所有 test
#: case，包括所有需要优化的形状，保证在 ≥3 — 供步骤4使用. A shape nothing timed is
#: not a test case module 4 can use.
PERFORMANCE_FLOOR = 3


def is_performance(role: str) -> bool:
    """Whether a `shapes[].role` means the shape is timed."""
    return role in PERFORMANCE_ROLES


def assign_roles(shapes: list[dict], floor: int = PERFORMANCE_FLOOR) -> list[str]:
    """The `role` of every shape, in `shapes` order. One rule, every caller.

    Reads `is_primary` and `observed` off each entry and returns the role
    strings; it does not write them, so a caller building a dict literal and a
    caller patching one both get the same answer.

    The rule, in priority order:

    1. **The primary shape is always timed.** It is the modal shape in the
       capture and the one a headline number refers to.
    2. **Every observed shape is timed.** M3.7.4.1's 所有需要优化的形状 — a shape
       the service actually runs is a shape the optimisation is for, and one
       that is measured for correctness only cannot be shown to have got faster.
    3. **Synthetic shapes are promoted, in order, until `floor` is reached.**
       A shape constructed to cover a tile boundary is a correctness probe by
       intent, but the floor is on the count and not on the provenance: three is
       where a performance claim is about an operator rather than about one
       shape, and an operator with one observed shape still owes that claim.

    Everything not promoted is `correctness`. Timing costs wall-clock, so this
    promotes to the floor and no further — beyond it, rule 2 alone decides.
    """
    order = sorted(range(len(shapes)),
                   key=lambda i: (not shapes[i].get("is_primary"), not shapes[i].get("observed"), i))
    timed = {i for i in range(len(shapes))
             if shapes[i].get("is_primary") or shapes[i].get("observed")}
    for i in order:
        if len(timed) >= floor:
            break
        timed.add(i)
    return ["correctness-and-performance" if i in timed else "correctness"
            for i in range(len(shapes))]


#: Where a validator leaves its reasoning, beside `verdict.json`.
VALIDATOR_REPORT = "validator_report.txt"

#: The prefix a validator writes when it **could not run**, as opposed to when
#: it refused. `write_report` keys the section heading on it, so a crash reads
#: as a crash rather than as a judgement.
#:
#: A constant rather than a convention: five validators across three owners
#: already write this exact sentence, having copied it, and a heading that
#: silently stops matching when someone rewords their message is worse than no
#: heading. Spotted by m2 while adopting the helper — the heading said
#: `REFUSED` while the text below it said the artefact had not been graded,
#: which is the contradiction the crash/refusal split exists to remove.
CRASH_MARKER = "THIS VALIDATOR DID NOT RUN"

#: **What `--impl PATH` must be**, as data rather than prose.
#:
#: Undeclared until 2026-09-04, when m4's STEP 4 reached the harness and got
#: *"the Definition's `candidate:sampler_vocab_softmax` defines no `run`"*. Two
#: sides disagreed about what the flag means and nothing stated it — the same
#: shape as the flag *spellings*, which are data for exactly this reason and
#: whose lesson this extends: **if a consumer has to read the harness to use
#: the harness, the contract is the harness's behaviour and not a contract.**
#:
#: Declared by the producer so a consumer conforms to a statement rather than
#: to whatever the instrument currently does. A seed fitted to observed
#: behaviour pins the undeclared thing instead of removing it.
IMPL_CONTRACT = {
    "form": "python-source-file",
    "entry_symbol": "run",
    "call": "run(**inputs)",
    "inputs_from": "definition.inputs",
    "replaces": "baseline",
    "notes": (
        "exec'd in a fresh namespace, not imported: no module, no package, no __file__, "
        "so relative imports do not work and the implementation must be self-contained in "
        "the one file. Called with keyword arguments named by the Definition's `inputs`. "
        "Never replaces `reference` — a candidate that could would be grading itself. "
        "A stock module is made to fit by carrying the whole file verbatim and appending a "
        "`run` that delegates to it; `mock_adapt.py:_read_seed` is the worked example."
    ),
}


def write_report(validator: str, findings: dict[str, tuple[list[str], list[str]]]) -> None:
    """Every problem and note this validator produced, on disk in the zone.

    **A verdict without its reasons is a number nobody can act on.** Measured
    five times on 2026-09-04: a validator's stdout is kept nowhere
    (`temp/bugs/2026-09-03-a-validators-stdout-is-not-kept-anywhere.md`), so a
    zone holds `args.json`, `inputs.json`, `materials.json` and `verdict.json`
    and **not one word about why**. The last instance was `check_workset_runs`
    refusing a workset **correctly** — a real finding about a real
    re-measurement, and the reason was gone before anyone could read it.

    Written **always**, not only on failure. A passing re-measurement's note
    (*"recorded 0.0415 ms, re-measured 0.0413 ms, 0.4% apart"*) is the evidence
    that the gate did its job; keeping it only when it fails would make the
    record of a working trust chain the one thing never kept.

    `dict[str, bool]` is all `verdict.json` can carry (`zone.py:132`), so this
    is the only place a *reason* can live until the verdict type grows a third
    state. Named beside it deliberately: whoever finds one finds the other.

    ## What this guarantees, for the four owners who do not own this file

    **Any validator in this package may call it**; nothing here is m3-specific.
    Adopted so far by `check_deploy_kit` (m1) and `check_optimization_shape`
    and `check_speedup_substantiated` (m4), which is why it is written down
    rather than left as a convention three people happen to share.

    * **Needs nothing from the environment.** A validator declares no agent, so
      the package's `env:` block never reaches it and `os.environ` is close to
      empty — which is the reason this problem keeps recurring. This function
      reads no variable: the path is relative to the cwd the runner already
      places it in, beside `verdict.json`. Verified by running it under
      `env -i`, not by inspection.
    * **Written always**, pass or refuse. A passing run's notes are the
      evidence the gate did its job; keeping reasons only when they are bad
      makes the record of a working gate the one thing never kept.
    * **Call it *before* `zone.write_verdict`.** A crash in the verdict writer
      then cannot take the reasons with it. This ordering is the lesson from a
      teardown that ran after the thing it protected and therefore never ran.
    * **A crash is not a refusal.** Wrap `_check` and record
      `THIS VALIDATOR DID NOT RUN` as a problem, so a broken instrument does
      not read as a judgement about the artefact. `verdict.json` cannot express
      the difference (`todo.md` T29); this text is the only place it exists.

    **A validator that legitimately differs should say so here or in its own
    file** — the point is that a reader can tell a deliberate difference from
    drift. `check_deploy_serves` writing `probe_plan.json` and
    `probe_results.json` is such a case: those are *probe artefacts*, the
    inputs to its judgement, not the judgement's reasons, and they are not a
    substitute for this.
    """
    lines = [f"# {validator}"]
    for hid, (problems, notes) in findings.items():
        crashed = any(CRASH_MARKER in p for p in problems)
        verdict = "DID NOT RUN" if crashed else ("REFUSED" if problems else "passed")
        lines.append(f"\n## {hid}: {verdict}")
        lines += [f"  note:    {n}" for n in notes]
        lines += [f"  PROBLEM: {p}" for p in problems]
        if not problems and not notes:
            lines.append("  (no findings)")
    Path(VALIDATOR_REPORT).write_text("\n".join(lines) + "\n", encoding="utf-8")


def workset_root(content: Path) -> Path:
    """The workset root **is** `items/codes/`, with no wrapper directory.

    `zone.find_one_dir` is the wrong tool here and the difference is worth
    stating: it insists on exactly one directory under `items/codes/`, which is
    right for a `code` handoff carrying one packup and wrong for this one. A
    workset carries `definitions/`, `workloads/`, `operators/` and `evidence/`
    side by side — four directories, all required — so the merged kind puts the
    workset at the root and the *operators* in a list inside it.
    """
    return content / "items" / "codes"


def load_workset(content: Path) -> dict:
    """`items/codes/workset.yaml`. Raises `FileNotFoundError` or a parse error."""
    path = workset_root(content) / "workset.yaml"
    import yaml  # a declared agent_sys dependency

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_definition(content: Path, relative: str) -> dict:
    """One flashinfer-bench Definition JSON, by its `workset.yaml`-relative path."""
    return json.loads((workset_root(content) / relative).read_text(encoding="utf-8"))


def load_workload(content: Path, relative: str) -> list[dict]:
    """One flashinfer-bench Workload JSONL, as a list, blank lines dropped.

    Order is preserved and is load-bearing: `workset.operators[].shapes` is a
    denormalised index of this file, and `check_workset_shape` checks the two
    correspond line for line rather than as sets. A set comparison would accept
    an index that has silently been re-sorted, and the `--shape` selector then
    selects a different case than the reader expects.
    """
    text = (workset_root(content) / relative).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_report(content: Path, relative: str) -> dict:
    """One `evidence/*.json` report."""
    return json.loads((workset_root(content) / relative).read_text(encoding="utf-8"))


def weighted_mean(per_group_ms: list[float], iters_total: int) -> float:
    """The reduction both sides use. Groups are equal-sized by construction, so
    this is the arithmetic mean — written as a weighted mean anyway because the
    protocol allows unequal groups and a formula that silently assumes equality
    is one that breaks the day somebody uses the freedom."""
    if not per_group_ms:
        return 0.0
    per = iters_total / len(per_group_ms) if iters_total else 1.0
    return sum(m * per for m in per_group_ms) / (per * len(per_group_ms))


def rsd(per_group_ms: list[float]) -> float:
    """Relative standard deviation across groups, population form.

    The measured within-arm round-to-round spread on a steady node is ~2%. A
    noisy baseline is worse than no baseline: an optimiser working against it
    takes the first candidate that lands on a fast sample for a win and chases
    noise for hours.
    """
    n = len(per_group_ms)
    if n < 2:
        return 0.0
    mean = sum(per_group_ms) / n
    if mean <= 0:
        return 0.0
    variance = sum((m - mean) ** 2 for m in per_group_ms) / n
    return (variance ** 0.5) / mean


def absolute_paths_in(path: Path) -> list[str]:
    """Absolute paths in one file that the seal would refuse, as `name:line: hit`.

    Three exemptions, each because its absence produced a measured false
    positive:

    * a URL's path component — 401 of the 627 false positives the framework
      measured, matched on the scheme, which is what distinguishes it;
    * a path continuing from a variable expansion, `"$HERE"/scripts/x.py` — see
      `_EXPANSION`; a `$`-prefixed path is a parameterised one, which is what
      this check exists to ask for;
    * a shebang, which names an interpreter rather than a produced artefact.

    What remains uncaught, and is left uncaught on purpose: a **relative** path
    preceded by `>` or `}` still matches, because the framework's lookbehind
    excludes neither — a `<operator_id>/scripts/x.py` fragment in a prose README
    reads as absolute. Callers scope this to `.py`/`.sh`/`.json`/`.jsonl` and
    leave prose alone, which is where that shape lives.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    out: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("#!"):
            continue
        # Substituted with a word character, not a space: blanking it would
        # leave the following `/scripts/...` at the start of a token and the
        # lookbehind would not fire.
        for match in _ABS.finditer(_EXPANSION.sub("V", _URL.sub(" ", line))):
            hit = match.group(0)
            if not hit.startswith(ABSOLUTE_PATH_ALLOW_LIST):
                out.append(f"{path.name}:{lineno}: {hit}")
    return out
