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
    "absolute_paths_in",
    "arg_num",
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
