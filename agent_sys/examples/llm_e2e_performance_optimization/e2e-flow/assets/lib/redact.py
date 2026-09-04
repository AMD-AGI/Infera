#!/usr/bin/env python3
"""Make evidence publishable: replace site-specific roots with placeholders.

`handoff` refuses to seal content that names an absolute path outside a small
allow-list, and it scans every text file rather than only the README:

    items/logs/worker.tail.log:88: '/apps/.../GLM-5.3-Flash-FP8' is a local path.
    A handoff names its dependencies and nothing about the machine that produced
    it (spec §7)

The rule is right — a record of one machine's afternoon is not a transferable
artefact. What is missing is the escape hatch its own source describes:
`locality.Oracles.image_prefixes` exists for "prefixes the declared container
image makes portable — the kind's `dependencies`", `handoff.schema.json` has the
matching `dependencies` field, and nothing connects the two. Reported in
`temp/bugs/002-handoff-dependencies-never-reach-locality-check.md`.

So this substitutes on the producing side, which is conda-build's
`PREFIX_PLACEHOLDER` design and one of the approaches `locality.py`'s docstring
cites as working. `${MODEL_MOUNT}/GLM-5.3-Flash-FP8` keeps the model's identity
and drops the mount root, which is the split spec §7 asks for.

**It fails rather than dropping what it cannot name.** After substituting, it
re-applies the same allow-list and the same shape regex the seal uses. Anything
left is reported with its file, line and text, so a new site-specific path shows
up here as a named error instead of as `output was never delivered` twenty
minutes later.

Usage:

    redact.py <dir> NAME=/absolute/prefix [NAME=/another ...]

Longer prefixes are substituted first, so a nested pair like
`/data/work` and `/data/work/profiles` cannot shadow each other by argument
order.
"""

import re
import sys
from pathlib import Path

#: `handoff.locality.ALLOWED_PREFIXES`, duplicated. A second reader of a fact
#: that module owns, admissible for the same reason `demo/assets/lib/store.py`
#: duplicates the store layout: the alternative is a task package importing a
#: component. Bounded to this tuple and the two regexes below.
ALLOWED_PREFIXES = (
    "/usr/", "/bin/", "/sbin/", "/lib/", "/lib64/", "/etc/", "/opt/",
    "/proc/", "/sys/", "/dev/", "/var/lib/", "/var/log/", "/run/", "/srv/",
    "/workspace/", "/app/",
)

#: `handoff.locality._CANDIDATE`, `_URL` and `_REQUEST_TARGET`, duplicated for
#: the same reason.
CANDIDATE = re.compile(
    r"(?<![A-Za-z0-9._~@+-])(?:[A-Za-z]:\\[^\s\"'<>|]*|(?:/[A-Za-z0-9._+@-]+){2,}/?)"
)
URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S*")
#: **Was missing, and its absence made this module stricter than the seal it
#: claims to mirror.** `locality._scan_text` blanks an HTTP request target before
#: running `_CANDIDATE`; this file copied `_URL` and the shebang skip and not
#: this one. Measured on a two-line tree: `locality.check` returned `None` and
#: this module exited 1 on `items/logs/access.log:1: /v1/chat/completions` out of
#: `INFO 127.0.0.1 - "GET /v1/chat/completions HTTP/1.1" 200`. Every engine
#: access log carries that line, so a handoff the seal would have taken was
#: refused here instead — and the docstring above told the next reader not to
#: check.
REQUEST_TARGET = re.compile(
    r"\b(?:GET|HEAD|POST|PUT|PATCH|DELETE|CONNECT|OPTIONS|TRACE)\s+(/\S*)\s+HTTP/\d(?:\.\d)?\b"
)

#: Matches `handoff.locality.check`, which skips anything larger. A file the seal
#: will not read does not need rewriting, and rewriting a 60 MB trace would cost
#: minutes for no effect.
MAX_BYTES = 4 << 20


#: autoconf's `@NAME@`, and the shape is load-bearing rather than stylistic.
#:
#: `${NAME}` was the first choice and it does not survive its own check:
#: `handoff.locality._CANDIDATE` begins `(?<![A-Za-z0-9._~@+-])`, and `}` is not
#: in that set, so `${TASK_PACKAGE}/assets/serve/mix_smoke.sh` still offers
#: `/assets/serve/mix_smoke.sh` as a fresh candidate and the seal rejects a line
#: this module just cleaned. `@` **is** in the set, so `@TASK_PACKAGE@/assets/...`
#: suppresses the match at the character before the slash.
PLACEHOLDER = "@%s@"


#: The environment record, in exactly the places `check_environment` looks for
#: it: `check_environment.validator/check.py:50 CANDIDATES`. Copied as a tuple of
#: globs rather than re-derived, because "which file is the record" is that
#: validator's fact and a second, looser answer here is what caused the bug this
#: tuple replaces.
#:
#: `items/codes/*/environment.yaml` is **one** level, matching the glob there. A
#: copy nested deeper — `items/codes/handoffs/<kind>/environment.yaml`, which is
#: what a packup carries — is payload that `find_record` never reads.
ENVIRONMENT_RECORDS = (
    "items/env/environment.yaml",
    "items/codes/environment.yaml",
    "items/codes/*/environment.yaml",
)


def is_environment_record(path, root) -> bool:
    """Is this the one file that must keep its absolute paths?

    **The skip is right and its old test was too broad.** It matched any
    `environment.yaml` with `env` or `codes` anywhere in its path, which
    includes the upstream records a packup nests under
    `items/codes/handoffs/<kind>/`. Those are payload — `find_record` stops at
    the single-level glob above and never reads them — so skipping them bought
    nothing and **cost the seal**: measured on a kit this package's own
    `packup.py` built, `handoff.locality.check` raised

        items/codes/handoffs/kernel_optimization/environment.yaml:7:
        '/shared_nfs/yihou/models/Qwen3.6-27B' is a local path

    and `e2e_packup` is `is_end: true`, so the flow could not produce its
    terminal artefact at all.

    **Why the record itself is still skipped, corrected.** The old note here
    said rewriting it fails `check_environment`'s
    `compare_fixed_across_inputs`. That comparison is real — `model_path` is in
    the list at `steps/common.yaml:76`, and a phase holding one redacted record
    beside one raw one does refuse. But it is not the failure that arrives
    first. Measured: `PLACEHOLDER` is `@NAME@`, and `model_path:
    @MODEL_MOUNT@/Qwen3.6-27B` is **not valid YAML** — a bare `@` cannot start a
    plain scalar — so the record becomes unparseable and every reader of it,
    including `check_environment`, dies on a `ScannerError` before any
    comparison happens. The `@` form is not optional either: it is what makes
    the placeholder survive `CANDIDATE`'s lookbehind (see `PLACEHOLDER` above).

    So: the record keeps its absolute paths, and the copies of it that are
    merely being carried do not.
    """
    if path.name != "environment.yaml":
        return False
    # Component-wise and anchored at the content root, not `PurePath.match`
    # (right-anchored, so `x/items/env/environment.yaml` would pass) and not
    # `fnmatch` (its `*` crosses `/`, so the deep copy would pass). Both of
    # those readmit exactly the paths this predicate was narrowed to exclude.
    parts = path.relative_to(root).parts
    for pattern in ENVIRONMENT_RECORDS:
        expected = pattern.split("/")
        if len(parts) == len(expected) and all(
            want == "*" or want == got for want, got in zip(expected, parts)
        ):
            return True
    return False


def substitute(text: str, mapping: list[tuple[str, str]]) -> str:
    for prefix, name in mapping:
        text = text.replace(prefix, PLACEHOLDER % name)
    return text


def blank_target(match: "re.Match[str]") -> str:
    """`locality._blank_target`: blank the request-target, keep the width."""
    text = match.group(0)
    start, end = match.span(1)
    return text[: start - match.start()] + " " * (end - start) + text[end - match.start():]


def offenders(text: str) -> list[tuple[int, str]]:
    """Every absolute path the seal would still reject, as (line, path).

    The three exemptions are `locality._scan_text`'s three, in its order. Two of
    them were here already; `REQUEST_TARGET` was not, and that made this module
    refuse artefacts the seal accepts.
    """
    out = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = REQUEST_TARGET.sub(blank_target, URL.sub(" ", line))
        if stripped.lstrip().startswith("#!"):
            continue  # a shebang names an interpreter, not a produced artefact
        for match in CANDIDATE.finditer(stripped):
            path = match.group(0)
            if not any(
                path == p.rstrip("/") or path.startswith(p) for p in ALLOWED_PREFIXES
            ):
                out.append((lineno, path))
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    root = Path(argv[0])

    mapping = []
    for spec in argv[1:]:
        name, _, prefix = spec.partition("=")
        if not prefix.startswith("/"):
            print(f"redact: {spec!r} is not NAME=/absolute/prefix", file=sys.stderr)
            return 2
        mapping.append((prefix.rstrip("/"), name))
    # Longest first: a nested pair must not depend on argument order.
    mapping.sort(key=lambda pair: len(pair[0]), reverse=True)

    rewritten = 0
    skipped: list[str] = []
    remaining: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        if is_environment_record(path, root):
            skipped.append(str(path.relative_to(root)))
            continue
        if path.stat().st_size > MAX_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary, or unreadable: the seal skips it too
        new = substitute(text, mapping)
        if new != text:
            path.write_text(new, encoding="utf-8")
            rewritten += 1
        for lineno, hit in offenders(new):
            remaining.append(f"  {path.relative_to(root)}:{lineno}: {hit}")

    print(f"redact: rewrote {rewritten} file(s) under {root}")
    if skipped:
        # Named rather than silent: a reader who expected the record to be
        # rewritten should see that it deliberately was not, and why.
        print(
            f"redact: left {len(skipped)} environment record(s) untouched "
            f"({', '.join(skipped)}) — the @NAME@ placeholder is not a valid YAML "
            "scalar start, so a rewritten record is unparseable to every reader of "
            "it. Copies carried as payload deeper than items/codes/*/ ARE rewritten."
        )
    if remaining:
        print(
            "redact: these absolute paths would still be refused by the seal.\n"
            "Add a NAME=/prefix for each, or stop putting it in the handoff:",
            file=sys.stderr,
        )
        # Deduplicated but not truncated: a partial list invites fixing one and
        # paying the whole deployment again for the next.
        for line in dict.fromkeys(remaining):
            print(line, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
