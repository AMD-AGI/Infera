#!/usr/bin/env python3
"""Make evidence publishable: replace site-specific roots with placeholders.

**The rule this enforces is portability, and it is NOT the seal's.** CONTRACT
§2.2, which this file spent two rounds contradicting: `handoff/store.py:447` and
`:493` both read `# locality.check — NOT CALLED. User-ruled 2026-08-31` — the
shape heuristic read an HTTP access-log line as a filesystem path and refused a
correct artefact, 97% false positive on a real kit. Corroborated from the other
side: the sealed `stage1-deploy/deploy_kit` carries `/shared_nfs/...` in five
content files and sealed cleanly.

So nothing here may say "the seal would refuse this", and two rounds of this
file's history are what that instruction is made of. The rule stands on its own
merit instead: **a script carrying one host's directory layout does not run on
the next host.** That merit is what scopes it, below.

The substitution is conda-build's `PREFIX_PLACEHOLDER` design.
`@MODEL_MOUNT@/GLM-5.3-Flash-FP8` keeps the model's identity and drops the mount
root, which is the split spec §7 asks for.

**Substitution is broad; refusal is narrow, and they are deliberately not the
same set** (m1's shape, and it is the right one). A real host path is worth
rewriting wherever it appears, prose included — a README naming
`/shared_nfs/yihou/...` is still telling a reader about one machine. But the
hard refusal only reaches **executable and generated content**, `REFUSE_SUFFIXES`
below, because that is the only content the portability argument reaches. Prose
saying `` `/v1/models` `` bakes in no host's directory layout, and refusing it
asks m1 to make an endpoint's documentation worse to satisfy a scanner.

What is left after substitution, in a file the rule reaches, is reported with
its file, line and text — so an unnameable host path shows up here rather than
as a deployment that ran and cannot be published.

Usage:

    redact.py <dir> NAME=/absolute/prefix [NAME=/another ...]

Longer prefixes are substituted first, so a nested pair like
`/data/work` and `/data/work/profiles` cannot shadow each other by argument
order.
"""

import re
import sys
from pathlib import Path

#: CONTRACT §2.2: *"scope it to executable and generated content (`.py`, `.sh`,
#: `.json`, `.jsonl`), skipping the environment record."* Verbatim, because this
#: file previously refused over **every** file it could decode as UTF-8 —
#: `.md` prose included — and that is what made `packup` unable to write
#: `e2e_packup` over m1's and m4's perfectly good documentation.
#:
#: The environment record needs no entry here: `.yaml` is not in the set, so the
#: suffix alone already excludes it. `is_environment_record` below is the second
#: guard and it governs **substitution**, which this does not.
REFUSE_SUFFIXES = frozenset({".py", ".sh", ".json", ".jsonl"})

#: `handoff.locality.ALLOWED_PREFIXES`, duplicated. A second reader of a fact
#: that module owns, admissible for the same reason `demo/assets/lib/store.py`
#: duplicates the store layout: the alternative is a task package importing a
#: component. Bounded to this tuple and the two regexes below.
#:
#: **Deliberately identical, and `/v1/` is not added to it** (m1's call, and it
#: is right): the tuple's whole claim is that it mirrors that module's, and a
#: divergence with no reason attached is worse than the false positive it would
#: silence. A request target is excluded by `REQUEST_TARGET` and by
#: `REFUSE_SUFFIXES`, which are the two places it belongs.
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
#: An HTTP request target is not a filesystem path, so it is not a portability
#: problem and `CANDIDATE` alone cannot tell the two apart. Measured on
#: `INFO 127.0.0.1 - "GET /v1/chat/completions HTTP/1.1" 200`: without this,
#: `/v1/chat/completions` is reported as a local path. Every engine access log
#: carries that line.
#:
#: `locality._scan_text` blanks it the same way and this mirrors that shape —
#: **as a heuristic worth copying, not because that module gates anything.** It
#: does not; see the docstring.
REQUEST_TARGET = re.compile(
    r"\b(?:GET|HEAD|POST|PUT|PATCH|DELETE|CONNECT|OPTIONS|TRACE)\s+(/\S*)\s+HTTP/\d(?:\.\d)?\b"
)

#: The same 4 MB ceiling `handoff.locality.check` uses. Rewriting a 60 MB trace
#: would cost minutes, and a file that large is evidence rather than something
#: anyone re-executes on another host.
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


def is_environment_record(path, root) -> bool:
    """An environment record, anywhere: never substituted into.

    CONTRACT §2.2 ends *"skipping the environment record"*, and the mechanism is
    sharper than the reason first written here. The old note said rewriting it
    fails `check_environment`'s `compare_fixed_across_inputs`. That comparison
    is real — `model_path` is in the list at `steps/common.yaml:76` — but it is
    not what arrives first. Measured: `PLACEHOLDER` is `@NAME@`, and
    `model_path: @MODEL_MOUNT@/Qwen3.6-27B` is **not valid YAML**, because a
    bare `@` cannot start a plain scalar. A rewritten record is unparseable to
    every reader, `check_environment` included, before any comparison happens.
    The `@` form is not optional: it is what makes the placeholder survive
    `CANDIDATE`'s lookbehind (see `PLACEHOLDER` above).

    **Broad on purpose, and one round of this file narrowed it and was wrong.**
    That round restricted the skip to the three paths
    `check_environment.find_record` reads, so that the copies a packup nests at
    `items/codes/handoffs/<kind>/environment.yaml` would be substituted into —
    on the theory that they had to be, or the artefact could not be sealed.
    They did not: `locality.check` is not called at a seal at all (§2.2), and
    with refusal now scoped to `REFUSE_SUFFIXES` a `.yaml` is never refused
    anywhere. So the narrowing bought nothing and turned a carried record into
    invalid YAML — damage to payload, in exchange for a blocker that did not
    exist.

    A record is a record wherever it is carried, so the test is the filename
    under any item directory.
    """
    if path.name != "environment.yaml":
        return False
    # Under an item directory only. An `environment.yaml` a producer wrote into
    # `result/` is its own artefact and is not the record.
    parts = path.relative_to(root).parts
    return "env" in parts or "codes" in parts


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
    """Every host path still naming a machine after substitution, as (line, path).

    The three exemptions — a URL, a request target, a shebang — are the three
    things that look like a path and carry no host's directory layout.
    `locality._scan_text` applies the same three in this order, which is why
    they are spelled the same way; that module is a good source for the
    heuristic and is not the thing enforcing it.

    Whether a hit is *refused* is the caller's decision, scoped by
    `REFUSE_SUFFIXES`. This function only finds them.
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
            continue  # binary, or unreadable: nothing to rewrite and nothing to read
        new = substitute(text, mapping)
        if new != text:
            path.write_text(new, encoding="utf-8")
            rewritten += 1
        # **Substitute everywhere, refuse only here.** CONTRACT §2.2. A host path
        # in prose was still worth rewriting above; only the hard refusal is
        # scoped, because only executable and generated content is what the
        # portability argument is about.
        if path.suffix not in REFUSE_SUFFIXES:
            continue
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
            "it. CONTRACT 2.2."
        )
    if remaining:
        print(
            "redact: these host paths are still named in executable or generated "
            f"content ({', '.join(sorted(REFUSE_SUFFIXES))}), so this artefact does "
            "not run on another host.\n"
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
