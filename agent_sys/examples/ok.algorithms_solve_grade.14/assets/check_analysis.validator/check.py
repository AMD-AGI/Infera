#!/usr/bin/env python3
"""`check_analysis` — trustworthiness, **weak**.

Every `notes.json` in the handoff names a non-empty `algorithm` and carries a
`time_complexity` and a `space_complexity` that parse as big-O.

**It checks the form of a claim, not its truth.** Nothing here reads
`solution.cpp`, so a program that is `O(n^2)` and says `O(n log n)` passes.
That is why the spec says `weak`, and why the readme beside this file spells
out what the regex does and does not accept.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402 — the path insert above is what makes it importable

#: One factor of a complexity expression. **The alternatives are ordered**, and
#: the order is the whole of why this works: `[A-Za-z]\w*` matches `log`, so the
#: logarithm and factorial forms have to be tried before the bare-name form or
#: they are eaten by it.
_FACTOR = r"""
    (?:
        \d+ \s* \^ \s* [A-Za-z]\w*                      # 2^n
      | sqrt \s* \( \s* [A-Za-z]\w* \s* \)              # sqrt(n)
      | (?:log|alpha) \s* \( \s* [A-Za-z]\w* \s* \)     # log(n), alpha(n)
        # min(m, n), max(V, E) -- two-argument, and the reason it is here is
        # measured: a `space_complexity` of `O(min(m, n))` on a Levenshtein
        # solution was rejected as "does not parse as big-O", the handoff was
        # invalidated, and the graph stalled. `min` of two dimensions is
        # ordinary notation for a rolling-buffer bound and there was nothing
        # wrong with the artefact. Arguments are simple terms rather than full
        # sums: this grammar is deliberately non-recursive, and `min(n, m)` is
        # the form that occurs.
      | (?:min|max) \s* \( \s* [A-Za-z]\w* (?:\s*\^\s*\d+)?
            (?: \s*,\s* [A-Za-z]\w* (?:\s*\^\s*\d+)? )+ \s* \)
      | log \s* (?: \^ \s* \d+ \s* )? [A-Za-z]\w*       # log n, log^2 n
      | [A-Za-z]\w* \s* !                               # n!
      | [A-Za-z]\w* \s* \^ \s* \d+                      # n^2
      | [A-Za-z]\w*                                     # n, m, V, E
      | \d+                                             # 1
    )
"""
#: A product: factors separated by whitespace or `*`. The separator is required
#: rather than optional, so the expression has to be written the way a reader
#: would write it.
_PRODUCT = rf"{_FACTOR} (?: (?: \s*\*\s* | \s+ ) {_FACTOR} )*"
#: A sum of products: `O(m + n)`, `O(V + E)`.
_SUM = rf"{_PRODUCT} (?: \s*\+\s* {_PRODUCT} )*"
BIG_O = re.compile(rf"\A \s* [Oo] \s* \( \s* {_SUM} \s* \) \s* \Z", re.VERBOSE)


def check_notes(path: Path) -> tuple[bool, str]:
    """One `notes.json`."""
    if not path.is_file():
        return False, "no notes.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"notes.json is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return False, "notes.json is not a JSON object"

    algorithm = data.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm.strip():
        return False, "algorithm is missing or empty"
    for key in ("time_complexity", "space_complexity"):
        value = data.get(key)
        if not isinstance(value, str) or not BIG_O.match(value):
            return False, f"{key} does not parse as big-O: {value!r}"
    return True, "ok"


def check(content: Path) -> tuple[bool, dict]:
    """The whole handoff. `all` over its problem directories, and empty is False.

    `all([])` is True, so a handoff carrying no notes at all would otherwise be
    indistinguishable from one where every note was well formed.
    """
    codes = content / "items" / "codes"
    if not codes.is_dir():
        return False, {"<content>": "no items/codes directory"}
    directories = sorted(e for e in codes.iterdir() if e.is_dir())
    if not directories:
        return False, {"<content>": "items/codes is empty"}
    detail = {}
    for directory in directories:
        passed, why = check_notes(directory / "notes.json")
        detail[directory.name] = ("ok: " if passed else "FAIL: ") + why
    return all(line.startswith("ok: ") for line in detail.values()), detail


def main() -> int:
    results = {}
    for hid in store.inputs():
        # `materials.json` first, `content_dir` as the fallback for a run with
        # no `env_mgr` wired. A handoff with no published content is not a pass.
        content = store.staged_content(hid) or store.content_dir(hid)
        if content is None:
            results[hid] = False
            print(f"check_analysis: {hid}: no published content")
            continue
        results[hid], detail = check(content)
        for name, line in sorted(detail.items()):
            print(f"check_analysis: {hid}: {name}: {line}")
    store.write_verdict(results)
    print(f"check_analysis: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
