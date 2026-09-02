#!/usr/bin/env python3
"""`check_problems` — completeness, strong.

Total over the document. The count matches; every problem carries every declared
field and none of them is empty; every `leetcode_slug` is in the closed index;
every `direction_id` names a direction that exists in the artefact this problem
set was built from; no `id` repeats.

Nothing here is approximate — each rule is a presence test, a set membership or
a count — which is what `strong` claims and the only thing that makes the label
worth carrying. Whether the problems are *solvable* is a different question, in
a different dimension, and `check_solvable` answers it honestly as `weak`.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402 — the path insert above is what makes it importable

#: Every key a problem must carry as a non-empty string. `examples` is checked
#: separately because it is a list, and its *contents* are `check_solvable`'s.
REQUIRED_TEXT_KEYS = (
    "id",
    "direction_id",
    "leetcode_slug",
    "title",
    "statement",
    "input_format",
    "output_format",
    "constraints",
)


def package_root() -> Path:
    """Where this package's `assets/` live. The two names and the reason they
    are both needed are in `check_directions/check.py`'s copy of this function —
    a validator body's environment depends on the phase that ran it, and this
    validator runs in four phases: its producer's output phase and each of the
    three students' input phases."""
    where = os.environ.get("AGENT_SYS_TASK_PACKAGE") or os.environ.get("AGENT_SYS_DEMO_PACKAGE")
    return Path(where) if where else Path(__file__).resolve().parent.parent.parent


def catalog_slugs(relative: str) -> set | None:
    """The slugs the index permits. `None`, never an empty set, when the index
    is missing or unreadable — an empty set fails every problem and reads
    exactly like a setter who invented all twelve."""
    path = package_root() / relative
    if not path.is_file():
        print(f"check_problems: index {path} is not a file")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"check_problems: index {path} is not JSON: {exc}")
        return None
    rows = data.get("problems") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        print(f"check_problems: index {path} has no non-empty `problems` list")
        return None
    return {row.get("slug") for row in rows if isinstance(row, dict)}


def direction_ids(kind: str) -> set | None:
    """The `id`s of the directions this problem set was built from.

    **Two routes, and the first is the exact one.** `declared_dir` reads
    `AGENT_SYS_INPUT_DIRECTIONS`, which names the artefact the producing task
    actually consumed — but it is exported only where the producing task's
    configuration is in scope, and `validator.choose_configuration` uses that
    row on the OUTPUT phase only (`validator/environment.py:116-142`). In the
    three students' input phases it is simply absent.

    So the fallback is `latest_of_kind`, whose own docstring calls it crude:
    it answers *the newest `directions` anywhere in the store*, which would be
    wrong in a graph with several. This graph has exactly one producer of
    `directions` — the single-slot rule in `models.py:360` guarantees a
    consumer could not see a second one anyway — so here the crude answer and
    the exact one coincide. It reaches the store through
    `AGENT_SYS_DEMO_STORE`, which the global configuration row does carry.

    `None` when neither route resolves, which the caller must treat as *could
    not check* and never as a pass. The same shape as `check_grounded`'s
    fallback in `examples/demo`, for the same reason.

    **The second route is guarded on its own precondition, and that guard was
    written after a probe crashed on its absence, not before.**
    `store.store_root()` is `os.environ["AGENT_SYS_DEMO_STORE"]` — a subscript,
    not a `.get` — so calling `latest_of_kind` without it raises `KeyError`
    *before* `write_verdict` runs, and a body that dies writes no
    `verdict.json` at all. That is strictly worse than a False: `PhaseRunner`
    is owed one boolean per declared handoff and gets nothing, so the failure
    surfaces as a missing file rather than as this check reporting that it
    could not run. Measured in
    `scratch/demo2-2026-08/probe_front_validators.py`; the real run always
    carries the variable (`cli/main.py:601-615`), which is exactly why the
    crash would have waited for the first run that did not.
    """
    content = store.declared_dir(kind)
    if content is None and os.environ.get("AGENT_SYS_DEMO_STORE"):
        content = store.latest_of_kind(kind)
    if content is None:
        print(f"check_problems: no `{kind}` artefact reachable by either route")
        return None
    document = content / "items" / "text.json"
    if not document.is_file():
        print(f"check_problems: `{kind}` artefact has no items/text.json")
        return None
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"check_problems: `{kind}` artefact is not JSON: {exc}")
        return None
    rows = data.get("directions") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        print(f"check_problems: `{kind}` artefact has no `directions` list")
        return None
    return {row.get("id") for row in rows if isinstance(row, dict)}


def nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def check(content: Path, expected: int, slugs: set, origins: set) -> bool:
    document = content / "items" / "text.json"
    if not document.is_file():
        print("check_problems: no items/text.json")
        return False
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"check_problems: items/text.json is not JSON: {exc}")
        return False
    rows = data.get("problems") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        print("check_problems: no `problems` list")
        return False

    ok = True
    if len(rows) != expected:
        print(f"check_problems: {len(rows)} problems, expected {expected}")
        ok = False

    # Every fault is reported rather than the first, because a run here costs
    # several model calls and one fix per run is not a debug loop.
    seen: set = set()
    for index, row in enumerate(rows):
        where = f"problems[{index}]"
        if not isinstance(row, dict):
            print(f"check_problems: {where} is not an object")
            ok = False
            continue
        empty = [key for key in REQUIRED_TEXT_KEYS if not nonempty(row.get(key))]
        if empty:
            print(f"check_problems: {where} is missing or empty at {empty}")
            ok = False
        examples = row.get("examples")
        if not isinstance(examples, list) or not examples:
            # Presence only. That every example is *worked* — two of them, with
            # exact bytes on both sides — is `check_solvable`'s question.
            print(f"check_problems: {where} has no `examples` list")
            ok = False
        identifier = row.get("id")
        if identifier in seen:
            print(f"check_problems: {where} repeats id {identifier!r}")
            ok = False
        seen.add(identifier)
        if nonempty(row.get("leetcode_slug")) and row["leetcode_slug"] not in slugs:
            print(f"check_problems: {where} slug {row['leetcode_slug']!r} is not in the index")
            ok = False
        if nonempty(row.get("direction_id")) and row["direction_id"] not in origins:
            print(f"check_problems: {where} direction_id {row['direction_id']!r} does not exist")
            ok = False
    return ok


def main() -> int:
    args = store.args()
    # A string after load-time substitution. `int()` is right to raise on a
    # misspelled knob rather than quietly meaning twelve.
    expected = int(args.get("expected_count", 12))
    slugs = catalog_slugs(str(args.get("catalog") or "assets/catalog/leetcode_index.json"))
    origins = direction_ids(str(args.get("origin_kind") or "directions"))

    results = {}
    for hid in store.inputs():
        content = store.staged_content(hid) or store.content_dir(hid)
        if content is None or slugs is None or origins is None:
            # Unpublished content, an unreadable index, or an unreachable
            # directions artefact. A check that could not run has not found
            # nothing, so none of the three is a pass.
            results[hid] = False
            continue
        results[hid] = check(content, expected, slugs, origins)
    store.write_verdict(results)
    print(f"check_problems: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
