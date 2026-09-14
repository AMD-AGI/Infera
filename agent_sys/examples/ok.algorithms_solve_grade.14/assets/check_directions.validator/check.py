#!/usr/bin/env python3
"""`check_directions` — completeness, strong.

Four rules, total over the document: the count matches, every
`(clrs_chapter, leetcode_tag)` pair is a row of the closed catalogue, no `id`
repeats, and every `why` is prose rather than a placeholder.

Three of the four are exact. The fourth reduces "prose" to a character floor and
a word count, and says so both here and in `readme.md` — a crude check that is
honestly described is a strong validator; a sophisticated one that is silently
approximate is not.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402 — the path insert above is what makes it importable

#: What "prose" is reduced to. A placeholder — `TBD`, `important`, `sorting` —
#: fails both; any real sentence passes both with room to spare. Stated as
#: constants rather than buried in a condition, because the readme quotes them
#: and a reader has to be able to check that the quote is true.
MIN_WHY_CHARS = 24
MIN_WHY_WORDS = 4

#: Every key a direction must carry.
REQUIRED_KEYS = ("id", "title", "clrs_chapter", "leetcode_tag", "why")


def package_root() -> Path:
    """Where this package's `assets/` live.

    Two names, because a validator body's environment depends on which phase
    ran it: `AGENT_SYS_TASK_PACKAGE` is a task body's variable and an INPUT
    phase — which is how `problems` re-runs this validator — gets the global
    configuration row instead, carrying `AGENT_SYS_DEMO_PACKAGE`. Same
    resolution order as `entry.sh`, deliberately, so the script and the shell
    that launched it cannot disagree about where they are.

    Kept here rather than added to `assets/lib/store.py`: that file is a
    verbatim copy of demo-1's and an agreement test compares it against
    `handoff`'s layout. Six lines duplicated in two validators is the cheaper
    of the two costs.
    """
    where = os.environ.get("AGENT_SYS_TASK_PACKAGE") or os.environ.get("AGENT_SYS_DEMO_PACKAGE")
    return Path(where) if where else Path(__file__).resolve().parent.parent.parent


def catalog_pairs(relative: str) -> set | None:
    """The `(clrs_chapter, leetcode_tag)` pairs the catalogue permits.

    `None` — never an empty set — when the catalogue is missing or unreadable.
    The distinction is the whole guard: an empty set silently fails every
    direction and reads exactly like a teacher who invented all five, while
    `None` reports that the check could not run.
    """
    path = package_root() / relative
    if not path.is_file():
        print(f"check_directions: catalogue {path} is not a file")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"check_directions: catalogue {path} is not JSON: {exc}")
        return None
    topics = data.get("topics") if isinstance(data, dict) else None
    if not isinstance(topics, list) or not topics:
        print(f"check_directions: catalogue {path} has no non-empty `topics` list")
        return None
    return {
        (row.get("clrs_chapter"), row.get("leetcode_tag"))
        for row in topics
        if isinstance(row, dict)
    }


def is_prose(value: object) -> bool:
    """Long enough and made of enough words to be a sentence someone wrote."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    return len(text) >= MIN_WHY_CHARS and len(text.split()) >= MIN_WHY_WORDS


def check(content: Path, expected: int, pairs: set) -> bool:
    document = content / "items" / "text.json"
    if not document.is_file():
        print("check_directions: no items/text.json")
        return False
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"check_directions: items/text.json is not JSON: {exc}")
        return False
    rows = data.get("directions") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        print("check_directions: no `directions` list")
        return False

    ok = True
    if len(rows) != expected:
        print(f"check_directions: {len(rows)} directions, expected {expected}")
        ok = False

    # **Every fault is reported, not just the first.** A validator that returns
    # on the first bad row makes the author fix one thing per run, and a run
    # here costs a model call.
    seen: set = set()
    for index, row in enumerate(rows):
        where = f"directions[{index}]"
        if not isinstance(row, dict):
            print(f"check_directions: {where} is not an object")
            ok = False
            continue
        missing = [key for key in REQUIRED_KEYS if key not in row]
        if missing:
            print(f"check_directions: {where} is missing {missing}")
            ok = False
            continue
        identifier = row["id"]
        if identifier in seen:
            print(f"check_directions: {where} repeats id {identifier!r}")
            ok = False
        seen.add(identifier)
        pair = (row["clrs_chapter"], row["leetcode_tag"])
        if pair not in pairs:
            print(f"check_directions: {where} pair {pair!r} is not a catalogue row")
            ok = False
        if not is_prose(row["why"]):
            print(
                f"check_directions: {where} `why` is not prose "
                f"(needs >= {MIN_WHY_CHARS} chars and >= {MIN_WHY_WORDS} words)"
            )
            ok = False
    return ok


def main() -> int:
    args = store.args()
    # `${n_directions:-5}` is substituted at load time and arrives as a string.
    # `int()` on a value the package author wrote is right to raise: a
    # misspelled knob must stop the run rather than silently become 5.
    expected = int(args.get("expected_count", 5))
    pairs = catalog_pairs(str(args.get("catalog") or "assets/catalog/clrs_topics.json"))

    results = {}
    for hid in store.inputs():
        # `materials.json` first: it is the declared route and a body reading it
        # needs to know neither that a store exists nor where it is.
        # `content_dir` is the fallback for a run with no `env_mgr` wired.
        content = store.staged_content(hid) or store.content_dir(hid)
        if content is None or pairs is None:
            # A handoff with no published content is not a pass, and neither is
            # a check whose catalogue never loaded. Not finding anything is not
            # the same as finding nothing wrong.
            results[hid] = False
            continue
        results[hid] = check(content, expected, pairs)
    store.write_verdict(results)
    print(f"check_directions: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
