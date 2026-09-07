#!/usr/bin/env python3
"""The four body-facing files a validation zone carries, and nothing else.

`validator/phase.py:236` names them: `args.json`, `inputs.json`,
`materials.json`, `verdict.json`. A validator body is started with `cwd` set to
a freshly allocated zone holding the first three, and it owes the fourth.

**This is deliberately not `examples/demo2/assets/lib/store.py`.** That module
reads `handoff`'s on-disk store layout directly — version-directory naming, the
manifest filename, the `content/` hop — through `AGENT_SYS_DEMO_STORE`, and it
is admissible there only because it is a verbatim copy of `examples/demo`'s and
`tests/cli/test_isolation_shown.py` pins *that* copy against `handoff`'s real
constants. A third copy here would be a third reader of a layout `handoff` owns,
and an unpinned one.

Neither validator in this package needs it. Both check the handoff they were
handed and nothing else, and `materials.json` names exactly that — so the whole
store-layout question does not arise. It matters twice over here:
`AGENT_SYS_DEMO_STORE` is **absent** from the environment on the output phase —
measured, and it is the PRODUCER row of `validator` spec §8.2 — which is this
package's only phase, so a body reaching for the store root would die on
`KeyError` rather than fall back.

There is one consequence and it is stated rather than worked around: a check
that must reach a handoff it was *not* handed cannot be written with this
module. That is F-D5 (`examples/demo/README.md`) and neither check here is one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

__all__ = ["PACKUP_DIRNAME", "args", "find_packup", "inputs", "materials", "write_verdict"]

#: The packup skill's folder naming: `<experiment-name>.packup_<YYYYMMDD>`.
#: Anchored at both ends, and the date is exactly eight digits — a folder called
#: `something.packup_soon` is not a packup and saying so is free.
PACKUP_DIRNAME = re.compile(r"\A.+\.packup_\d{8}\Z")


def inputs() -> list[str]:
    """The handoff ids this body is validating, as `ScriptBodyRunner` wrote them."""
    return list(json.loads(Path("inputs.json").read_text(encoding="utf-8")))


def args() -> dict:
    """This validator spec's `args` block, or `{}` when it declared none."""
    path = Path("args.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def materials() -> dict[str, Path]:
    """The staged copies of what this phase validates, **by handoff id**.

    Written unconditionally (`validator/phase.py:269`), so an empty mapping is a
    record rather than an absence. It is a JSON **object**; `demo2`'s copy of
    this idea read it as a list once and every lookup silently missed.
    """
    path = Path("materials.json")
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return {str(hid): Path(where) for hid, where in loaded.items()}


def content_of(hid: str) -> Path | None:
    """The staged content directory of one handoff, or `None`.

    The staged path **is** the content directory — `env_mgr` narrowed `stage` to
    `content/`, so there is no `content/` hop below it. `None` means the phase
    staged nothing for this id, and a caller must treat that as *no content* and
    never as a pass.
    """
    staged = materials().get(hid)
    return staged if staged is not None and staged.is_dir() else None


def find_packup(content: Path) -> tuple[Path | None, str]:
    """The one packup directory inside a `code`-typed handoff's content.

    Returns `(path, reason)`; `path` is `None` iff the content does not hold
    exactly one. **Exactly one is the rule**, and both other cardinalities are
    real failures rather than pedantry: zero means the agent wrote its kit
    somewhere the layout does not put it, and two means a reproducer has to
    guess which kit is the one that worked.
    """
    codes = content / "items" / "codes"
    if not codes.is_dir():
        return None, "no items/codes directory"
    found = sorted(e for e in codes.iterdir() if e.is_dir() and PACKUP_DIRNAME.match(e.name))
    if not found:
        loose = sorted(e.name for e in codes.iterdir())
        return None, f"no <name>.packup_<YYYYMMDD> directory under items/codes (found: {loose})"
    if len(found) > 1:
        return None, f"more than one packup directory: {[e.name for e in found]}"
    return found[0], found[0].name


def write_verdict(results: dict[str, bool]) -> None:
    """`verdict.json`: one entry per declared handoff. A missing one raises at
    `PhaseRunner`'s seam rather than folding as falsy."""
    Path("verdict.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
