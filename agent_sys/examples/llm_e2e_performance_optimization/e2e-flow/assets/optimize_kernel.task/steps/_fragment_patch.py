#!/usr/bin/env python3
"""Build a `patch` entry for a `call_site_fragment` operator.

**The applier could always do this; the producer could never ask for it.**
`apply.py:643-659` has run `patch -p1 --batch --forward` since it was written and
`patchkit.FILE_REQUIRED` lists `patch` — but `60_write_handoff.py` only ever
emitted `replacement`, and `workset.schema.json`'s `apply_mode` enum had exactly
one value. So a fragment operator had no way to ask for the path that already
existed, and m5's applier refused the only shape it could be given. Leader's
ruling, 2026-09-04, after the user's *"最挫的 apply 方式难道不是找到调用的地方，
把那一行替换掉"* — which is a diff, and always was.

**Anchor on the fragment, never on `entry_function_line`.** Measured: the workset
records `entry_function_line: 183` and the fragment sits at **line 207** of the
image this patch is cut against — the analysis was done on sglang 0.5.14 and the
deployed image is 0.5.16. The line number is precise, plausible and 24 lines
wrong; the source text is not.

**The marker rides inside the edit.** `check_patch_live` greps the *engine log*
for `runtime_marker.first_call`, and a `call_site_fragment` operator has no
`public_symbol` to derive one from — which is why synthesising a `public_symbol`
was refused this morning by both m4 and m5. A diff that edits a line can add a
line, so the thing that proves execution is the thing that was installed.

Two properties the inserted line must have, and a comment satisfies neither:

* **it must execute** — a comment changes bytes, so it passes `apply.py:747`'s
  *"applied but changed nothing"* gate, and then produces no log hit and fails
  `check_patch_live` at the far end of a bring-up;
* **it must fire once per process**, not once per token. The reference figure is
  *"18 import hits and 8 first_call hits on an 8-rank deployment"* — per rank,
  not per decode step.
"""

from __future__ import annotations

import difflib
import re

#: Set on the module logger the first time the fragment runs. An attribute
#: rather than a module-level flag: it adds no name to the module surface, which
#: is the thing `module_symbols` and m5's `surface_regressions` both count.
_GUARD = "_m4_marker_emitted"


def marker_line(operator_id: str, rev: str, indent: str, *, no_optimisation: bool) -> str:
    """The line the diff inserts, and the token `first_call` will match.

    **The token says what the patch is.** A marker-only diff makes
    `check_patch_live` green on a patch that optimises nothing, and a green mock
    rung must not be readable as evidence of an optimisation — six months out a
    reader has the overlay and not the thread that produced it. So the *token
    itself* carries it, not a note beside it (leader's condition, 2026-09-04).
    """
    token = ("M4_MARKER_ONLY_NO_OPTIMISATION" if no_optimisation else "M4_FRAGMENT_FIRST_CALL")
    why = "  # marker only: this patch applies NO optimisation" if no_optimisation else ""
    return (
        f'{indent}if not getattr(logger, "{_GUARD}", False):{why}\n'
        f'{indent}    logger.warning("{token} {operator_id} {rev}")\n'
        f'{indent}    setattr(logger, "{_GUARD}", True)\n'
    )


def first_call_regex(operator_id: str, rev: str, *, no_optimisation: bool) -> str:
    token = ("M4_MARKER_ONLY_NO_OPTIMISATION" if no_optimisation else "M4_FRAGMENT_FIRST_CALL")
    return rf"{token}\s+{re.escape(operator_id)}\s+{re.escape(rev)}"


def build(stock_text: str, rel_path: str, anchor: str, operator_id: str, rev: str,
          *, no_optimisation: bool, replacement_fragment: str | None = None) -> tuple[str, dict]:
    """`(unified diff, runtime_marker)` — or raise with the reason.

    `anchor` is the fragment's source text, taken from the workset rather than
    reconstructed. `replacement_fragment` replaces it when a campaign produced
    one; `None` keeps it and inserts only the marker, which is the mock.
    """
    lines = stock_text.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if anchor in line]
    if not hits:
        raise SystemExit(
            f"the fragment {anchor!r} is not in {rel_path} as extracted from the image. "
            "The workset's `entry_function_line` is NOT a fallback -- it is recorded against "
            "the version analysed, which is not necessarily the version deployed (measured: "
            "183 recorded, 207 actual). A patch cut against a file that does not contain the "
            "fragment would apply somewhere else or not at all"
        )
    if len(hits) > 1:
        raise SystemExit(
            f"the fragment {anchor!r} appears {len(hits)} times in {rel_path}; a diff would be "
            "ambiguous about which call site it edits. The workset must name a unique fragment"
        )
    at = hits[0]
    indent = re.match(r"[ \t]*", lines[at]).group(0)

    # **`splitlines(keepends=True)`, because `difflib` prefixes list ELEMENTS.**
    # Passing the three-line marker as one element produced a diff whose first
    # line carried `+` and whose next two carried nothing — they read as context
    # lines that are not in the original, and `patch -p1` refused with
    # `Hunk #1 FAILED at 204`. Caught by applying it rather than by reading it.
    body = marker_line(operator_id, rev, indent, no_optimisation=no_optimisation).splitlines(keepends=True)
    patched = list(lines)
    if replacement_fragment is None:
        patched[at:at] = body                          # marker, then the untouched call site
    else:
        patched[at:at + 1] = body + [replacement_fragment]

    diff = "".join(difflib.unified_diff(
        lines, patched, fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}", n=3
    ))
    if not diff.strip():
        raise SystemExit(f"the generated diff for {rel_path} is empty; nothing would be applied")
    return diff, {"first_call": first_call_regex(operator_id, rev, no_optimisation=no_optimisation)}
