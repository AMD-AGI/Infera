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

import ast
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


def entry_body_line(stock_text: str, entry_function: str) -> tuple[int, str]:
    """`(0-based line of the first statement, its indent)` for `Class.method`.

    **Located by AST, never by `entry_function_line`.** The workset records
    `183` and the statement is at `207` in the image the patch is cut against —
    the analysis was sglang 0.5.14, the deployment is 0.5.16. A line number is
    precise, plausible and 24 lines wrong, which is worse than an absent field
    because it invites use.

    **And not by an anchor string either**, which is what this first tried:
    the workset declares no fragment text. The nearest thing is prose inside
    `integration.invariants` — *"the call site is `logits[:] = …`"* — and
    parsing a sentence for the thing a patch will be cut against is the kind of
    dependency that works until someone rewords a comment. Asking m3 to declare
    a fragment field would be the alternative, and it is not needed: **the
    marker has to prove the function was ENTERED, so the top of its body is
    exactly the right place** and needs nothing the workset does not already
    say.
    """
    want = entry_function.split(".")
    tree = ast.parse(stock_text)

    def find(nodes, path):
        head, rest = path[0], path[1:]
        for node in nodes:
            if getattr(node, "name", None) != head:
                continue
            if not rest:
                return node
            return find(node.body, rest)
        return None

    node = find(tree.body, want)
    if node is None or not getattr(node, "body", None):
        raise SystemExit(
            f"`edit_target.entry_function` is {entry_function!r} and no such function is defined "
            "in the file extracted from the image. The workset's `entry_function_line` is NOT a "
            "fallback: it is recorded against the version analysed, not the version deployed "
            "(measured: 183 recorded, 207 actual)"
        )
    first = node.body[0]
    # A docstring is the body's first statement and the marker belongs after it,
    # or the diff moves the docstring and `help()` shows a log line.
    if (isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant)
            and isinstance(first.value.value, str) and len(node.body) > 1):
        first = node.body[1]
    at = first.lineno - 1
    indent = re.match(r"[ \t]*", stock_text.splitlines(keepends=True)[at]).group(0)
    return at, indent


def build(stock_text: str, rel_path: str, entry_function: str, operator_id: str, rev: str,
          *, no_optimisation: bool) -> tuple[str, dict]:
    """`(unified diff, runtime_marker)` — or raise with the reason.

    The marker is inserted at the top of `entry_function`'s body. A campaign's
    own edits do not come through here: for the real path the diff is
    `stock` against the file forge edited in place, and this adds the marker to
    that instead of replacing it.
    """
    lines = stock_text.splitlines(keepends=True)
    at, indent = entry_body_line(stock_text, entry_function)

    # **`splitlines(keepends=True)`, because `difflib` prefixes list ELEMENTS.**
    # Passing the three-line marker as one element produced a diff whose first
    # line carried `+` and whose next two carried nothing — they read as context
    # lines that are not in the original, and `patch -p1` refused with
    # `Hunk #1 FAILED at 204`. Caught by applying it rather than by reading it.
    body = marker_line(operator_id, rev, indent, no_optimisation=no_optimisation).splitlines(keepends=True)
    patched = list(lines)
    patched[at:at] = body

    diff = "".join(difflib.unified_diff(
        lines, patched, fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}", n=3
    ))
    if not diff.strip():
        raise SystemExit(f"the generated diff for {rel_path} is empty; nothing would be applied")
    return diff, {"first_call": first_call_regex(operator_id, rev, no_optimisation=no_optimisation)}
