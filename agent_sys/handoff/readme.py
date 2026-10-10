"""The README check: three layers, a CommonMark AST, and no autofix.

Criterion 2 is "no README, or a missing section, is malformed". A
presence-only check on an agent-authored artefact is theatre by construction —
an LLM told "your README must have sections X, Y, Z" emits exactly X, Y and Z
with placeholder bodies. Hugging Face's live validator returns HTTP 200 for a
card whose entire prose is `[More Information Needed]`, a string its own
template emits 39 times and which appears in 636,321 repositories.

So three layers: the section **exists** at document root, has **non-empty**
inline text, and is **not a placeholder** — `design.md` §9.1.

Imports nothing of this package except `errors`, which is what lets it be
tested with no store and reused without a cycle.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from markdown_it import MarkdownIt

from handoff.errors import Malformed

__all__ = ["PLACEHOLDERS", "README_NAME", "check", "sections", "template"]

README_NAME = "README.md"

#: The one placeholder body our own template emits. Every reject string is
#: **derived** from `template()` below rather than hand-listed, because a
#: hand-written list drifts away from the templates that produce the strings it
#: is meant to catch.
_TEMPLATE_BODY = "[More Information Needed]"

_MD = MarkdownIt("commonmark")

#: A leading YAML front-matter block. Stripped before parsing because `---` is
#: a setext underline: measured, `---\ntitle: x\n---` parses as an `hr` plus an
#: `<h2>title: x</h2>`, so a document with front matter acquires a heading
#: nobody wrote. mdtoc carries a `stripFrontMatter` for the same reason.
_FRONT_MATTER = re.compile(r"\A---\r?\n.*?\r?\n---[ \t]*\r?\n", re.DOTALL)

#: Inline token types whose `content` is rendered text a reader sees. Entities
#: are already decoded into `text` children by the parser, which is why
#: `&nbsp;` does not pass the non-empty test.
_TEXT_TYPES = frozenset({"text", "code_inline"})

#: A line break carries no `content`, and joining children without it turns
#: `information\nneeded` into `informationneeded` — which would let a
#: line-wrapped placeholder through the reject list.
_BREAK_TYPES = frozenset({"softbreak", "hardbreak"})


def _inline_text(token) -> str:  # noqa: ANN001 - markdown_it.token.Token
    if token.children:
        return "".join(
            " " if c.type in _BREAK_TYPES else c.content
            for c in token.children
            if c.type in _TEXT_TYPES or c.type in _BREAK_TYPES
        )
    return token.content


def sections(text: str) -> dict[str, str]:
    """Map every **document-root** heading to the inline text of its body.

    Parsed to a CommonMark AST rather than matched with `^#{1,6}\\s+(.+)$`,
    because the regex is wrong in **both** directions, measured: it misses a
    setext heading and a heading indented by up to three spaces, and it finds
    `## Results` inside a fenced code block. The false positives are the
    security-relevant half — a producer satisfies the anti-blob check with
    headings that are invisible in the rendered README.

    `token.level == 0` is the document root. markdownlint's MD043 filter is
    flat, so a heading inside a blockquote or a list item satisfies it; no
    surveyed tool implements this refinement, and measured, both nest at
    `level >= 1`.

    A duplicate heading keeps the **first** body, so a later empty repeat
    cannot rescue an empty section and an earlier empty one cannot be rescued.
    """
    tokens = _MD.parse(_FRONT_MATTER.sub("", text))
    out: dict[str, str] = {}
    title: str | None = None
    body: list[str] = []

    def flush() -> None:
        if title is not None and title not in out:
            out[title] = " ".join(p for p in body if p).strip()

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "heading_open" and tok.level == 0:
            flush()
            title = _inline_text(tokens[i + 1]).strip() if i + 1 < len(tokens) else ""
            body = []
            i += 3  # heading_open, inline, heading_close
            continue
        if title is not None:
            if tok.type == "inline":
                body.append(_inline_text(tok))
            elif tok.type in ("code_block", "fence"):
                body.append(tok.content)
        i += 1
    flush()
    return out


def template(required: Iterable[str]) -> str:
    """A README skeleton for `required`. **The source of the reject list.**

    Emitted with `PLACEHOLDERS` as every body precisely so that `check` rejects
    what this function produces: a template filled in by nobody is not a
    README, and the two must not be able to drift apart.
    """
    parts = ["# Handoff", ""]
    for name in required:
        parts += [f"## {name}", "", _TEMPLATE_BODY, ""]
    return "\n".join(parts)


#: Derived, not written: whatever bodies `template()` emits are what a filled-in
#: -by-nobody README contains. Compared case-insensitively after whitespace
#: collapse, so `[more information needed]` does not slip through.
PLACEHOLDERS = frozenset(v.casefold() for v in sections(template(("_",))).values() if v)

_WS = re.compile(r"\s+")


def check(content_dir: Path, required: Sequence[str]) -> None:
    """Raise `Malformed` unless every name in `required` is a real section.

    Runs before publication, never after: a malformed handoff that reached
    storage would need retracting, and nobody anywhere has solved retraction
    (`design.md` §15 O3).

    **Never edits the README.** Ruff #23562 is the precedent — line-by-line
    section matching misfired inside a `.. code-block:: yaml` and `--fix`
    rewrote the user's docstring. This raises and names the section.
    """
    readme = Path(content_dir) / README_NAME
    if not readme.is_file():
        raise Malformed(
            f"{readme}: every handoff opens with a {README_NAME} "
            f"(spec §3.1). An artefact only a program can open is a blob, and "
            f"blobs do not get reviewed"
        )

    found = sections(readme.read_text(encoding="utf-8", errors="replace"))
    for name in required:
        if name not in found:
            raise Malformed(
                f"{readme}: required section {name!r} is missing. "
                f"Present at document root: {sorted(found) or 'none'} — a "
                f"heading inside a blockquote, a list or a code fence is not a "
                f"section"
            )
        text = _WS.sub(" ", found[name]).strip()
        if not text:
            raise Malformed(f"{readme}: required section {name!r} is empty")
        if text.casefold() in PLACEHOLDERS:
            raise Malformed(
                f"{readme}: required section {name!r} still holds the template "
                f"placeholder {text!r}. A presence check cannot tell a value "
                f"from a placeholder, so this one checks the value"
            )
