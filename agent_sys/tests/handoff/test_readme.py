"""Criterion 2, and the CommonMark cases a regex gets wrong in both directions."""

from __future__ import annotations

from pathlib import Path

import pytest

from handoff import readme
from handoff.errors import Malformed

REQUIRED = ("Purpose", "Result")


def _write(tmp_path: Path, text: str) -> Path:
    (tmp_path / readme.README_NAME).write_text(text, encoding="utf-8")
    return tmp_path


def test_missing_readme(tmp_path: Path) -> None:
    with pytest.raises(Malformed, match="README.md"):
        readme.check(tmp_path, REQUIRED)


def test_missing_section(tmp_path: Path) -> None:
    root = _write(tmp_path, "# H\n\n## Purpose\n\nWhy this exists.\n")
    with pytest.raises(Malformed, match="'Result' is missing"):
        readme.check(root, REQUIRED)


def test_a_complete_readme_passes(tmp_path: Path) -> None:
    root = _write(tmp_path, "# H\n\n## Purpose\n\nWhy.\n\n## Result\n\n42 ms.\n")
    readme.check(root, REQUIRED)


def test_section_in_code_fence_is_not_a_section(tmp_path: Path) -> None:
    """The false positives are the security-relevant half: a producer would
    otherwise satisfy the anti-blob check with headings that are invisible in
    the rendered README."""
    root = _write(tmp_path, "# H\n\n## Purpose\n\nWhy.\n\n```\n## Result\n```\n")
    with pytest.raises(Malformed, match="'Result' is missing"):
        readme.check(root, REQUIRED)


def test_a_heading_in_a_blockquote_or_list_is_not_a_section(tmp_path: Path) -> None:
    """markdownlint's MD043 filter is flat, so a heading nested in a blockquote
    satisfies it. No surveyed tool implements this refinement."""
    root = _write(tmp_path, "# H\n\n## Purpose\n\nWhy.\n\n> ## Result\n>\n> 42\n")
    with pytest.raises(Malformed, match="'Result' is missing"):
        readme.check(root, REQUIRED)

    root = _write(tmp_path, "# H\n\n## Purpose\n\nWhy.\n\n- ## Result\n")
    with pytest.raises(Malformed, match="'Result' is missing"):
        readme.check(root, REQUIRED)


def test_setext_heading_counts(tmp_path: Path) -> None:
    """A naive `^#{1,6}` regex misses it; CommonMark does not."""
    root = _write(tmp_path, "# H\n\n## Purpose\n\nWhy.\n\nResult\n------\n\n42 ms.\n")
    readme.check(root, REQUIRED)


def test_three_spaces_is_still_a_heading_and_four_is_a_code_block(tmp_path: Path) -> None:
    root = _write(tmp_path, "# H\n\n## Purpose\n\nWhy.\n\n   ## Result\n\n   42\n")
    readme.check(root, REQUIRED)

    root = _write(tmp_path, "# H\n\n## Purpose\n\nWhy.\n\n    ## Result\n")
    with pytest.raises(Malformed, match="'Result' is missing"):
        readme.check(root, REQUIRED)


def test_front_matter_does_not_invent_a_heading(tmp_path: Path) -> None:
    """`---` is a setext underline: measured, `---\\ntitle: x\\n---` parses as an
    `hr` plus `<h2>title: x</h2>`."""
    text = "---\ntitle: x\n---\n\n# H\n\n## Purpose\n\nWhy.\n\n## Result\n\n42\n"
    assert "title: x" not in readme.sections(text)
    readme.check(_write(tmp_path, text), REQUIRED)


def test_an_empty_section_is_malformed(tmp_path: Path) -> None:
    root = _write(tmp_path, "# H\n\n## Purpose\n\nWhy.\n\n## Result\n\n<!-- todo -->\n")
    with pytest.raises(Malformed, match="'Result' is empty"):
        readme.check(root, REQUIRED)


def test_nbsp_is_not_content(tmp_path: Path) -> None:
    """`&nbsp;` survives a token count and is decoded to whitespace here."""
    root = _write(tmp_path, "# H\n\n## Purpose\n\nWhy.\n\n## Result\n\n&nbsp;\n")
    with pytest.raises(Malformed, match="'Result' is empty"):
        readme.check(root, REQUIRED)


def test_the_template_is_rejected_by_the_check_it_belongs_to(tmp_path: Path) -> None:
    """A presence check cannot tell a value from a placeholder, so this one
    checks the value — and the reject list is *generated* from `template()`,
    never hand-listed, so the two cannot drift apart."""
    root = _write(tmp_path, readme.template(REQUIRED))
    with pytest.raises(Malformed, match="placeholder"):
        readme.check(root, REQUIRED)


def test_placeholder_matching_ignores_case_and_spacing(tmp_path: Path) -> None:
    root = _write(
        tmp_path,
        "# H\n\n## Purpose\n\nWhy.\n\n## Result\n\n[more   information\nneeded]\n",
    )
    with pytest.raises(Malformed, match="placeholder"):
        readme.check(root, REQUIRED)
