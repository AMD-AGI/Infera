"""Criterion 17, and the false positives that make the naive version useless.

Measured on this repository: a bare absolute-path regex over 276 files gives
650 matches of which **23 are genuinely local and 627 (96%) need a suppression
rule**. Every project that made a path check mandatory acquired false positives
within a release or two — delocate #255, Bazel #26150, lintian #1002451,
rpmlint #1350, conda-build #1409 — so the negatives are pinned as executable
tests, which is delocate's own discipline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from handoff import locality
from handoff.errors import Malformed


def _content(root: Path, text: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "notes.md").write_text(text, encoding="utf-8")
    return root


def test_oracle_hit_rejected(tmp_path: Path) -> None:
    """An oracle hit is **certain**: a playground path in a published artefact
    is a record of one machine's afternoon by construction."""
    playground = tmp_path / "playground" / "task-7"
    oracles = locality.Oracles(playground_root=playground)
    root = _content(tmp_path / "c", f"we ran it in {playground}/run3\n")

    with pytest.raises(Malformed) as exc:
        locality.check(root, oracles=oracles)
    assert "oracle" in str(exc.value) and "notes.md:1" in str(exc.value)


def test_a_heuristic_hit_says_it_is_a_heuristic(tmp_path: Path) -> None:
    """The message states which rule fired, because a check that claims more
    than it delivers is how Debian ended up unable to see its own build-path
    leakage across 1841 packages."""
    root = _content(tmp_path / "c", "cd /home/someone/run3 && ./go\n")
    with pytest.raises(Malformed) as exc:
        locality.check(root, oracles=locality.Oracles())
    assert "heuristic" in str(exc.value)


@pytest.mark.parametrize(
    "line",
    [
        "#!/usr/bin/env python3",
        "LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu",
        "see https://example.com/a/b/c for details",
        "the ROCm build lives at /opt/rocm/bin",
        "workdir /workspace/repo/build",
        "read ./data/in.json relative to here",
        "one /segment only",
    ],
)
def test_the_pinned_negatives(tmp_path: Path, line: str) -> None:
    """Each row is one of the 627. Pinned as executable tests rather than kept
    in a comment, because that is the half every surveyed project got wrong."""
    locality.check(_content(tmp_path / "c", line), oracles=locality.Oracles())


def test_system_path_allowed(tmp_path: Path) -> None:
    root = _content(tmp_path / "c", "libs from /usr/lib and /etc/hosts and /var/log/x\n")
    locality.check(root, oracles=locality.Oracles())


def test_url_allowed(tmp_path: Path) -> None:
    root = _content(tmp_path / "c", "https://github.com/org/repo/blob/main/a/b/c.py\n")
    locality.check(root, oracles=locality.Oracles())


def test_a_windows_path_is_caught(tmp_path: Path) -> None:
    r"""The POSIX alternation alone misses `C:\Users\bob\run` entirely."""
    root = _content(tmp_path / "c", "output went to C:\\Users\\bob\\run\n")
    with pytest.raises(Malformed):
        locality.check(root, oracles=locality.Oracles())


def test_a_declared_image_prefix_is_portable(tmp_path: Path) -> None:
    """The kind's `dependencies` name the container image, and a path inside a
    declared image is portable by construction."""
    root = _content(tmp_path / "c", "the toolchain is at /rocm-7.0/bin/hipcc\n")
    with pytest.raises(Malformed):
        locality.check(root, oracles=locality.Oracles())
    locality.check(root, oracles=locality.Oracles(image_prefixes=("/rocm-7.0/",)))


def test_a_binary_file_is_skipped_not_guessed_at(tmp_path: Path) -> None:
    """Three false negatives are known and stated rather than papered over:
    compression, runtime concatenation, and Nix's own caveat that a clean scan
    asserts nothing was found, not that there is nothing to find."""
    root = tmp_path / "c"
    root.mkdir()
    (root / "blob.bin").write_bytes(b"\x00\xff/home/someone/run3\x00")
    locality.check(root, oracles=locality.Oracles())


def test_an_unusable_oracle_root_is_refused_not_ignored(tmp_path: Path) -> None:
    """conda-build raises when a declared prefix entry turns out to match
    nothing, so a stale exclusion fails closed rather than widening the blind
    spot silently."""
    with pytest.raises(Malformed, match="filesystem root"):
        locality.Oracles(store_root=Path("/")).prefixes()


def test_the_patterns_are_data_and_do_not_trigger_themselves(tmp_path: Path) -> None:
    """Self-application, in the form the claim actually takes.

    The allow-list is **data** — a tuple, never a literal spliced into a
    scanning expression — so an artefact that quotes it is not flagged by it.
    conda splits its own placeholder across two source strings for the same
    reason, and bandit `# nosec`s its own default list twice.

    Scanning the whole module is *not* the claim: its docstring carries a
    genuine example of a local path, and flagging that is the checker working.
    """
    assert isinstance(locality.ALLOWED_PREFIXES, tuple)
    quoted = "\n".join(f"ALLOWED: {p}" for p in locality.ALLOWED_PREFIXES)
    locality.check(_content(tmp_path / "c", quoted), oracles=locality.Oracles())

    example = Path(locality.__file__).read_text(encoding="utf-8")
    with pytest.raises(Malformed):
        locality.check(_content(tmp_path / "d", example), oracles=locality.Oracles())
