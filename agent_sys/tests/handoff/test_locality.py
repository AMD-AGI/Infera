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


def test_an_http_request_target_is_not_a_local_path(tmp_path: Path) -> None:
    """**The check refused a correct handoff, every time, 2026-08-31.**

    `_URL` already decided that an API path is not a filesystem path, and keys
    on the scheme. An HTTP request-line carries the request-target **bare** —
    `"POST /v1/chat/completions HTTP/1.1"` — so the mitigation did not fire and
    `_CANDIDATE` read three segments as an absolute local path.

    What made it more than a nuisance: the task's brief ordered its agent to
    prove the completion had gone through the router rather than the engine's
    own port, and the natural evidence is the router's access log. So **every
    correct kit for that task contained the string** and the seal rejected all
    of them. The run then hung for 65 minutes, for a separate reason.

    This completes `_URL`'s decision rather than making a new one.
    """
    log = _content(
        tmp_path / "c",
        'INFO: 127.0.0.1:56726 - "POST /v1/chat/completions HTTP/1.1" 200 OK\n',
    )
    locality.check(log, oracles=locality.Oracles())


def test_the_request_target_rule_does_not_make_the_check_vacuous(tmp_path: Path) -> None:
    """**The non-vacuity control, and it is why the rule is anchored on a method.**

    An over-broad fix — stop matching absolute paths, or strip anything that
    looks API-shaped — passes the test above and leaves a check that finds
    nothing. `interfaces.md` §8.11g: an instrument pointed only at the safe case
    proves nothing.

    So a genuinely local path on an ordinary line must still be caught, and a
    line that merely *mentions* an access log must not become a blanket
    suppression of everything on it.
    """
    plain = _content(tmp_path / "a", "the file /home/someone/run3/out.json\n")
    with pytest.raises(Malformed):
        locality.check(plain, oracles=locality.Oracles())

    # Only the request-target is suppressed; the rest of the line still scans.
    mixed = _content(
        tmp_path / "b",
        '"GET /v1/workers HTTP/1.1" 200 - wrote /home/someone/run3/workers.json\n',
    )
    with pytest.raises(Malformed) as caught:
        locality.check(mixed, oracles=locality.Oracles())
    assert "/home/someone/run3/workers.json" in str(caught.value)


def test_a_bare_verb_cannot_cloak_a_local_path(tmp_path: Path) -> None:
    """**The suppression is anchored on `HTTP/x.y`, not on the verb.**

    A first version matched `VERB SP /path` alone. An automated review was right
    that it opens a cloak: anyone writing a verb before a path — deliberately or
    by accident — escapes the check entirely. Requiring the version token means
    only a genuine request-line is stripped, and RFC 9112 puts that token on
    every one of them.

    Not anchored to start-of-line, which was the other half of that suggestion:
    the line this was measured on is
    `INFO: 127.0.0.1:56726 - "POST /v1/chat/completions HTTP/1.1" 200 OK`, where
    the request-line sits inside a quoted field mid-record, so `^` would have
    reverted the fix it was meant to protect.
    """
    root = _content(tmp_path / "c", "POST /home/someone/run3\n")
    with pytest.raises(Malformed) as caught:
        locality.check(root, oracles=locality.Oracles())
    assert "/home/someone/run3" in str(caught.value)


def test_a_forged_request_line_cannot_cloak_a_path_we_minted(tmp_path: Path) -> None:
    """**The residual is real, and this pins what bounds it.**

    A local path written as a *complete* request line is still suppressed, and
    no shape rule can tell it from an API path — that is Debian #1002451 in one
    line. What keeps it from mattering is that **the certain half is not
    cloakable**: `_scan_text` runs the oracle loop over the raw line before any
    stripping, so a path this system minted fires regardless of the syntax
    around it.

    Both halves are asserted here, because the first without the second reads as
    a hole and the second without the first hides one.
    """
    minted = locality.Oracles(playground_root=Path("/var/tmp/playground-7"))

    forged = _content(tmp_path / "d", "GET /var/tmp/playground-7/run3/out.json HTTP/1.1\n")
    with pytest.raises(Malformed) as caught:
        locality.check(forged, oracles=minted)
    assert "certain" in str(caught.value)  # the oracle branch fired, not the heuristic

    # And the acknowledged gap, asserted so it cannot close by accident and go
    # unnoticed: not minted by us, forged as a request line, therefore missed.
    escapes = _content(tmp_path / "e", "GET /home/alice/secret/out.json HTTP/1.1\n")
    locality.check(escapes, oracles=minted)
