"""Criterion 6, plus the measured facts a future change could silently break.

`test_digest_matches_reference_vectors` is the only defence against the Matrix
failure mode — `canonicaljson` does not implement its own specification, and
fixing it required a **new room version** because non-conforming digests were
already persisted. The vectors are computed by `_reference_digest` below, a
second implementation written from `design.md` §4.2's prose rather than from
`digest.py`, and pinned as literals. A refactor that changes the algorithm
fails loudly instead of re-deriving a new "correct" answer.
"""

from __future__ import annotations

import os
import shutil
import stat
from hashlib import sha256
from pathlib import Path

import pytest

from handoff import canonical, tree_digest
from handoff.errors import Malformed

# --------------------------------------------------------------------------- #
# A second implementation, from the prose. `design.md` O8 asks for one.


def _reference_digest(root: bytes) -> bytes:
    """Written from the prose of §4.2, with `os.listdir` rather than
    `os.scandir` and `os.lstat` rather than `DirEntry.stat`, so that it shares
    no call with the implementation it checks."""
    entries = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode):
            mode, digest = b"120000", sha256(os.readlink(path)).digest()
        elif stat.S_ISDIR(st.st_mode):
            mode, digest = b"040000", _reference_digest(path)
        else:
            mode = b"100755" if st.st_mode & stat.S_IXUSR else b"100644"
            with open(path, "rb") as fh:
                digest = sha256(fh.read()).digest()
        entries.append((name, mode, digest))
    entries.sort(key=lambda e: e[0])
    body = b"".join(m + b" " + n + b"\0" + d for n, m, d in entries)
    return sha256(b"tree " + str(len(body)).encode() + b"\0" + body).digest()


#: Pinned, and produced by `_reference_digest`. A change to either literal
#: after the alpha is a `v2` of `ALGORITHM`, never an edit.
EMPTY_TREE = "6ef19b41225c5369f1c104d45d8d85efa9b057b53b14b4b9b939dd74decc5321"
SMALL_TREE = "6b12dda96b7f09e273429634bd35a940abe2ae6b87a1eb760740fe00704c49bb"


def test_digest_matches_reference_vectors(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert tree_digest(os.fsencode(empty)).hex() == EMPTY_TREE
    assert EMPTY_TREE == sha256(b"tree 0\0").hexdigest(), "the empty tree is checkable by hand"

    one = tmp_path / "one"
    (one / "d").mkdir(parents=True)
    (one / "a.txt").write_bytes(b"alpha\n")
    (one / "d" / "b.txt").write_bytes(b"beta\n")
    os.symlink("a.txt", one / "link")

    got = tree_digest(os.fsencode(one))
    assert got == _reference_digest(os.fsencode(one))
    assert got.hex() == SMALL_TREE


def test_digest_survives_round_trip(kinded_store, tmp_path: Path) -> None:
    """Criterion 6: recomputing over a consumed handoff reproduces its digest,
    after storage and a copy into a playground."""
    from tests.handoff.conftest import make_content

    store, hid = kinded_store
    src = make_content(tmp_path / "produced")
    version = store.put(hid, src, producer=_task_id())

    before = store.get_manifest(hid, version).digest["sha256"]
    playground = tmp_path / "playground" / "in"
    content = store.copy_out(hid, version, playground)

    assert content.root == playground
    assert tree_digest(os.fsencode(playground)).hex() == before


def test_copy_out_raises_on_a_tampered_version(kinded_store, tmp_path: Path) -> None:
    from tests.handoff.conftest import make_content

    store, hid = kinded_store
    version = store.put(hid, make_content(tmp_path / "produced"), producer=_task_id())
    (store.root / str(hid) / f"v{version}" / "content" / "items" / "result").write_text("99\n")

    from handoff.errors import DigestMismatch

    with pytest.raises(DigestMismatch) as exc:
        store.copy_out(hid, version, tmp_path / "out")
    assert "sha256=" in str(exc.value)


def test_sort_is_byte_order_not_str_order(tmp_path: Path) -> None:
    """§4.3. A file named `b"\\x80name"` surrogate-escapes to U+DC80 — a *high*
    codepoint — so `sorted(os.listdir())` puts it after `éname` while byte order
    puts it before. They agree for all valid UTF-8, so nothing else catches it."""
    root = tmp_path / "t"
    root.mkdir()
    weird = os.path.join(os.fsencode(root), b"\x80name")
    with open(weird, "wb") as fh:
        fh.write(b"x")
    (root / "éname").write_bytes(b"y")

    names = os.listdir(os.fsencode(root))
    assert sorted(names) != sorted(os.fsdecode(n) for n in names), (
        "the two orders must differ, or this test is asserting nothing"
    )
    assert tree_digest(os.fsencode(root)) == _reference_digest(os.fsencode(root))


def test_exec_bit_in_digest_full_mode_not(tmp_path: Path) -> None:
    """§4.4. `chmod 0600` does not move the digest; `chmod -x` does.

    Measured elsewhere: under `umask 077`, `cp -r` turns 0644 into 0600 and
    0755 into 0700 — but the executable bit survives all six copy × umask
    combinations. So the whitelist is exactly {x}."""
    root = tmp_path / "t"
    root.mkdir()
    script = root / "run.sh"
    script.write_bytes(b"#!/bin/sh\n")
    script.chmod(0o755)

    before = tree_digest(os.fsencode(root))
    script.chmod(0o700)
    assert tree_digest(os.fsencode(root)) == before, "a permission change other than x moved it"
    script.chmod(0o600)
    assert tree_digest(os.fsencode(root)) != before, "dropping x must move it"


def test_empty_directory_is_recorded(tmp_path: Path) -> None:
    """§4.5. `cp -r`, `cp -a`, `copytree`, tar and zip all preserve one; only
    git cannot represent it. A declared `logs` directory a run left empty is a
    different artefact from one with no `logs` at all."""
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "logs").mkdir(parents=True)
    b.mkdir()
    assert tree_digest(os.fsencode(a)) != tree_digest(os.fsencode(b))


def test_symlink_is_hashed_over_its_target_text(tmp_path: Path) -> None:
    root = tmp_path / "t"
    root.mkdir()
    (root / "real").write_bytes(b"x")
    os.symlink("real", root / "link")
    before = tree_digest(os.fsencode(root))

    (root / "link").unlink()
    os.symlink("other", root / "link")
    assert tree_digest(os.fsencode(root)) != before

    copy = tmp_path / "copy"
    shutil.copytree(root, copy, symlinks=True)
    assert tree_digest(os.fsencode(copy)) == tree_digest(os.fsencode(root))


def test_fifo_raises_rather_than_being_skipped(tmp_path: Path) -> None:
    """Skipping would make two different trees hash the same, which is the one
    failure a digest must not have."""
    root = tmp_path / "t"
    root.mkdir()
    os.mkfifo(root / "pipe")
    with pytest.raises(Malformed, match="not a file, directory or symlink"):
        tree_digest(os.fsencode(root))


def test_content_swapped_between_filenames_differs(tmp_path: Path) -> None:
    """`checksumdir`'s own docstring concedes it hashes "only file contents and
    not filenames", so two trees with contents swapped hash identically. Ours
    must not — the name is in the body."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "one").write_bytes(b"X")
    (a / "two").write_bytes(b"Y")
    (b / "one").write_bytes(b"Y")
    (b / "two").write_bytes(b"X")
    assert tree_digest(os.fsencode(a)) != tree_digest(os.fsencode(b))


# --------------------------------------------------------------------------- #
# The canonical encoder


def test_large_int_refused_not_rounded() -> None:
    """§4.6, and the specific value matters: on `12345678901234567168` three
    libraries do three different things — `rfc8785` raises, `jcs` silently
    rounds to `…567000`, and `canonicaljson` passes it through. A library swap
    could start rounding it, and a rounded value digests correctly."""
    with pytest.raises(Malformed) as exc:
        canonical({"n": 12345678901234567168})
    assert "12345678901234567168" in str(exc.value)
    assert "567000" not in canonical({"n": 9007199254740991}).decode()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.0])
def test_unrepresentable_floats_refused(value: float) -> None:
    with pytest.raises(Malformed):
        canonical({"x": value})


def test_canonical_sorts_and_is_stable() -> None:
    assert canonical({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
    assert canonical({"a": 0.1}) == b'{"a":0.1}'


def test_nfc_and_nfd_stay_two_keys() -> None:
    """JCS §3.1 refuses to normalise Unicode — *"MUST preserve Unicode string
    data 'as is'"* — so two keys that render identically produce different
    digests, with no warning. `design.md` O4 records the consequence rather
    than papering over it."""
    assert canonical({"é": 1}) != canonical({"é": 1})


def _task_id():
    from task_graph.ids import TaskId

    return TaskId.new()
