"""The tree digest, and the canonical encoder `items.json` goes through.

Two functions, one rule each, and both rules are specifications rather than
descriptions — `design.md` §4.2 and §4.6. A second implementation in another
language must be able to reproduce them from the docstrings.

Nothing here imports anything of this package except `errors`.
"""

from __future__ import annotations

import math
import os
import stat
from hashlib import sha256
from typing import Any

import rfc8785

from handoff.errors import Malformed

__all__ = ["ALGORITHM", "canonical", "tree_digest"]

#: What §4.2's walk is registered as. A change to the walk is a `v2`, never a
#: silent redefinition — `design.md` §4.7, and DVC's md5 migration is the cost
#: of not having done this.
ALGORITHM = "agent_sys.handoff.tree.v1"

#: RFC 7493 §2.2's range, which RFC 8785 Appendix D adopts. `rfc8785` enforces
#: it too; the constant is here because the `-0.0` walk needs the same bound in
#: one place and a reader should not have to infer it from a library.
_MAX_SAFE_INT = 2**53 - 1

_CHUNK = 1 << 20


def _file_digest(path: bytes) -> bytes:
    h = sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.digest()


def tree_digest(root: bytes) -> bytes:
    """sha256 over a subtree, git-shaped, specified exactly.

    ```
    file   := sha256(contents)
    link   := sha256(os.readlink(path))        # the target text, never followed
    dir    := sha256(b"tree " + len(body) + b"\\0" + body)
               body = concat over entries sorted by name_bytes of:
                      mode + b" " + name_bytes + b"\\0" + digest32
    mode   in {100644, 100755, 120000, 040000}
    ```

    Three properties that are not incidental:

    **Bytes throughout.** `os.scandir` is given `bytes` and yields `bytes`
    names; `pathlib.Path` rejects bytes paths outright, and mixing the two is
    how a name comparison silently becomes `False`.

    **Sorted by `os.fsencode(name)` in plain byte order**, not by the `str`.
    A file named `b"\\x80name"` surrogate-escapes to U+DC80 — a *high*
    codepoint — so `sorted(os.listdir())` places it after `éname` where byte
    order places it before. The two agree for all valid UTF-8, which is why the
    bug would survive every ordinary test.

    **A device node, FIFO or socket raises rather than being skipped**, because
    skipping makes two different trees hash the same. Empty directories are
    recorded; only git cannot represent one, and we owe git nothing here.
    """
    entries: list[tuple[bytes, bytes, bytes]] = []
    with os.scandir(root) as it:
        for e in it:
            st = e.stat(follow_symlinks=False)
            if stat.S_ISLNK(st.st_mode):
                mode, d = b"120000", sha256(os.readlink(e.path)).digest()
            elif stat.S_ISDIR(st.st_mode):
                mode, d = b"040000", tree_digest(e.path)
            elif stat.S_ISREG(st.st_mode):
                mode = b"100755" if st.st_mode & stat.S_IXUSR else b"100644"
                d = _file_digest(e.path)
            else:
                raise Malformed(
                    f"{os.fsdecode(e.path)}: not a file, directory or symlink "
                    f"(mode {stat.filemode(st.st_mode)})"
                )
            entries.append((e.name, mode, d))

    entries.sort()  # byte order, by construction: e.name is bytes
    body = b"".join(m + b" " + n + b"\0" + d for n, m, d in entries)
    # The length prefix is git's and it is load-bearing: without it two
    # different entry lists can concatenate to the same body.
    return sha256(b"tree " + str(len(body)).encode() + b"\0" + body).digest()


def _reject_unrepresentable(value: Any, path: str = "$") -> None:
    """Walk for the three values RFC 8785 either cannot carry or carries wrongly.

    `rfc8785` raises on a large int and on NaN/±Inf, and this walk would not be
    needed for those alone. It exists for **`-0.0`**, which `rfc8785` emits as
    `0` — measured. jsonnet renders it `-0`, `json.loads` returns `-0.0`, and
    `-0.0 == 0.0` is true while the serialisations differ, so a value that came
    from two places would digest the same and print differently.
    """
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        if abs(value) > _MAX_SAFE_INT:
            raise Malformed(
                f"{path}: {value} is outside the safe integer range "
                f"±{_MAX_SAFE_INT} (RFC 8785 Appendix D on RFC 7493 §2.2). "
                f"Wrap it as a JSON string — it is not rounded here, because a "
                f"rounded value digests correctly and is the wrong value"
            )
        return
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise Malformed(f"{path}: {value} is not representable in JCS (RFC 8785 §3.2.2.3)")
        if value == 0.0 and math.copysign(1.0, value) < 0:
            raise Malformed(
                f"{path}: -0.0 is rejected. JCS serialises it as `0`, so it "
                f"would digest identically to 0.0 while printing differently"
            )
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise Malformed(f"{path}: object key {k!r} is not a string")
            _reject_unrepresentable(v, f"{path}.{k}")
        return
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _reject_unrepresentable(v, f"{path}[{i}]")


def canonical(value: Any) -> bytes:
    """RFC 8785 (JCS) bytes for a parsed JSON value. **Refuses, never coerces.**

    The values in `items.json` come from a jsonnet-rendered kind default and
    from an agent at runtime, and must digest identically either way — so what
    is serialised is the *parsed* value, never rendered text. Measured:
    `_jsonnet` 0.22.0 emits `0.10000000000000001` where `rjsonnet` 0.5.6 emits
    `0.1`, and they agree only after `json.loads`.

    On one value our own toolchain produces, `12345678901234567168`, three
    libraries do three different things: `rfc8785` raises, `jcs` silently
    rounds, and `canonicaljson` passes it through. Silent rounding yields a
    correct-looking digest for the wrong value, so this raises `Malformed`.

    `json.dumps(sort_keys=True)` is **not** a canonicalisation and is not used:
    RFC 8785 sorts by UTF-16 code units (§3.2.3), and Python's `sorted()` does
    not for non-BMP keys. JCS refuses to normalise Unicode (§3.1), so NFC and
    NFD remain two distinct keys — `design.md` O4.
    """
    _reject_unrepresentable(value)
    try:
        return rfc8785.dumps(value)
    except rfc8785.CanonicalizationError as exc:  # pragma: no cover - walk covers these
        raise Malformed(f"not canonicalisable: {exc}") from exc
