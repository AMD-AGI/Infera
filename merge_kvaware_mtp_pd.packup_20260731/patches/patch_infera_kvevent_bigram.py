#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Teach infera's kv-event client to read SGLang's bigram token view (MTP).

WHAT: with EAGLE/MTP on, SGLang keys its radix tree on bigrams
(``RadixKey.is_bigram``, set from ``is_eagle``), and the kv-event emitter reports
a block's tokens as the overlapping pairs ``(t[i], t[i+1])`` instead of bare ints:

    sglang/srt/mem_cache/events.py
        is_bigram = node.key.is_bigram
        if is_bigram:
            page_tokens = [(raw[j], raw[j + 1]) for j in range(start, end)]

infera's router hashes those pairs as-is, so the cache view it builds can never
be matched by a query (which chunks the flat token slice). KV-aware routing then
degrades to "every worker scores zero" with nothing in any log — the same silent
failure as having no kv-events at all.

FIX (mirrors AMD-AGI/Infera#56): take the first element of each pair, which
rebuilds ``t[start:end]`` — the flat slice ``hash_request`` chunks on the query
side. Radix nodes split on page boundaries, so the two chunkings stay aligned.
The msgspec schema is widened to ``list[int | tuple[int, int]]`` so the pairs
decode at all.

Not applied to the Rust router (``--router-backend rust``): infera defaults to
the python backend and every run here uses it.

Self-locating and idempotent. All edits or none: a half-patched client decodes
pairs and then hashes them, which is the bug it is meant to fix.
"""

import importlib.util
import sys
from pathlib import Path

_TAG = "[kvevent-bigram]"

_HELPER = '''

def _flat_tokens(token_ids: list) -> list[int]:
    """The flat token ids of a stored block, whatever view the engine reports.

    Under EAGLE/MTP, SGLang keys its radix tree on bigrams, so a block's tokens
    arrive as the overlapping pairs ``(t[i], t[i+1])``. The first element of each
    pair rebuilds ``t[start:end]`` -- the same flat slice ``hash_request`` chunks
    on the query side, and radix nodes split on page boundaries so the two
    chunkings stay aligned. Hashing the pairs as-is builds a view that no request
    can ever match.
    """
    if token_ids and isinstance(token_ids[0], (list, tuple)):
        return [pair[0] for pair in token_ids]
    return token_ids

'''

# Path under the infera package -> [(anchor, replacement)]. Each anchor must
# occur exactly once; the replacement doubles as the already-applied marker.
_EDITS: dict[str, list[tuple[str, str]]] = {
    "router/kv_event/client.py": [
        (
            '''    head, _, port = endpoint.rpartition(":")
    return f"{head}:{int(port) + rank}"
''',
            '''    head, _, port = endpoint.rpartition(":")
    return f"{head}:{int(port) + rank}"
'''
            + _HELPER,
        ),
        (
            """        bs = sub.block_size
        n = len(ev.token_ids) // bs
        for i in range(n):
            chunk = ev.token_ids[i * bs : (i + 1) * bs]
""",
            """        bs = sub.block_size
        tokens = _flat_tokens(ev.token_ids)
        n = len(tokens) // bs
        for i in range(n):
            chunk = tokens[i * bs : (i + 1) * bs]
""",
        ),
    ],
    "router/kv_event/events.py": [
        (
            """class SglangBlockStored(_SglangKVCacheEvent, tag="BlockStored"):
    block_hashes: list[int]
    parent_block_hash: int | None
    token_ids: list[int]
""",
            """class SglangBlockStored(_SglangKVCacheEvent, tag="BlockStored"):
    block_hashes: list[int]
    parent_block_hash: int | None
    # With EAGLE/MTP the radix key is a bigram view (``RadixKey.is_bigram``, set
    # from ``is_eagle``), and the engine reports a block's tokens as the
    # overlapping pairs ``(t[i], t[i+1])`` instead of bare ints -- so this field
    # is list[int] on a plain engine and list[tuple[int, int]] under MTP. See
    # ``client._flat_tokens`` for how the pairs map back onto flat tokens.
    token_ids: list[int | tuple[int, int]]
""",
        ),
    ],
}


def _infera_dirs() -> list[Path]:
    """Every infera package copy on this machine that could be imported.

    The engine image carries TWO: the pip-installed one in site-packages and the
    source tree at the image's WORKDIR (``/opt/infera/infera``), which shadows it
    whenever a process starts with that cwd -- which is exactly what
    ``docker exec`` does. Patching only the one ``find_spec`` happens to return
    leaves the running router on unpatched code, so patch them all.
    """
    roots: list[Path] = []
    seen: set[Path] = set()

    def _add(d: Path) -> None:
        d = d.resolve()
        if d.is_dir() and (d / "router" / "kv_event" / "client.py").is_file() and d not in seen:
            seen.add(d)
            roots.append(d)

    spec = importlib.util.find_spec("infera")
    if spec and spec.origin:
        _add(Path(spec.origin).parent)
    for extra in ("/opt/infera/infera",):
        _add(Path(extra))
    return roots


def _patch_root(root: Path) -> int:
    edited: list[tuple[Path, str]] = []
    for rel, edits in _EDITS.items():
        f = root / rel
        if not f.is_file():
            print(f"{_TAG} {f} is missing — infera layout changed, re-anchor the patch")
            return 1
        src = out = f.read_text()
        for old, new in edits:
            if new in out:
                continue  # already applied
            found = out.count(old)
            if found != 1:
                where = "absent" if found == 0 else f"{found}x ambiguous"
                print(f"{_TAG} anchor {where} in {rel}: {old.splitlines()[0]!r}")
                print(f"{_TAG} infera drifted — re-cut the patch, nothing written")
                return 1
            out = out.replace(old, new, 1)
        if out != src:
            edited.append((f, out))

    if not edited:
        print(f"{_TAG} {root}: already present — skipping")
        return 0
    for f, out in edited:
        f.write_text(out)
        print(f"{_TAG} patched {f}")
    return 0


def main() -> int:
    roots = _infera_dirs()
    if not roots:
        print(f"{_TAG} no infera package found — cannot patch")
        return 1
    for root in roots:
        rc = _patch_root(root)
        if rc != 0:
            return rc
    print(f"{_TAG} kv-event client now flattens the MTP bigram token view "
          f"({len(roots)} copy/copies)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
