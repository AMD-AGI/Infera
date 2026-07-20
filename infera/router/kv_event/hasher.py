###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Chained XXH3-64 hashing for KV-aware routing.

Both the query side (tokenized prompt) and the event side (KvEventClient
processing ``BlockStored``) feed tokens through the same chain so the
hashes match.
"""

from __future__ import annotations

from xxhash import xxh3_64_intdigest

ROUTER_SEED: int = 0


def hash_chunk(parent: int, token_ids: list[int]) -> int:
    """XXH3-64 over ``parent || token_ids`` (LE uint32 per token)."""
    buf = bytearray(8 + 4 * len(token_ids))
    buf[0:8] = parent.to_bytes(8, "little", signed=False)
    offset = 8
    for t in token_ids:
        buf[offset : offset + 4] = t.to_bytes(4, "little", signed=False)
        offset += 4
    return xxh3_64_intdigest(bytes(buf))


def hash_request(token_ids: list[int], block_size: int) -> list[int]:
    """Chained block hashes for a token sequence; trailing partial block dropped."""
    if block_size <= 0:
        return []
    parent = ROUTER_SEED
    n_full = len(token_ids) // block_size
    out: list[int] = []
    for i in range(n_full):
        chunk = token_ids[i * block_size : (i + 1) * block_size]
        parent = hash_chunk(parent, chunk)
        out.append(parent)
    return out
