###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Block-hash chain for prefix-cache-aware routing.

See 03-data-model.md (algorithm) and 12-multimodal.md (multimodal extension).

Algorithm (text-only):
    base_seed = 1337  (xor LoRA name hash if LoRA is in use)
    block_hash[i]    = xxh3_64(seed=base_seed, bytes_of(tokens[i*N : (i+1)*N]))
    sequence_hash[0] = block_hash[0]
    sequence_hash[i] = xxh3_64(seed=base_seed,
                               sequence_hash[i-1].to_bytes(8) +
                               block_hash[i].to_bytes(8))

Multimodal extension (per Dynamo `lib/kv-hashing`):
    For a block that contains at least one MM placeholder, each slot is
    encoded as a 13-byte frame:
        Real token:  [tag=0x00 | u32 LE token_id   | u64 LE 0]
        MM placeholder: [tag=0x01 | u32 LE run_offset | u64 LE mm_hash]
    `run_offset` is the slot's position within its parent MM run, not its
    global position. This makes the same image at any prompt position
    produce identical per-block hashes for matching offset-in-run slots.
"""

from __future__ import annotations

from dataclasses import dataclass

import xxhash

from infera.kv.types import BlockKey, MmRun

# Matches Dynamo's seed so on-the-wire hashes are interoperable if anyone
# ever proxies events between systems. See 01-prior-art.md.
BASE_SEED: int = 1337

# Slot encoding constants.
_FRAME_LEN = 13
_TAG_TOKEN = 0x00
_TAG_MM = 0x01


def _seed_for_lora(lora_name: str | None) -> int:
    if not lora_name:
        return BASE_SEED
    return (BASE_SEED ^ xxhash.xxh3_64_intdigest(lora_name.encode("utf-8"))) & 0xFFFFFFFFFFFFFFFF


def _project_mm_runs_to_block(
    runs: tuple[MmRun, ...],
    block_start: int,
    block_end: int,
) -> list[tuple[MmRun, int, int]]:
    """For one block, list the (run, run_start_in_block, run_end_in_block).

    Empty if the block contains no MM placeholders. Returned spans are
    half-open in block-local indices (0..block_size).
    """
    out: list[tuple[MmRun, int, int]] = []
    for run in runs:
        if run.end <= block_start or run.start >= block_end:
            continue
        lo = max(run.start, block_start) - block_start
        hi = min(run.end, block_end) - block_start
        out.append((run, lo, hi))
    return out


def _block_bytes_text_only(tokens: list[int], lo: int, hi: int) -> bytes:
    """4-byte-per-slot encoding for pure-text blocks (cheap path)."""
    return b"".join(int(t).to_bytes(4, "little", signed=False) for t in tokens[lo:hi])


def _block_bytes_with_mm(
    tokens: list[int],
    block_start: int,
    block_end: int,
    mm_in_block: list[tuple[MmRun, int, int]],
) -> bytes:
    """13-byte-per-slot encoding for blocks that overlap one or more MM runs.

    For each slot in [block_start, block_end), emit either a token-slot
    or a placeholder-slot frame. The placeholder's u32 ``run_offset`` is
    the position within its parent MM run, NOT the global position —
    this is the trick that makes the same image hash the same across
    different prompt positions.
    """
    block_size = block_end - block_start
    # Build a per-slot map: slot index → (mm_hash, run_offset) or None.
    placeholder: list[tuple[int, int] | None] = [None] * block_size
    for run, lo, hi in mm_in_block:
        for slot in range(lo, hi):
            global_pos = block_start + slot
            run_offset = global_pos - run.start
            placeholder[slot] = (run.mm_hash, run_offset)

    buf = bytearray(_FRAME_LEN * block_size)
    for slot in range(block_size):
        off = slot * _FRAME_LEN
        ph = placeholder[slot]
        if ph is None:
            token_id = tokens[block_start + slot]
            buf[off] = _TAG_TOKEN
            buf[off + 1 : off + 5] = int(token_id).to_bytes(4, "little", signed=False)
            # bytes 5..13 already zero
        else:
            mm_hash, run_offset = ph
            buf[off] = _TAG_MM
            buf[off + 1 : off + 5] = int(run_offset).to_bytes(4, "little", signed=False)
            buf[off + 5 : off + 13] = int(mm_hash).to_bytes(8, "little", signed=False)
    return bytes(buf)


def hash_token_blocks(
    tokens: list[int],
    block_size: int,
    *,
    mm_runs: tuple[MmRun, ...] | list[MmRun] = (),
    lora_name: str | None = None,
) -> list[BlockKey]:
    """Compute the block-hash chain for a token sequence.

    Only full blocks are returned (trailing partial block is omitted).
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if not tokens:
        return []

    n_full_blocks = len(tokens) // block_size
    if n_full_blocks == 0:
        return []

    # Validate mm_runs once; cheap.
    runs_t: tuple[MmRun, ...] = tuple(mm_runs)
    for run in runs_t:
        if run.end > len(tokens):
            raise ValueError(f"MmRun.end ({run.end}) exceeds tokens length ({len(tokens)})")
        if run.start < 0:
            raise ValueError(f"MmRun.start ({run.start}) is negative")

    seed = _seed_for_lora(lora_name)
    out: list[BlockKey] = []
    prev_seq: int | None = None

    for i in range(n_full_blocks):
        lo = i * block_size
        hi = lo + block_size
        mm_in_block = _project_mm_runs_to_block(runs_t, lo, hi)

        if mm_in_block:
            block_bytes = _block_bytes_with_mm(tokens, lo, hi, mm_in_block)
        else:
            block_bytes = _block_bytes_text_only(tokens, lo, hi)

        block_hash = xxhash.xxh3_64_intdigest(block_bytes, seed=seed)

        if prev_seq is None:
            seq = block_hash
        else:
            chain_buf = prev_seq.to_bytes(8, "little", signed=False) + block_hash.to_bytes(
                8, "little", signed=False
            )
            seq = xxhash.xxh3_64_intdigest(chain_buf, seed=seed)

        out.append(
            BlockKey(
                sequence_hash=seq,
                block_hash=block_hash,
                parent_sequence_hash=prev_seq,
            )
        )
        prev_seq = seq

    return out


@dataclass(frozen=True)
class HashedPrompt:
    """Result bundle: the chain + a few diagnostics."""

    chain: list[BlockKey]
    block_size: int
    n_tokens: int
    n_full_blocks: int
    n_trailing_tokens: int  # tokens dropped from the last partial block
    n_mm_runs: int

    @property
    def covered_token_count(self) -> int:
        return self.n_full_blocks * self.block_size


def hash_prompt(
    tokens: list[int],
    block_size: int,
    *,
    mm_runs: tuple[MmRun, ...] | list[MmRun] = (),
    lora_name: str | None = None,
) -> HashedPrompt:
    """Convenience: hash + diagnostics in one call, used by the routing policy."""
    chain = hash_token_blocks(tokens, block_size, mm_runs=mm_runs, lora_name=lora_name)
    n_tokens = len(tokens)
    n_full = n_tokens // block_size if block_size > 0 else 0
    return HashedPrompt(
        chain=chain,
        block_size=block_size,
        n_tokens=n_tokens,
        n_full_blocks=n_full,
        n_trailing_tokens=n_tokens - n_full * block_size,
        n_mm_runs=len(tuple(mm_runs)),
    )
