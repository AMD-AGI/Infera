###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import pytest

from infera.kv.hashing import hash_prompt, hash_token_blocks
from infera.kv.types import MmRun


def test_empty_tokens_returns_empty_chain() -> None:
    assert hash_token_blocks([], block_size=4) == []


def test_block_size_invalid() -> None:
    with pytest.raises(ValueError):
        hash_token_blocks([1, 2, 3], block_size=0)
    with pytest.raises(ValueError):
        hash_token_blocks([1, 2, 3], block_size=-1)


def test_full_blocks_only() -> None:
    # 7 tokens, block_size=4 → 1 full block, 3 trailing dropped.
    chain = hash_token_blocks([1, 2, 3, 4, 5, 6, 7], block_size=4)
    assert len(chain) == 1


def test_chain_length_for_aligned_input() -> None:
    chain = hash_token_blocks(list(range(16)), block_size=4)
    assert len(chain) == 4


def test_determinism_same_input_same_chain() -> None:
    tokens = list(range(20))
    c1 = hash_token_blocks(tokens, block_size=4)
    c2 = hash_token_blocks(tokens, block_size=4)
    assert [b.sequence_hash for b in c1] == [b.sequence_hash for b in c2]
    assert [b.block_hash for b in c1] == [b.block_hash for b in c2]


def test_sensitivity_changing_a_token_cascades() -> None:
    base = hash_token_blocks(list(range(16)), block_size=4)
    tokens = list(range(16))
    tokens[5] = 9999  # in second block (positions 4..8)
    changed = hash_token_blocks(tokens, block_size=4)

    # Block 0 unchanged.
    assert base[0].sequence_hash == changed[0].sequence_hash
    assert base[0].block_hash == changed[0].block_hash

    # Block 1 (containing position 5) differs.
    assert base[1].block_hash != changed[1].block_hash
    assert base[1].sequence_hash != changed[1].sequence_hash

    # Downstream blocks differ via chain.
    assert base[2].sequence_hash != changed[2].sequence_hash
    assert base[3].sequence_hash != changed[3].sequence_hash


def test_chain_head_has_no_parent() -> None:
    chain = hash_token_blocks(list(range(8)), block_size=4)
    assert chain[0].parent_sequence_hash is None
    # First sequence_hash == first block_hash by definition.
    assert chain[0].sequence_hash == chain[0].block_hash


def test_chain_links_use_previous_sequence_hash() -> None:
    chain = hash_token_blocks(list(range(8)), block_size=4)
    assert len(chain) == 2
    assert chain[1].parent_sequence_hash == chain[0].sequence_hash


def test_lora_seed_changes_chain() -> None:
    base = hash_token_blocks(list(range(8)), block_size=4)
    with_lora = hash_token_blocks(list(range(8)), block_size=4, lora_name="my-adapter")
    other_lora = hash_token_blocks(list(range(8)), block_size=4, lora_name="different")

    assert base[0].block_hash != with_lora[0].block_hash
    assert with_lora[0].block_hash != other_lora[0].block_hash


def test_lora_none_equivalent_to_no_lora() -> None:
    c1 = hash_token_blocks(list(range(8)), block_size=4)
    c2 = hash_token_blocks(list(range(8)), block_size=4, lora_name=None)
    c3 = hash_token_blocks(list(range(8)), block_size=4, lora_name="")
    assert c1[0].block_hash == c2[0].block_hash == c3[0].block_hash


def test_mm_block_differs_from_text_only() -> None:
    """A block that contains MM placeholders uses the 13-byte frame, so its
    hash differs from the pure-text path even if it overlaps the same tokens."""
    tokens = list(range(16))  # 4 blocks of size 4
    # Insert one MM run covering tokens [4, 8).
    runs = (MmRun(mm_hash=0xCAFEBABEDEADBEEF, start=4, end=8),)
    base = hash_token_blocks(tokens, block_size=4)
    with_mm = hash_token_blocks(tokens, block_size=4, mm_runs=runs)

    # Block 0 (no MM overlap): unchanged.
    assert base[0].block_hash == with_mm[0].block_hash

    # Block 1 (fully covered by MM run): different (used 13-byte frame).
    assert base[1].block_hash != with_mm[1].block_hash

    # Downstream blocks differ via chain even though they're pure text.
    assert base[2].sequence_hash != with_mm[2].sequence_hash


def test_mm_offset_relative_encoding_position_invariant() -> None:
    """The same MM run starting at different prompt positions produces the
    same block_hashes for slots at matching offset-in-run positions.

    Construct two prompts:
      P1: [text(4 tokens)] [MM run at [4, 20)] [text]
      P2: [text(8 tokens)] [MM run at [8, 24)] [text]

    Block 1 of P1 (tokens [4, 8)) = run_offset [0..4)
    Block 2 of P2 (tokens [8, 12)) = run_offset [0..4)
    These two blocks should hash to the SAME block_hash because the
    13-byte slot frames are identical.
    """
    block_size = 4

    # P1: text(0..4), MM_RUN(4..20), text(20..24)
    p1_tokens = list(range(24))
    p1_runs = (MmRun(mm_hash=0xDEADBEEF, start=4, end=20),)
    c1 = hash_token_blocks(p1_tokens, block_size=block_size, mm_runs=p1_runs)

    # P2: text(0..8), MM_RUN(8..24)
    p2_tokens = list(range(24))
    p2_runs = (MmRun(mm_hash=0xDEADBEEF, start=8, end=24),)
    c2 = hash_token_blocks(p2_tokens, block_size=block_size, mm_runs=p2_runs)

    # P1 block 1 (covering tokens [4, 8)) is the first block of the MM run.
    # P2 block 2 (covering tokens [8, 12)) is also the first block of the MM run.
    # Both encode run_offset [0..4); both have the same mm_hash.
    # Their per-block bytes are byte-identical → same block_hash.
    assert c1[1].block_hash == c2[2].block_hash


def test_mm_run_validates_bounds() -> None:
    with pytest.raises(ValueError, match="exceeds tokens"):
        hash_token_blocks(
            list(range(16)),
            block_size=4,
            mm_runs=(MmRun(mm_hash=1, start=0, end=20),),
        )


def test_mm_run_partial_overlap_of_block() -> None:
    """A block partially overlapped by an MM run is still encoded with
    13-byte frames (every slot in the block), some carrying the placeholder
    tag and some the token tag."""
    tokens = list(range(16))
    # MM run covers tokens [2, 6), overlapping block 0 (0..4) on slots 2..4
    # and block 1 (4..8) on slots 0..2.
    runs = (MmRun(mm_hash=0x1234, start=2, end=6),)
    chain = hash_token_blocks(tokens, block_size=4, mm_runs=runs)
    pure = hash_token_blocks(tokens, block_size=4)

    # Both block 0 and block 1 differ from the pure-text version.
    assert chain[0].block_hash != pure[0].block_hash
    assert chain[1].block_hash != pure[1].block_hash


def test_different_mm_hashes_produce_different_block_hashes() -> None:
    tokens = list(range(8))
    c1 = hash_token_blocks(tokens, block_size=4, mm_runs=(MmRun(mm_hash=0xAAAA, start=0, end=4),))
    c2 = hash_token_blocks(tokens, block_size=4, mm_runs=(MmRun(mm_hash=0xBBBB, start=0, end=4),))
    assert c1[0].block_hash != c2[0].block_hash


def test_hash_prompt_diagnostics() -> None:
    hp = hash_prompt(list(range(11)), block_size=4)
    assert hp.n_tokens == 11
    assert hp.n_full_blocks == 2  # 11 // 4
    assert hp.n_trailing_tokens == 3
    assert hp.n_mm_runs == 0
    assert hp.covered_token_count == 8
    assert len(hp.chain) == 2


def test_hash_prompt_with_mm() -> None:
    hp = hash_prompt(
        list(range(8)),
        block_size=4,
        mm_runs=(MmRun(mm_hash=1, start=0, end=4),),
    )
    assert hp.n_mm_runs == 1
    assert hp.n_full_blocks == 2


def test_block_hashes_are_u64() -> None:
    chain = hash_token_blocks(list(range(8)), block_size=4)
    for b in chain:
        assert 0 <= b.block_hash < 2**64
        assert 0 <= b.sequence_hash < 2**64
