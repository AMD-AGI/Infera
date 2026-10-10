###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The prefix cache and the running requests are the same pool.

``_Req.reserved_kv`` does not charge a request for the leading blocks it hit
on, because an earlier request is holding them. That is the right account, and
it only balances if whatever is holding them is charged instead. The holder is
the block cache, and it was being handed the whole pool at the same time as the
running set, so the two structures booked the same memory and neither ever saw
it run out.

Where the pool is roomy against the working set the correction is nothing, and
that is most of the corpus. Where it is not, it is the entire behaviour: on the
AgentX ladders GLM-5.2 on MI325X has a pool that holds 5.5 of these
conversations, and the hardware's reuse falls 0.79 -> 0.30 -> 0.12 across
C=4,5,6 while TTFT goes from 3s to 198s. Double-booked, the replay reported
reuse of 0.96 and a TTFT flat within a factor of two across that whole
collapse, so the ladder came out ranked by nothing in particular.
"""

from __future__ import annotations

from infera.projection.core.projection.inference_projection import des as des_mod

_BLOCK = 64


def test_a_roomy_pool_leaves_the_cache_what_it_was_given():
    """The correction has to be silent where it does not apply."""
    cap = 1000
    got = des_mod._free_cache_blocks(
        kv_cache_tokens=10_000_000, live_tokens=100_000, block_size=_BLOCK, cap_blocks=cap
    )
    assert got == cap


def test_a_full_pool_leaves_the_cache_nothing():
    """No room is no room, however the request was charged for it."""
    got = des_mod._free_cache_blocks(
        kv_cache_tokens=1_048_576, live_tokens=1_500_000, block_size=_BLOCK, cap_blocks=16_384
    )
    assert got == 1


def test_no_room_is_never_read_as_unlimited_room():
    """``_BlockStore`` takes a zero capacity as unbounded.

    Returning the honest zero would hand a pool with nothing left in it the
    one cache that never evicts, which is the failure this is here to fix
    rather than a smaller version of it.
    """
    for live in (1_048_576, 2_000_000, 10**9):
        got = des_mod._free_cache_blocks(
            kv_cache_tokens=1_048_576, live_tokens=live, block_size=_BLOCK, cap_blocks=16_384
        )
        assert got >= 1


def test_the_cache_shrinks_as_the_running_set_grows():
    """Reuse degrades with pressure rather than falling off a single step."""
    pool = 1_048_576
    caps = [
        des_mod._free_cache_blocks(pool, c * 190_000, _BLOCK, 16_384) for c in (1, 2, 3, 4, 5, 6)
    ]
    assert caps == sorted(caps, reverse=True), caps
    assert caps[0] > caps[3] > caps[-1]


def test_an_unmeasured_pool_is_left_alone():
    """Without a pool figure there is nothing to divide, so nothing is claimed."""
    assert des_mod._free_cache_blocks(0, 500_000, _BLOCK, 4096) == 4096
    assert des_mod._free_cache_blocks(1_048_576, 500_000, 0, 4096) == 4096
