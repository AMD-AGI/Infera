###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The refinement-step schedule of one request.

A masked-diffusion model starts from a canvas of ``gen_len`` mask tokens after
the prompt and unmasks it over ``steps`` forwards, ``block_len`` positions at a
time left to right (semi-autoregressive remasking). What each forward costs
depends on what the decoder caches:

``none``
    LLaDA as released: every step recomputes the whole canvas, prompt included.
``prefix``
    Fast-dLLM's prefix cache: the first step of a block refreshes everything
    and caches the prompt and the finished blocks; the remaining steps of the
    block recompute only the current and later blocks.
``dual``
    Fast-dLLM's dual cache: as ``prefix``, but the later (still masked) blocks
    are cached too, so a step recomputes only the current block.
``block``
    Block diffusion (BD3-LM, SDAR): attention is causal across blocks, so the
    cache is exact. The prompt is prefilled once and every step recomputes the
    current block against the cache.

The cached modes are approximations the decoder accepts for speed -- their
quality against ``none`` is the decoder's to validate, not the simulator's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .model import SeqShape

CACHE_MODES = ("none", "prefix", "dual", "block")


@dataclass(frozen=True)
class Step:
    shape: SeqShape
    # The step finishes block ``block`` (0-based); ``-1`` otherwise.
    finishes_block: int = -1


@dataclass
class DLLMWorkload:
    prompt_len: int = 512
    gen_len: int = 512
    # Positions unmasked left to right per block; 0 = the whole canvas at once.
    block_len: int = 0
    # Refinement steps over the whole canvas; 0 = one per generated token.
    steps: int = 0
    cache: str = "none"
    # ``all``: the head predicts every recomputed position, as the released
    # samplers do. ``active``: only the still-masked positions of the current
    # block, which a sampler that slices before the head can do.
    logits: str = "all"
    # Classifier-free guidance runs an unconditional copy of every sequence.
    cfg: bool = False

    def __post_init__(self) -> None:
        if self.cache not in CACHE_MODES:
            raise ValueError(f"[dllm-sim] cache must be one of {CACHE_MODES}")
        if self.logits not in ("all", "active"):
            raise ValueError("[dllm-sim] logits must be 'all' or 'active'")
        if self.gen_len <= 0 or self.prompt_len < 0:
            raise ValueError("[dllm-sim] gen_len must be > 0 and prompt_len >= 0")
        if self.block_len <= 0 or self.block_len > self.gen_len:
            self.block_len = self.gen_len
        if self.steps <= 0:
            self.steps = self.gen_len
        if self.steps < self.num_blocks:
            raise ValueError("[dllm-sim] steps must be at least one per block")

    @property
    def num_blocks(self) -> int:
        return math.ceil(self.gen_len / self.block_len)

    @property
    def seqs_per_request(self) -> int:
        return 2 if self.cfg else 1

    @property
    def canvas(self) -> int:
        return self.prompt_len + self.gen_len

    def steps_in_block(self, b: int) -> int:
        """The step budget, spread over blocks as the reference samplers do."""
        base, extra = divmod(self.steps, self.num_blocks)
        return base + (1 if b < extra else 0)

    def schedule(self) -> list[Step]:
        P, G, B = self.prompt_len, self.gen_len, self.block_len
        out: list[Step] = []
        if self.cache == "block" and P > 0:
            out.append(Step(SeqShape(P, P, 0)))
        for b in range(self.num_blocks):
            done = b * B
            blk = min(B, G - done)
            n = self.steps_in_block(b)
            for i in range(n):
                unmasked = blk * i // n
                active = blk - unmasked
                if self.cache == "none" or (self.cache in ("prefix", "dual") and i == 0):
                    q, kv = P + G, P + G
                elif self.cache == "prefix":
                    q, kv = G - done, P + G
                elif self.cache == "dual":
                    q, kv = blk, P + G
                else:
                    q, kv = blk, P + done + blk
                logits = q if self.logits == "all" else active
                out.append(Step(SeqShape(q, kv, logits), b if i == n - 1 else -1))
        return out

    def kv_cached_tokens(self) -> int:
        """Positions a request keeps K and V for between steps."""
        return 0 if self.cache == "none" else self.canvas
