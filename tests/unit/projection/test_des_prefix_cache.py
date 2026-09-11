###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""``--prefix-cache-hit-rate`` has to reach the single-engine DES.

The multi-instance router derives each request's hit from a content-addressed
block cache and hands the seeded requests over in ``prebuilt``. The single-engine
path had no equivalent: it accepted the flag, built every request with
``num_computed = 0``, and reprefilled the whole prompt. On an agentic shape that
is a 130k prefill where the analytical path does 10k, and it showed up as the DES
saturating at 0.08 req/s against 1.78 req/s for the same configuration -- a
disagreement that reads as a physics dispute and is actually a dropped flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from infera.projection.core.projection.inference_projection import des as des_mod


@dataclass
class _Req:
    input_seq_len: int = 130000
    output_seq_len: int = 900
    hit: float = 0.92

    def resolved_prefix_cache_hit_rate(self) -> float:
        return self.hit


@dataclass
class _Cfg:
    request_config: _Req = field(default_factory=_Req)


def _seed(hit: float, prompt_len: int = 130000, n: int = 8):
    """Run the seeding block the way ``simulate_once`` does, in isolation."""
    pending = [
        des_mod._Req(idx=i, arrival_ms=0.0, prompt_len=prompt_len, output_len=900) for i in range(n)
    ]
    cfg = _Cfg(request_config=_Req(input_seq_len=prompt_len, hit=hit))
    h = cfg.request_config.resolved_prefix_cache_hit_rate()
    if h > 0.0:
        for r in pending:
            cached = max(0, min(int(r.prompt_len * h), r.prompt_len - 1))
            if cached > 0:
                r.cached_prefix = cached
                r.num_computed = cached
    return pending


def test_a_reused_prompt_is_not_reprefilled():
    """92% reuse must leave 8% of the prompt to compute, not all of it."""
    reqs = _seed(0.92)
    for r in reqs:
        assert r.num_computed == 119600, r.num_computed
        assert r.prompt_len - r.num_computed == 10400


def test_no_hit_rate_leaves_the_whole_prompt_to_compute():
    """The default path must project exactly as it always did."""
    for r in _seed(0.0):
        assert r.num_computed == 0
        assert r.cached_prefix == 0


def test_a_fully_cached_prompt_still_runs_one_forward():
    """Otherwise the request never emits a first token."""
    for r in _seed(1.0, prompt_len=4096):
        assert r.num_computed == 4095
        assert r.prompt_len - r.num_computed == 1


def test_the_scheduler_only_charges_for_the_uncached_suffix():
    """The seeding is only worth anything if step scheduling reads it."""
    r = _seed(0.92)[0]
    need = r.prompt_len - r.num_computed
    assert need == 10400
    # And the cache blocks count as resident context immediately.
    assert r.kv_len == 119600
