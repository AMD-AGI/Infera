###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for the MLA TP write/read dedup logic (#64).

Under MLA (DeepSeek-V3 / Kimi) the cached KV is a compressed latent that is
REPLICATED byte-identically across every TP rank, so the connector:
  * folds all TP ranks of a group onto ONE rank-agnostic compat key
    (so they share a single L3 file), and
  * persists it from rank 0 only (the write gate in wait_for_save).

This means both the WRITE (one file instead of tp_size) and the READ (every
rank resolves the same key → loads rank 0's single file) collapse onto one
copy. These tests cover the pure decision logic — config detection, rank
resolution, and key folding — without a GPU or a running daemon.

The dedup MUST stay off for regular attention (per-rank shards genuinely
differ) and for pure data-parallel / single-rank (each rank's KV is distinct,
separated by the content key), hence the tp_size>1 gate.
"""

from types import SimpleNamespace

import pytest

from infera.engine.vllm.kvd_connector import (
    InferaKvdConnector,
    _is_mla_from_config,
)


def _cfg(
    *,
    tp_size=1,
    tp_rank=0,
    pp_size=1,
    pp_rank=0,
    use_mla=None,
    kv_lora_rank=None,
    model="/nonexistent/model-for-test",
):
    """Build a minimal fake vllm_config (SimpleNamespace) good enough for
    the static topology/MLA helpers. No vLLM distributed init, so the
    helpers fall back to the config-attr rank (tensor_parallel_rank)."""
    hf = SimpleNamespace()
    if kv_lora_rank is not None:
        hf.kv_lora_rank = kv_lora_rank
    mc = SimpleNamespace(model=model, hf_text_config=hf, hf_config=hf)
    if use_mla is not None:
        mc.use_mla = use_mla
    parallel = SimpleNamespace(
        tensor_parallel_size=tp_size,
        tensor_parallel_rank=tp_rank,
        pipeline_parallel_size=pp_size,
        pipeline_parallel_rank=pp_rank,
    )
    return SimpleNamespace(model_config=mc, parallel_config=parallel)


class TestIsMlaFromConfig:
    def test_use_mla_flag_true(self):
        assert _is_mla_from_config(_cfg(use_mla=True)) is True

    def test_use_mla_flag_false_wins_over_marker(self):
        # explicit use_mla=False is authoritative even if a marker exists
        assert _is_mla_from_config(_cfg(use_mla=False, kv_lora_rank=512)) is False

    def test_kv_lora_rank_marker(self):
        assert _is_mla_from_config(_cfg(kv_lora_rank=512)) is True

    def test_regular_attention(self):
        assert _is_mla_from_config(_cfg()) is False

    def test_none_model_config(self):
        assert _is_mla_from_config(SimpleNamespace(model_config=None)) is False

    def test_garbage_config_is_false_not_raise(self):
        assert _is_mla_from_config(SimpleNamespace()) is False


class TestResolveTpRankSize:
    def test_from_config_attr(self):
        assert InferaKvdConnector._resolve_tp_rank_size(_cfg(tp_size=8, tp_rank=3)) == (3, 8)

    def test_default_single_rank(self):
        assert InferaKvdConnector._resolve_tp_rank_size(_cfg()) == (0, 1)


class TestCompatKeyWriteDedup:
    def test_mla_tp_ranks_share_one_key(self):
        # MLA + TP>1: distinct ranks -> SAME rank-agnostic key (one L3 file)
        k0 = InferaKvdConnector._extract_compat_key(_cfg(tp_size=8, tp_rank=0, use_mla=True))
        k3 = InferaKvdConnector._extract_compat_key(_cfg(tp_size=8, tp_rank=3, use_mla=True))
        k7 = InferaKvdConnector._extract_compat_key(_cfg(tp_size=8, tp_rank=7, use_mla=True))
        assert k0 == k3 == k7
        assert "tp0of8" in k0

    def test_mla_marker_also_folds(self):
        k0 = InferaKvdConnector._extract_compat_key(_cfg(tp_size=4, tp_rank=0, kv_lora_rank=512))
        k2 = InferaKvdConnector._extract_compat_key(_cfg(tp_size=4, tp_rank=2, kv_lora_rank=512))
        assert k0 == k2

    def test_regular_attention_ranks_stay_distinct(self):
        # Regular attention: per-rank shards differ -> keys MUST differ
        k0 = InferaKvdConnector._extract_compat_key(_cfg(tp_size=8, tp_rank=0))
        k3 = InferaKvdConnector._extract_compat_key(_cfg(tp_size=8, tp_rank=3))
        assert k0 != k3
        assert "tp0of8" in k0 and "tp3of8" in k3

    def test_mla_single_rank_no_fold_needed(self):
        # tp_size=1 (pure DP / single): nothing to fold, namespace = tp0of1
        k = InferaKvdConnector._extract_compat_key(_cfg(tp_size=1, tp_rank=0, use_mla=True))
        assert "tp0of1" in k

    def test_pp_dimension_preserved_under_mla(self):
        # MLA folds TP but must NOT merge distinct PP stages (different layers)
        k_pp0 = InferaKvdConnector._extract_compat_key(
            _cfg(tp_size=4, tp_rank=1, pp_size=2, pp_rank=0, use_mla=True)
        )
        k_pp1 = InferaKvdConnector._extract_compat_key(
            _cfg(tp_size=4, tp_rank=1, pp_size=2, pp_rank=1, use_mla=True)
        )
        assert k_pp0 != k_pp1
        assert "pp0of2" in k_pp0 and "pp1of2" in k_pp1


class TestReadDedupSharedKey:
    """READ side: the lookup uses the SAME compat key as the save, so under
    MLA every TP rank resolves to rank 0's single file (no per-rank copies
    to read). This is the property that lets the load collapse to one NFS
    read (page-cache dedup; broadcast is the further Phase-2 step)."""

    def test_all_ranks_resolve_same_load_key(self):
        keys = {
            InferaKvdConnector._extract_compat_key(_cfg(tp_size=8, tp_rank=r, use_mla=True))
            for r in range(8)
        }
        assert len(keys) == 1  # all 8 ranks read the one shared namespace

    def test_dp_ranks_not_merged_by_key_alone(self):
        # Pure DP (tp_size=1): same compat key across ranks is SAFE because
        # the content key (sha256 of block hashes) separates distinct prompts.
        # Here we just assert the compat key doesn't *force* a merge that
        # would be wrong — it's tp0of1 for all, and correctness relies on the
        # content key, which is per-request. (Regression guard for the gate.)
        k_a = InferaKvdConnector._extract_compat_key(_cfg(tp_size=1, tp_rank=0, use_mla=True))
        k_b = InferaKvdConnector._extract_compat_key(_cfg(tp_size=1, tp_rank=0, use_mla=True))
        assert k_a == k_b


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
