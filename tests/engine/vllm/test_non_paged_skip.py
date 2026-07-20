###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for the non-paged (Mamba / linear-attention / conv) KV-cache
group skip in the kvd L3 connector (#60 Part 2).

Hybrid models (e.g. Mamba/SSM + attention) expose several kv_cache_groups.
The recurrent/conv groups hold an SSM/conv STATE, not a paged attention KV
cache — there is no block-table prefix to reuse, so they must never be
offloaded to L3. The danger is shape-based detection alone: a Mamba state
tensor is often 3-D, which would fall into ``register_kv_caches``'s
``len(shape) == 3 -> MLA`` branch and be mis-registered as an MLA latent
cache (num_kv_channels=1). The connector therefore consults
``group.kv_cache_spec`` and skips non-paged groups BEFORE the shape probe,
on both the worker (``register_kv_caches``) and scheduler
(``_bootstrap_group_kv_spec_from_config``) sides.

These tests cover the decision logic and both registration paths without a
GPU or a running daemon.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infera.engine.vllm.kvd_connector import (
    _KVD_STAT_KEYS,
    _L3_LOG_EVERY,
    InferaKvdConnector,
    _expected_plain_mla_hidden,
    _is_non_paged_kv_spec,
)


class TestL3ActivityLog:
    """The periodic `kvd L3 activity` INFO summary (via _stat_inc) is the ONLY
    signal that GPU-direct/file-tier offload is working: vLLM's stats logger is
    muted by --disable-log-stats and the daemon statctl is all-zero under
    file-tier. Test the fire logic without depending on logger propagation."""

    @staticmethod
    def _conn():
        import threading

        c = InferaKvdConnector.__new__(InferaKvdConnector)
        c._stat_lock = threading.Lock()
        c._stat_counters = {k: 0 for k in _KVD_STAT_KEYS}
        c._l3_cum = {k: 0 for k in _KVD_STAT_KEYS}
        c._l3_log_at = {"saved_chunks": 1, "lookup_requests": 1}
        c._l3_log_enabled = True
        c._role = "worker"
        return c

    def test_first_save_fires_then_period(self):
        c = self._conn()
        c._stat_inc("saved_chunks", 1)  # first save → fires immediately
        assert c._l3_cum["saved_chunks"] == 1
        assert c._l3_log_at["saved_chunks"] == 1 + _L3_LOG_EVERY  # next scheduled
        # a few more saves below the next threshold do NOT re-schedule
        c._stat_inc("saved_chunks", 10)
        assert c._l3_log_at["saved_chunks"] == 1 + _L3_LOG_EVERY

    def test_disabled_never_fires(self):
        c = self._conn()
        c._l3_log_enabled = False
        c._stat_inc("saved_chunks", 5)
        assert c._l3_log_at["saved_chunks"] == 1  # threshold untouched

    def test_unknown_key_is_noop(self):
        c = self._conn()
        c._stat_inc("not_a_metric", 3)  # typo-guard: no KeyError, no accounting
        assert all(v == 0 for v in c._l3_cum.values())


class TestExpectedPlainMlaHidden:
    """`_expected_plain_mla_hidden` = kv_lora_rank + qk_rope_head_dim, the
    signature register_kv_caches uses to AUTO-ALLOW a plain fp8 MLA latent
    (no interleaved scale) without the INFERA_KVD_ALLOW_PACKED_KV toggle."""

    @staticmethod
    def _cfg(hf):
        return SimpleNamespace(model_config=SimpleNamespace(hf_text_config=hf, hf_config=hf))

    def test_kimi_style_mla_sum(self):
        # Kimi-K2.6: 512 + 64 = 576 (the validated plain-fp8 latent width).
        hf = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)
        assert _expected_plain_mla_hidden(self._cfg(hf)) == 576

    def test_nested_text_config(self):
        # Multimodal wrapper (Kimi K2.6 has vision) — dims live under text_config.
        inner = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)
        outer = SimpleNamespace(text_config=inner)
        assert _expected_plain_mla_hidden(self._cfg(outer)) == 576

    def test_non_mla_returns_none(self):
        # Regular attention (no kv_lora_rank) → None → guard stays conservative.
        hf = SimpleNamespace(num_attention_heads=64)
        assert _expected_plain_mla_hidden(self._cfg(hf)) is None

    def test_missing_rope_returns_none(self):
        hf = SimpleNamespace(kv_lora_rank=512)
        assert _expected_plain_mla_hidden(self._cfg(hf)) is None

    def test_no_model_config_returns_none(self):
        assert _expected_plain_mla_hidden(SimpleNamespace()) is None


# --- stub KVCacheSpec classes ------------------------------------------------
# _is_non_paged_kv_spec keys off (1) attention attrs, (2) isinstance MambaSpec
# (real vLLM, may be absent in the unit env), (3) class-name substring. These
# stubs exercise (1) and (3); the names deliberately contain the hint tokens.


class FakeFullAttentionSpec:
    def __init__(self, block_size=16, num_kv_heads=8, head_size=128, num_blocks=100):
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.num_blocks = num_blocks
        self.page_size_bytes = 2 * num_kv_heads * head_size * block_size * 2


class FakeMLAAttentionSpec:
    # MLA is still ATTENTION: it advertises num_kv_heads + head_size, so it
    # must NOT be skipped (num_kv_channels=1 handling stays intact).
    def __init__(self, block_size=16, head_size=576, num_blocks=100):
        self.block_size = block_size
        self.num_kv_heads = 1
        self.head_size = head_size
        self.num_blocks = num_blocks
        self.page_size_bytes = head_size * block_size * 2


class FakeMambaSpec:
    # Recurrent state: NO num_kv_heads/head_size; class name carries "mamba".
    def __init__(self, block_size=16):
        self.block_size = block_size


class FakeShortConvSpec:
    def __init__(self, block_size=16):
        self.block_size = block_size


def _group(layer_names, spec):
    return SimpleNamespace(layer_names=list(layer_names), kv_cache_spec=spec)


def _kv_cfg(groups, block_size=16, num_blocks=100):
    return SimpleNamespace(
        kv_cache_groups=list(groups),
        block_size=block_size,
        num_blocks=num_blocks,
    )


class TestIsNonPagedKvSpec:
    def test_full_attention_is_paged(self):
        assert _is_non_paged_kv_spec(FakeFullAttentionSpec()) is False

    def test_mla_attention_is_paged(self):
        # MLA must be treated as paged attention (we offload its latent).
        assert _is_non_paged_kv_spec(FakeMLAAttentionSpec()) is False

    def test_mamba_spec_by_name(self):
        assert _is_non_paged_kv_spec(FakeMambaSpec()) is True

    def test_short_conv_spec_by_name(self):
        assert _is_non_paged_kv_spec(FakeShortConvSpec()) is True

    def test_none_spec_not_skipped(self):
        # None spec (older vLLM / stub configs) -> preserve legacy path.
        assert _is_non_paged_kv_spec(None) is False

    def test_attention_attrs_win_over_name(self):
        # Even a confusingly named spec is kept if it exposes attention attrs.
        class MambaLikeAttentionSpec:
            block_size = 16
            num_kv_heads = 8
            head_size = 128

        assert _is_non_paged_kv_spec(MambaLikeAttentionSpec()) is False

    def test_unknown_attentionless_spec_not_skipped(self):
        # A spec with neither attention attrs nor a recurrent name is left to
        # the legacy shape probe (conservative: don't skip what we don't know).
        class WeirdSpec:
            block_size = 16

        assert _is_non_paged_kv_spec(WeirdSpec()) is False


def _bare_connector():
    """A connector instance with __init__ bypassed and only the attributes
    the registration paths touch. _autosize_chunk_tokens is stubbed to a
    no-op so we don't need a fully-initialized connector."""
    conn = InferaKvdConnector.__new__(InferaKvdConnector)
    conn._kv_caches = {}
    conn._layer_to_group = {}
    conn._group_kv_spec = {}
    conn._has_non_paged_groups = False
    conn._warned_hybrid_no_load = False
    conn._plain_mla_hidden = None
    conn._autosize_chunk_tokens = lambda: None
    return conn


class TestRegisterKvCachesSkip:
    """Worker side: register_kv_caches must skip Mamba groups and still
    register the attention group correctly."""

    def _conn(self, kv_cfg):
        conn = _bare_connector()
        conn._kv_cache_config = kv_cfg
        return conn

    def test_hybrid_skips_mamba_keeps_attention(self):
        torch = pytest.importorskip("torch")
        attn_t = torch.zeros((2, 100, 16, 8, 128), dtype=torch.bfloat16)
        mamba_t = torch.zeros((100, 16, 2048), dtype=torch.bfloat16)  # 3-D!
        kv_cfg = _kv_cfg(
            [
                _group(["a0", "a1"], FakeFullAttentionSpec()),
                _group(["m0"], FakeMambaSpec()),
            ]
        )
        conn = self._conn(kv_cfg)
        conn.register_kv_caches({"a0": attn_t, "a1": attn_t, "m0": mamba_t})
        # Attention group (gid 0) registered as regular K/V (2 channels).
        assert 0 in conn._group_kv_spec
        assert conn._group_kv_spec[0]["num_kv_channels"] == 2
        # Mamba group (gid 1) skipped entirely.
        assert 1 not in conn._group_kv_spec
        # And crucially: the 3-D mamba tensor did NOT register as MLA.
        assert all(s["num_kv_channels"] != 1 for s in conn._group_kv_spec.values())

    def test_mla_group_still_registers_as_one_channel(self):
        # Regression: a genuine MLA group (3-D tensor, attention spec) must
        # STILL be detected as num_kv_channels=1 — the skip is spec-gated,
        # not shape-gated.
        torch = pytest.importorskip("torch")
        mla_t = torch.zeros((100, 16, 576), dtype=torch.bfloat16)
        kv_cfg = _kv_cfg([_group(["l0"], FakeMLAAttentionSpec())])
        conn = self._conn(kv_cfg)
        conn.register_kv_caches({"l0": mla_t})
        assert 0 in conn._group_kv_spec
        assert conn._group_kv_spec[0]["num_kv_channels"] == 1

    def test_pure_recurrent_fallback_skips(self):
        # Single Mamba group -> main loop skips it -> _group_kv_spec empty ->
        # fallback runs, consults the spec, and skips again (no MLA revival).
        torch = pytest.importorskip("torch")
        mamba_t = torch.zeros((100, 16, 2048), dtype=torch.bfloat16)
        kv_cfg = _kv_cfg([_group(["m0"], FakeMambaSpec())])
        conn = self._conn(kv_cfg)
        conn.register_kv_caches({"m0": mamba_t})
        assert conn._group_kv_spec == {}

    def test_list_valued_recurrent_cache_dropped_no_crash(self):
        # Real hybrid models (Qwen3.5 GDN) register recurrent state as a
        # *list* of tensors (conv_state, ssm_state), NOT a single tensor, and
        # the recurrent layer is often layer 0 (registered first). The save
        # path probes `next(iter(self._kv_caches.values())).device`; a list
        # has no `.device` -> the engine crashed with
        # "'list' object has no attribute 'device'". register_kv_caches must
        # drop non-tensor entries so this can never happen.
        torch = pytest.importorskip("torch")
        attn_t = torch.zeros((2, 100, 16, 8, 128), dtype=torch.bfloat16)
        mamba_state = [torch.zeros((100, 4, 128)), torch.zeros((100, 16, 128))]
        kv_cfg = _kv_cfg(
            [
                _group(["m0"], FakeMambaSpec()),  # registered FIRST, like layer 0
                _group(["a0", "a1"], FakeFullAttentionSpec()),
            ]
        )
        conn = self._conn(kv_cfg)
        # the list value must not raise during registration
        conn.register_kv_caches({"m0": mamba_state, "a0": attn_t, "a1": attn_t})
        # non-tensor recurrent state dropped from the offload set
        assert "m0" not in conn._kv_caches
        assert "a0" in conn._kv_caches and "a1" in conn._kv_caches
        # attention group still registered as regular K/V
        assert any(s["num_kv_channels"] == 2 for s in conn._group_kv_spec.values())
        # and the save-stream device probe must run past sample-selection
        # WITHOUT raising AttributeError on the list. On a CPU test tensor it
        # returns None cleanly (device.type != "cuda"); the point is no crash.
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True, Stream=lambda device: ("stream", device)
            )
        )
        assert conn._ensure_save_stream(fake_torch) is None


class TestSchedulerBootstrapSkip:
    """Scheduler side: _bootstrap_group_kv_spec_from_config must mirror the
    worker skip so build_connector_meta never plans chunks for a Mamba group.
    No torch needed — this path is config-only."""

    def _conn(self):
        conn = _bare_connector()
        conn._kv_cache_config = None
        return conn

    def test_hybrid_bootstrap_skips_mamba(self):
        kv_cfg = _kv_cfg(
            [
                _group(["a0", "a1"], FakeFullAttentionSpec()),
                _group(["m0"], FakeMambaSpec()),
            ]
        )
        conn = self._conn()
        conn._bootstrap_group_kv_spec_from_config(kv_cfg)
        assert 0 in conn._group_kv_spec
        assert 1 not in conn._group_kv_spec
        # Mamba layer must not be mapped to a planned group.
        assert "m0" not in conn._layer_to_group
        assert conn._layer_to_group.get("a0") == 0
        # observing the Mamba group must flip the hybrid flag
        assert conn._has_non_paged_groups is True

    def test_hybrid_gates_external_load(self):
        # vLLM asserts num_external_computed_tokens==0 for hybrid (Mamba)
        # models. Once a non-paged group is seen, get_num_new_matched_tokens
        # MUST advertise zero external match or the engine crashes on the
        # first L3 hit. Saves must STILL be planned (hashes stashed), so the
        # gate sits after the stash, not before.
        conn = self._conn()
        conn._has_non_paged_groups = True
        conn._pending_block_hashes = {}
        conn._req_id_of = lambda r: "req-x"
        conn._extract_block_hashes_after = lambda r, n: [b"h0", b"h1"]
        assert conn.get_num_new_matched_tokens(object(), 0) == (0, False)
        # load gated, but attention-only SAVE remains possible (hashes stashed)
        assert conn._pending_block_hashes.get("req-x") == [b"h0", b"h1"]


class TestDsaSplitOffload:
    """A DSA mixed group (main MLA latent 576 + sparse-attention indexer 132, as
    in glm_moe_dsa / deepseek_v32) is SPLIT into two uniform MLA sub-specs and
    BOTH are offloaded: the main latent at its own gid, the indexer at
    _DSA_INDEXER_GID_BASE+gid (aliasing group-0's block ids). Restoring both on an
    external L3 hit reproduces native L1 prefix-cache reuse (the sparse-attn scan
    reads K over the whole sequence from the cache), which is byte+scale faithful
    (test_dsa_indexer_roundtrip) and GPU-E2E deconfounded reload==cold 8/8 on
    DeepSeek-V3.2-Exp."""

    _MAIN = "model.layers.0.self_attn.attn"
    _IDX = "model.layers.0.self_attn.indexer.k_cache"

    def test_worker_register_splits_dsa_group(self):
        from infera.engine.vllm.kvd_connector import _DSA_INDEXER_GID_BASE

        torch = pytest.importorskip("torch")
        main_t = torch.zeros((100, 16, 576), dtype=torch.bfloat16)  # MLA latent
        idx_t = torch.zeros((100, 16, 132), dtype=torch.uint8)  # indexer
        kv_cfg = _kv_cfg([_group([self._MAIN, self._IDX], FakeMLAAttentionSpec())])
        conn = _bare_connector()
        conn._kv_cache_config = kv_cfg
        conn.register_kv_caches({self._MAIN: main_t, self._IDX: idx_t})
        # Main latent -> gid 0 (hidden 576); indexer -> gid 1000 (hidden 132).
        assert 0 in conn._group_kv_spec
        assert conn._group_kv_spec[0]["hidden_dim"] == 576
        assert conn._group_kv_spec[0]["num_kv_channels"] == 1
        assert _DSA_INDEXER_GID_BASE in conn._group_kv_spec
        assert conn._group_kv_spec[_DSA_INDEXER_GID_BASE]["hidden_dim"] == 132
        # Both layers stay in the offload set, mapped to their sub-spec gids.
        assert self._MAIN in conn._kv_caches and self._IDX in conn._kv_caches
        assert conn._layer_to_group[self._MAIN] == 0
        assert conn._layer_to_group[self._IDX] == _DSA_INDEXER_GID_BASE

    def test_scheduler_bootstrap_registers_dsa_group_unsplit(self):
        # The scheduler bootstrap does NOT split the DSA group (it registers it as
        # one group); only the worker splits. Splitting on both sides desyncs
        # chunk_tokens -> L3 misses -> garbage (GPU-E2E confirmed). So gid 0 is
        # present and the synthetic indexer gid is NOT.
        from infera.engine.vllm.kvd_connector import _DSA_INDEXER_GID_BASE

        kv_cfg = _kv_cfg([_group([self._MAIN, self._IDX], FakeMLAAttentionSpec())])
        conn = _bare_connector()
        conn._kv_cache_config = None
        conn._bootstrap_group_kv_spec_from_config(kv_cfg)
        assert 0 in conn._group_kv_spec
        assert _DSA_INDEXER_GID_BASE not in conn._group_kv_spec

    def test_non_dsa_mla_group_still_offloads(self):
        # Strict no-op for non-DSA: a plain MLA group (no "indexer" layer) must
        # STILL register (num_kv_channels=1) — the skip is DSA-specific, not a
        # blanket MLA skip.
        torch = pytest.importorskip("torch")
        mla_t = torch.zeros((100, 16, 576), dtype=torch.bfloat16)
        kv_cfg = _kv_cfg([_group(["l0"], FakeMLAAttentionSpec())])
        conn = _bare_connector()
        conn._kv_cache_config = kv_cfg
        conn.register_kv_caches({"l0": mla_t})
        assert 0 in conn._group_kv_spec
        assert conn._group_kv_spec[0]["num_kv_channels"] == 1

    def _split_conn(self):
        # A worker that has split a mixed DSA group (gid 0 main + gid 1000
        # indexer), the config vLLM produces for glm_moe_dsa / GLM-5.2.
        torch = pytest.importorskip("torch")
        main_t = torch.zeros((100, 16, 576), dtype=torch.bfloat16)
        idx_t = torch.zeros((100, 16, 132), dtype=torch.uint8)
        kv_cfg = _kv_cfg([_group([self._MAIN, self._IDX], FakeMLAAttentionSpec())])
        conn = _bare_connector()
        conn._kv_cache_config = kv_cfg
        conn.register_kv_caches({self._MAIN: main_t, self._IDX: idx_t})
        return conn

    def test_split_dsa_entry_expands_mixed_save_and_load(self):
        # The scheduler emits ONE gid-0 chunk carrying BOTH main + indexer
        # layer names (it does not split). _split_dsa_entry must fan it into
        # main (gid 0, hidden 576, original key) + indexer (gid 1000, hidden
        # 132, key with the trailing gid byte swapped to 1000 & 0xFF = 232) so
        # each is gathered at its own hidden_dim -- without this the 132-wide
        # indexer tensors overrun a 576-wide gather (the GLM-5.2 OOB bug).
        from infera.engine.vllm.kvd_connector import _DSA_INDEXER_GID_BASE

        conn = self._split_conn()
        blocks = ((5,), (6,))
        key = b"\x01\x02\x03\x04\x05\x06\x07\x00"  # 7 content + gid byte 0
        idx_byte = bytes([_DSA_INDEXER_GID_BASE & 0xFF])  # 1000 & 0xFF = 232

        save = (blocks, key, "long", 0, [self._MAIN, self._IDX])
        out = conn._split_dsa_entry(save)
        assert len(out) == 2
        assert out[0] == (blocks, key, "long", 0, [self._MAIN])
        assert out[1] == (blocks, key[:7] + idx_byte, "long", _DSA_INDEXER_GID_BASE, [self._IDX])

        load = (blocks, key, 0, [self._MAIN, self._IDX])
        out = conn._split_dsa_entry(load)
        assert len(out) == 2
        assert out[0] == (blocks, key, 0, [self._MAIN])
        assert out[1] == (blocks, key[:7] + idx_byte, _DSA_INDEXER_GID_BASE, [self._IDX])

    def test_split_dsa_entry_noop_when_not_mixed(self):
        # No indexer sub-spec registered (plain MLA / regular attn / DeepSeek's
        # natively-separate groups) OR an entry with no indexer layers -> the
        # entry passes through untouched.
        conn = self._split_conn()
        blocks = ((5,),)
        key = b"\x01\x02\x03\x04\x05\x06\x07\x00"
        main_only = (blocks, key, "long", 0, [self._MAIN])
        assert conn._split_dsa_entry(main_only) == [main_only]
        other = (blocks, key, "long", 7, [self._MAIN, self._IDX])
        assert conn._split_dsa_entry(other) == [other]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
