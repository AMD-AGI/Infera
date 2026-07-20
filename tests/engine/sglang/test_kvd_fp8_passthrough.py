###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""fp8 KV passthrough for the SGLang kvd adapter.

Unlike the vLLM connector (which packs/unpacks chunks and historically
mis-viewed fp8 payloads), the SGLang adapter (`InferaKvdBackend`) moves
KV *pages* as raw bytes: `_tensor_to_bytes` does `value.view(uint8)
.tobytes()` and `_bytes_into_tensor` size-checks via
`target.numel() * target.element_size()` then `target.view(uint8)
.copy_(...)`. Page strides in `_get_pool_buffer_info` derive from
`dtype.itemsize`. So fp8 (itemsize=1) round-trips by construction — there
is no quantize/dequant/scale and no new knob; the KV dtype follows the
engine.

These tests LOCK IN that invariant so a future change can't reintroduce a
hardcoded 2-byte / bf16 assumption that would silently corrupt an fp8 KV
cache. Pure CPU (no sglang, no GPU, no daemon).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from infera.engine.sglang.kvd_adapter import (  # noqa: E402
    InferaKvdBackend,
    _bytes_into_tensor,
    _tensor_to_bytes,
)


def _fp8_dtype():
    """Prefer ROCm's native e4m3fnuz, fall back to CUDA e4m3fn, then uint8.
    All are 1-byte; the passthrough is identical regardless."""
    for name in ("float8_e4m3fnuz", "float8_e4m3fn"):
        dt = getattr(torch, name, None)
        if dt is not None:
            return dt
    return torch.uint8


def _deterministic_bytes_tensor(nelem: int, dtype, salt: int = 0):
    """Build a tensor whose underlying bytes are a fixed pattern, via a
    uint8 staging buffer bit-cast to `dtype` — so equality is meaningful
    even for fp8 (avoids NaN payloads from random fp8 bit patterns)."""
    itemsize = torch.empty(0, dtype=dtype).element_size()
    flat = torch.tensor(
        [(i * 7 + salt) & 0xFF for i in range(nelem * itemsize)],
        dtype=torch.uint8,
    )
    return flat.view(dtype).reshape(nelem)


def test_fp8_page_roundtrip_bit_identical():
    dt = _fp8_dtype()
    src = _deterministic_bytes_tensor(4096, dt, salt=3)
    payload = _tensor_to_bytes(src)
    # fp8 is 1 byte/elem: payload is exactly nelem bytes (NOT 2x).
    assert len(payload) == src.numel() * src.element_size() == 4096
    target = torch.zeros(4096, dtype=dt)
    out = _bytes_into_tensor(payload, target)
    assert out is not None
    assert torch.equal(out.view(torch.uint8), src.view(torch.uint8))


def test_bf16_page_roundtrip_bit_identical():
    src = _deterministic_bytes_tensor(2048, torch.bfloat16, salt=9)
    payload = _tensor_to_bytes(src)
    assert len(payload) == src.numel() * 2  # bf16 = 2 bytes/elem
    target = torch.zeros(2048, dtype=torch.bfloat16)
    out = _bytes_into_tensor(payload, target)
    assert out is not None
    assert torch.equal(out.view(torch.uint8), src.view(torch.uint8))


def test_size_check_is_dtype_aware():
    """An fp8 payload (half the bytes of bf16) must be accepted into an
    fp8 target and rejected by a bf16 target of the same element count —
    proving the size check keys off element_size(), not a fixed width."""
    dt = _fp8_dtype()
    fp8_payload = _tensor_to_bytes(_deterministic_bytes_tensor(1024, dt))
    assert _bytes_into_tensor(fp8_payload, torch.zeros(1024, dtype=dt)) is not None
    # Same element count, but bf16 target expects 2x the bytes -> miss.
    assert _bytes_into_tensor(fp8_payload, torch.zeros(1024, dtype=torch.bfloat16)) is None


def test_pool_buffer_info_uses_fp8_itemsize():
    """`_get_pool_buffer_info` must compute page_stride_bytes with the
    pool's real dtype itemsize. For an fp8 pool that is 1, not 2 — a
    regression to a hardcoded 2 would double the stride and corrupt reads."""
    dt = _fp8_dtype()
    page_size, kv_cache_dim = 64, 576
    buf = torch.zeros(page_size * kv_cache_dim * 8, dtype=dt)

    class _FakeMLAPool:
        kv_buffer = buf
        page_size = 64
        kv_cache_dim = 576
        dtype = dt

    info = InferaKvdBackend._get_pool_buffer_info(_FakeMLAPool())
    assert info is not None
    _base, _total, page_stride_bytes = info
    assert page_stride_bytes == page_size * kv_cache_dim * 1  # itemsize == 1 for fp8

    # And bf16 on the same shape must be 2x — confirms it tracks the dtype.
    class _FakeMLAPoolBf16(_FakeMLAPool):
        kv_buffer = torch.zeros(page_size * kv_cache_dim * 8, dtype=torch.bfloat16)
        dtype = torch.bfloat16

    info16 = InferaKvdBackend._get_pool_buffer_info(_FakeMLAPoolBf16())
    assert info16 is not None
    assert info16[2] == page_size * kv_cache_dim * 2
