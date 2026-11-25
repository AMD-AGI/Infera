#!/usr/bin/env python3
"""Checks of the MXFP4 indexer K cache overlay (scripts/make_idxfp4_overlay.py, vLLM #61985 target-only backport). Run
in the vLLM image with the five overlay files mounted over the image's own (OUT_DIR/mounts.txt), so `import vllm`
picks them up:
  test_idxfp4.py cpu   no GPU needed (VLLM_TARGET_DEVICE=cpu; ROCm / CDNA4 / aiter checks are stubbed):
                       - every overlaid module imports;
                       - dsa_indexer_uses_fp4 takes GLM-5.2, DeepSeek-V3.2 and the V3.2 MTP draft, still rejects
                         DeepSeekMTPModel, and is off with indexer_kv_dtype unset (the PR's plan test);
                       - target-only selection: with mxfp4, layers.0 / layers.77 get the MXFP4 cache (68 B per token,
                         DEEPSEEK_V32_ROCM_MXFP4_INDEXER), layers.78 and any layer of an MTP-architecture model keep
                         FP8 (132 B, DEEPSEEK_V32_INDEXER); VPD_IDX_MXFP4_DRAFT=1 puts the draft on MXFP4 too;
                         indexer_kv_dtype unset keeps every layer on FP8;
                       - the decode top-k hook declines what the Triton port does not cover (compressed or
                         candidate-masked layers, > 128 rows, non-int32 lengths) and is off without VPD_TOPK_IMPL.
  test_idxfp4.py gpu   1 GPU (gfx950), PYTHONPATH=<dir with vpd_triton_topk_decode.py>, VPD_TOPK_IMPL=triton|native:
                       the hook in `_dense_decode` gives an exact top-2048 per row, -1 padded, on decode-shaped rows
                       (fp32 logits max_model_len = 1M wide, int32 lengths 1000 .. 700003 and tie rows, int32 out with
                       the topk buffer's row stride).
"""
import os
import sys
import types
from unittest import mock

MODE = sys.argv[1]
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if MODE == "cpu":
    # The hook's module reads its switches at import.
    os.environ["VPD_TOPK_IMPL"] = "triton"
    sys.path.insert(0, SCRIPTS)

import torch  # noqa: E402

GLM = ("glm_moe_dsa", "GlmMoeDsaForCausalLM")
MTP = ("deepseek_mtp", "DeepseekV32MTPModel")
PREFIX = "model.layers.{}.self_attn.indexer.k_cache"


def vllm_config(kv_dtype, model_type, arch):
    from vllm.config import AttentionConfig

    return types.SimpleNamespace(
        attention_config=AttentionConfig(indexer_kv_dtype=kv_dtype),
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(model_type=model_type),
            hf_text_config=types.SimpleNamespace(index_head_dim=128),
            architectures=[arch],
        ),
        compilation_config=types.SimpleNamespace(static_forward_context={}),
        cache_config=types.SimpleNamespace(block_size=64),
    )


def rocm_stubs():
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.backends.mla import indexer

    return [
        mock.patch.object(indexer, "current_platform", types.SimpleNamespace(is_rocm=lambda: True)),
        mock.patch("vllm.platforms.rocm.get_cdna_version", lambda: 4),
        mock.patch.object(rocm_aiter_ops, "is_enabled", classmethod(lambda cls: True)),
    ]


def build_cache(cfg, layer):
    from vllm.config import set_current_vllm_config
    from vllm.models.deepseek_v32.amd import rocm

    with set_current_vllm_config(cfg):
        c = rocm.DeepseekV32ROCmIndexerCache(
            head_dim=128 + 128 // 128 * 4, dtype=torch.uint8, prefix=PREFIX.format(layer), cache_config=cfg.cache_config
        )
        spec = c.get_kv_cache_spec(cfg)
    return c.mxfp4, c.head_dim, spec.page_size_bytes, c.get_attn_backend().get_name()


def cpu():
    import vllm.models.deepseek_v32.amd.rocm as rocm
    import vllm.models.deepseek_v32.common.kernels as kernels
    import vllm.v1.attention.backends.mla.rocm_paged_mxfp4_indexer as backend
    import vllm.v1.attention.ops.rocm_paged_mxfp4_indexer as ops
    from vllm.v1.attention.backends.mla import indexer

    assert hasattr(kernels, "_mxfp4_quant_and_cache_write") and hasattr(backend, "DeepseekV32RocmMxfp4IndexerBackend")
    assert ops._VPD_TOPK_IMPL == "triton" and hasattr(ops, "_vpd_tk")
    print("imports ok:", ", ".join(m.__name__ for m in (rocm, kernels, indexer, backend, ops)))

    stubs = rocm_stubs()
    for s in stubs:
        s.start()
    try:
        for model_type, arch, wired in [("deepseek_v4", "DeepseekV4ForCausalLM", True), (*GLM, True),
                                        ("deepseek_v32", "DeepseekV32ForCausalLM", True), (*MTP, True),
                                        ("deepseek_mtp", "DeepSeekMTPModel", False)]:
            assert not indexer.dsa_indexer_uses_fp4(vllm_config("auto", model_type, arch))
            if wired:
                assert indexer.dsa_indexer_uses_fp4(vllm_config("mxfp4", model_type, arch)), arch
            else:
                try:
                    indexer.dsa_indexer_uses_fp4(vllm_config("mxfp4", model_type, arch))
                    raise AssertionError(f"{arch} took the MXFP4 cache")
                except ValueError as e:
                    assert "only supported for" in str(e)
        print("dsa_indexer_uses_fp4: GLM-5.2 / V3.2 / V3.2 MTP wired, DeepSeekMTPModel rejected, auto = fp8")

        mx = (True, 68, 64 * 68, "DEEPSEEK_V32_ROCM_MXFP4_INDEXER")
        fp8 = (False, 132, 64 * 132, "DEEPSEEK_V32_INDEXER")
        cases = [
            ("mxfp4", GLM, 0, mx), ("mxfp4", GLM, 77, mx), ("mxfp4", GLM, 78, fp8), ("mxfp4", MTP, 78, fp8),
            ("mxfp4", MTP, 0, fp8), ("auto", GLM, 3, fp8), ("auto", MTP, 78, fp8), ("fp8", GLM, 3, fp8),
        ]
        for kv, (model_type, arch), layer, want in cases:
            got = build_cache(vllm_config(kv, model_type, arch), layer)
            assert got == want, (kv, arch, layer, got, want)
            print(f"  {kv:5s} {arch:22s} layers.{layer:<2d} -> mxfp4={got[0]} head={got[1]} page={got[2]} {got[3]}")
        with mock.patch.object(rocm, "_VPD_IDX_MXFP4_DRAFT", True):
            for (model_type, arch), layer in ((GLM, 78), (MTP, 78)):
                got = build_cache(vllm_config("mxfp4", model_type, arch), layer)
                assert got == mx, (arch, layer, got)
        print("target-only selection ok (draft FP8 by default, MXFP4 with VPD_IDX_MXFP4_DRAFT=1)")
    finally:
        for s in stubs:
            s.stop()

    # Hook guards: every case below must decline before any kernel runs (CPU tensors).
    def layer(rows, compress_ratio=1, candidates=None):
        return types.SimpleNamespace(compress_ratio=compress_ratio, candidates=candidates, topk_tokens=2048,
                                     topk_buffer=torch.full((rows, 2048), -7, dtype=torch.int32))

    def call(lay, rows, lens_dtype=torch.int32):
        logits = torch.zeros(rows, 4096)
        return ops._vpd_decode_topk(lay, logits, torch.full((rows,), 4096, dtype=lens_dtype), PREFIX.format(3))

    assert not call(layer(8, compress_ratio=4), 8)
    assert not call(layer(8, candidates=torch.zeros(8, 4)), 8)
    assert not call(layer(129), 129)
    assert not call(layer(8), 8, torch.int64)
    with mock.patch.object(ops, "_VPD_TOPK_SCOPE", "draft"):
        assert not call(layer(8), 8)
    with mock.patch.object(ops, "_VPD_TOPK_IMPL", ""):
        assert not call(layer(8), 8)
    print("decode top-k hook guards ok")
    print("idxfp4 overlay cpu test passed")


def gpu():
    import vllm.v1.attention.ops.rocm_paged_mxfp4_indexer as ops

    impl = ops._VPD_TOPK_IMPL
    assert impl in ("triton", "native"), "set VPD_TOPK_IMPL=triton or native"
    k, W = 2048, 1048576
    lens = torch.tensor([1000, 1001, 1002, 1003, 150_000, 150_001, 150_002, 150_003, 700_000, 700_001, 700_002, 700_003,
                         700_000, 700_001, 5_000, 300_000], dtype=torch.int32, device="cuda")
    rows = lens.numel()
    logits = torch.full((rows, W), float("-inf"), device="cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    for r in range(rows):
        n = int(lens[r])
        x = torch.randn(n, device="cuda", generator=g)
        if r >= 12:  # tie rows: all 0 except a few positives
            frac = (0.0015, 0.0015, 0.0, 0.02)[r - 12]
            x = torch.where(torch.rand(n, device="cuda", generator=g) < frac, x.abs() + 0.1, torch.zeros_like(x))
        logits[r, :n] = x
    buf = torch.full((rows + 3, k), -7, dtype=torch.int32, device="cuda")  # the topk buffer has spare rows
    lay = types.SimpleNamespace(compress_ratio=1, candidates=None, topk_tokens=k, topk_buffer=buf)
    assert ops._vpd_decode_topk(lay, logits, lens, PREFIX.format(3)), "hook declined a decode-shaped call"
    torch.cuda.synchronize()
    for r in range(rows):
        n = int(lens[r])
        kk = min(n, k)
        sel, pad = buf[r, :kk], buf[r, kk:]
        assert (pad == -1).all() and (sel >= 0).all() and (sel < n).all() and sel.unique().numel() == kk, r
        kth = torch.topk(logits[r, :n], kk).values[-1]
        assert (logits[r, sel.long()] >= kth).all(), r
    assert (buf[rows:] == -7).all(), "wrote past the decode rows"
    print(f"{impl} top-k through the MXFP4 decode hook exact on {rows} rows (lengths 1000 .. 700003)")
    print("idxfp4 overlay gpu test passed")


if __name__ == "__main__":
    {"cpu": cpu, "gpu": gpu}[MODE]()
