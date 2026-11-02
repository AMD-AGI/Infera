# B8 / S02: MI355X validation of SGLang #46918, November 2, 2026

**The proposed cap fixes the original compiler-abort reproducer at the current PR head.**
On one MI355X GPU, the uncapped M=256 configuration aborts with the original LLVM
assertion on three seeds. The cap and the existing `config=None` workaround both
complete, and their BF16 outputs are bitwise identical for every tested shape/seed.
This supports the scoped fix in [SGLang #46918](https://github.com/sgl-project/sglang/pull/46918)
for [issue #46829](https://github.com/sgl-project/sglang/issues/46829).

## Exact source and environment

- SGLang PR head: `9763ef56bf57d03f8af4f9adc9eebb717cd0d9b1`, clean Python tree.
- One MI355X (`gfx950`, 256 compute units), one visible GPU, no model weights.
- PyTorch `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`, Triton `3.7.0`.
- AITER `e7d2453f25e5aaeea72a3a40b284ba453e026624`. Both wrapper and underlying kernel
  hashes match the October 27 reproducer artifacts. Sources were copied to a
  writable task directory; neither the dispatcher nor the AITER kernel was edited.
- The pinned image supplies dependencies; SGLang is imported from the PR checkout,
  not from the image's 0.5.21 source. See [setup and commands](README.md).

The Q storage is BF16 `[M,64,256]`, sliced to `[:,:,:192]` without making it
contiguous. Weights are FP8 E4M3FN `[64,192,512]`; FP32 weight scales are all one.
All arms call the real production `rocm_absorb_q_bmm(..., is_capture_mode=False)`.
Identical input hashes are checked before comparing results. Each arm has a
separate process and fresh Triton cache, with core dumps disabled.

## Results

Times are median microseconds for the production call, including activation
quantization and allocation: 20 warmups, seven samples of 100 calls, GPU events.
These are descriptive microbenchmarks without a locked-clock performance campaign.

| Run ID | M | Seed | Original waves | Workaround us | Cap=2 us | Uncapped us | Cap vs workaround output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| m256 | 256 | 104729 | 4 | 26.44 | 26.83 | LLVM abort | Bitwise equal |
| boundaries | 128 | 130363 | 2 | 26.53 | 27.02 | 26.45 | Bitwise equal |
| boundaries | 129 | 130363 | 3 | 26.12 | 27.16 | 66.12 | Bitwise equal |
| boundaries | 257 | 130363 | None | 51.89 | 51.90 | 51.98 | Bitwise equal |
| boundaries | 256 | 130363 | 4 | 26.34 | 26.92 | LLVM abort | Bitwise equal |
| public-smoke | 256 | 155921 | 4 | 27.47 | 26.54 | LLVM abort | Bitwise equal |

At M=256, capped and uncapped hand configurations differ only in `waves_per_eu`
(2 versus 4). The uncapped process exits on SIGABRT with
`Sequence.h:275` / `iota_range` / `Begin <= End`. The identical failure was observed
for all three M=256 seeds. M=128 is a positive hand-config control, M=129 exercises
the 3-to-2 cap transition, and M=257 verifies the decode-limit fallback: all arms
select `config=None` there. Every successful uncapped output also matches the
workaround bitwise.

All 12 required capped/workaround cases completed; the uncapped controls produced
three successful boundary cases and three expected compiler aborts. The final
`public-smoke` run executes the published [harness](validate_b8.py). The first two
runs used the same validation logic with a local hostname exclusion; the public
version removes private hostname literals and records the explicit GEMM CSV.
Per-case versions, source/input hashes, configurations, raw timing samples and
numerical diagnostics are in [RESULTS.json](RESULTS.json).

The M=256 measurements do not establish a consistent speed advantage over the existing workaround.
The M=129 uncapped timing is recorded as an observation, not a general speedup
claim. No 48k deployment throughput is attributed to this patch.

CPU validation of the candidate's registered unit file also passed: **11 tests,
265 subtests**, Python 3.13 / PyTorch 2.14.1+cpu:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python -m pytest -q -p no:cacheprovider \
  test/registered/unit/models/test_rocm_mla_absorb_fusion_gate.py
```

## What this establishes and what remains

Changing only the wave budget removes the reported abort in the original
compiler environment and preserves the workaround's outputs on this matrix.
The FP32 comparison in the JSON uses original BF16 activations and dequantized
stored FP8 weights. AITER additionally quantizes activations, so its error is a
diagnostic, not a quantization-matched accuracy pass; bitwise paired agreement is
the numerical regression control.

This is not V-absorb coverage, an all-head-count or all-architecture qualification,
or a full-model correctness test. It neither establishes the underlying LLVM
defect nor proves that cap=2 is optimal on every device. It does not validate the
historical simulated-MTP 48k deployment. Retain the existing workaround until an
appropriate image includes a reviewed fix.

The PR is already Ready. The original MI355X reproducer is no longer an untested
hardware gate; maintainers still need to enable required CI and review/merge the
candidate. Other environments and any changed code after a rebase need their
appropriate coverage. No duplicate PR or mutation of the original author's
branch was made.
