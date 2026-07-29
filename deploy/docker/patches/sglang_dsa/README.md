# sglang DSA patches

Patches against the sglang source tree bundled in the ROCm engine images
(`lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` and derivatives, where sglang is an
editable checkout at `/sgl-workspace/sglang`).

## `dsa_indexer_hip_dp_padded_rows.diff`

**Enables DP-attention and EAGLE MTP to run together for GLM-5.2 DSA on gfx950.**
Without it, the combination crashes on the first batch:

```
RuntimeError: Expected lengths.size(0) == B to be true, but got false.
```

The DSA indexer's aiter/HIP paged-MQA path sizes its `logits` output from the
**DP-padded** row count, while the CUDA path slices to the real row count (`q_offset`) —
which is what `lengths` is sized to. The patch makes HIP follow the CUDA contract: run
top-k over the real rows, then restore the padding. Two guards keep the non-DP path
bit-identical. Full rationale is in the patch header.

Applies to `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`, exact context, no
fuzz:

```bash
cd /sgl-workspace/sglang
patch -p1 --fuzz=0 < dsa_indexer_hip_dp_padded_rows.diff
```

Or bind-mount the patched file over the container's copy.

#### On sglang v0.5.16, drop `--fuzz=0`

The diff was cut against 0.5.15.post1. On v0.5.16 it still applies, but hunk 2 needs one
line of fuzz, so `--fuzz=0` fails and `git apply` (which has no fuzz at all) fails too:

```
Hunk #1 succeeded at 975 (offset 72 lines).
Hunk #2 succeeded at 1047 with fuzz 1 (offset 73 lines).
```

The fuzz is benign. v0.5.16 replaced one context line above the padding-restore guard
with a new `self._mask_init_and_local_tokens(logits, seqlens_32)` call; the line the
patch actually rewrites (`if not _is_hip and q_offset < q_fp8.shape[0]:`) is unchanged
and unambiguous. The new call is also consistent with the fix rather than at odds with
it: it sizes `row_starts` from `seqlens_32`, which is the unpadded (`q_offset`-sized)
`get_seqlens_expanded()` on the verify / draft-extend paths — so once the patch makes
`logits` carry `q_offset` rows too, the mask's row count matches the logits' exactly,
where before the patch it was merely `<=`.

Verified by applying to a v0.5.16 tree and diffing the result: both hunks land in the
intended place, and the resulting file matches the patch's intent line for line.
`examples/sglang_glm5.2/patch_sglang.sh` therefore uses `patch -p1` with default fuzz.

### Verified

8× MI355X (gfx950), ROCm 7.2.0, sglang 0.5.15.post1, GLM-5.2-MXFP4, single node
`--dp-size 8 --enable-dp-attention --ep-size 8` + `--speculative-algorithm EAGLE
--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`:

| Check | Result |
|---|---|
| Correctness probe | 4/4 |
| Spec-dec accept length | median 3.86 / 4 (n=251) |
| conc=64, ISL/OSL 1k/1k, 256 prompts | 256/256, 0 failed, median TPOT 17.2 ms |
| Regression: DP-attention only | 4/4, unchanged |
| Regression: MTP only | 4/4, unchanged |

Full reproduction kit (environment, commands, raw logs, evidence, root-cause notes) lives
in the `infera.yihou.glm5.2.mxfp4` workspace under
`glm52.mxfp4.spur.mooncake.packup_20260728/dpa_mtp_fix/`.

Also verified on **8× MI325X (gfx942), ROCm 7.2, sglang v0.5.16, GLM-5.2-FP8**, both
single-node DP8 and 1P1D PD disaggregation, with MTP=1. Without the patch, warmup's first
batch throws the assert on DP0 and DP3 and `scheduler_0 crashed with exit code -3`; with
it, warmup passes and the full `verify_correctness` suite is green (needle 9/9,
humaneval-long 20/20, +0% long-context delta), accept length 3.78/4 on real prompts, 0
asserts across a conc=64 run. See `examples/sglang_glm5.2/REPORT.zh.md` §1.3.

### One more thing PD needs, which this patch does not fix

On the **PD decode leg** with MTP, this patch clears the assert but the leg then deadlocks
in PD warmup and `/health` stays 503 forever. That is a **separate defect** with the same
upstream origin (#30839): `dsa_topk_indices` arrives None on a PREBUILT batch, which makes
`can_cuda_graph` a per-rank decision in `eagle_worker_v2.py`, and the ranks split between
the draft CUDA graph and eager with mismatched collectives. Upstream issue #32527, fix PR
#32209 (open). Until that lands, PD + MTP additionally needs

```
--json-model-override-args '{"index_share_for_mtp_iteration":false}'
```

on both legs, which costs nothing under PD because upstream already disables fused DSA
top-k there anyway. `examples/sglang_glm5.2/*_sglang_{prefill,decode}.sh` add it
automatically when `MTP != 0`. Full trace in REPORT.zh.md §1.2.

### Upstream

The bug this patch fixes came in with **#30839** ("Stabilize GLM-5.2 MTP IndexShare across
PD and CUDA graph replay", 2026-07-14, in v0.5.16; cherry-picked to v0.5.15.post1 as
#31083). That PR falls back from graph replay to eager when the DSA seed is missing, and it
is the eager path that carries physically padded rows.

The same fix is in flight for other backends, in the same shape as this one — trim to the
real row count before the indexer, restore the physical shape after:

| Upstream | Backend | State |
|---|---|---|
| PR #32762 | NPU (`npu_lightning_indexer`) | open, CI red |
| PR #32209, 4th item | TRT-LLM | open, CI red |
| this patch | **aiter / HIP** | no upstream counterpart |

So the HIP path is the one nobody has covered yet, which makes this diff a reasonable
upstream candidate; #32762 is the closest template.

It is **not** the same issue as sglang PR #30378 / #30427 ("clamp padded-row seq_lens to
>= 0" in `triton_ops/pad.py::seqlens_expand_kernel`), which fixes padded row *values* and
is already present in this image. This one fixes the HIP-side row *count*.
