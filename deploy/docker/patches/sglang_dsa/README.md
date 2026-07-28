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

### Known limitation

Under **PD disaggregation** with MTP on the decode leg, the server starts cleanly and the
crash above no longer occurs, but the first routed request deadlocks in the EAGLE
draft-extend metadata path (`dsa_backend.py::init_forward_metadata`, a `.max().item()`
GPU→CPU sync racing a collective across DP ranks). That is a **separate defect**, not
addressed by this patch — PD with DP-attention only is unaffected and passes.

### Upstream

This is ROCm-specific; the CUDA path was always correct. It is **not** the same issue as
sglang PR #30378 / #30427 ("clamp padded-row seq_lens to >= 0" in
`triton_ops/pad.py::seqlens_expand_kernel`), which fixes padded row *values* and is
already present in this image. This one fixes the HIP-side row *count*.
