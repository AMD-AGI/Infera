# GLM-5.3-Flash-MXFP4 fixlen p50 vs the GLM-5.2-MXFP4 baseline

Ours: node `smci355-ccs-aus-n01-33`, 2026-09-02, MIX (aggregated), **TP4 / 4 GPUs**,
decode CUDA graphs on, `--disable-shared-experts-fusion`, kv-aware router,
no DP-attention, no MTP, no kvd.

Baseline: `/home/yihou/dev/git.16-19/infera.glm52.mix.experiment/fixlen.glm52.mix.packup_20260806/`
(external to this packup; verified present 2026-09-02),
GLM-5.2-MXFP4, MIX, **TP8 / 8 GPUs**, DP-attention + MTP + kvd + prefix cache all ON.

Same harness (`sglang.bench_serving`), same shape (isl 7400 / osl 320,
`--random-range-ratio 1.0`, `--num-prompts 10 x conc`, temp 1.0 / top-p 0.95).

| conc | GLM-5.3-Flash-MXFP4 TP4 out tok/s | GLM-5.2-MXFP4 TP8 out tok/s | ratio | our TTFT p50 | their TTFT p50 |
|---:|---:|---:|---:|---:|---:|
| 1  | **111.02** | 82.56  | 1.34x | 255 ms  | 1047 ms |
| 8  | **561.04** | 395.58 | 1.42x | 1065 ms | 2012 ms |
| 16 | **962.55** | 679.82 | 1.42x | 744 ms  | 1978 ms |
| 24 | **1391.15**| 746.75 | 1.86x | 619 ms  | 2276 ms |

## How to read this, and how NOT to

The alignment bar in `mission.md` is *"performance roughly in line with GLM-5.2"*.
It is cleared and then some — but the comparison is **not** same-model-new-version,
and three differences all point the same way, so do not quote the ratio as a
speedup:

1. **Different models.** GLM-5.3-Flash is 320 B total / 18 B active with hybrid
   KDA attention. GLM-5.2 is 744 B with uniform MLA+DSA. A smaller, sparser
   model being faster is expected, not a finding.
2. **Half the hardware.** Ours is TP4 on 4 GPUs; the baseline is TP8 on 8. Per
   GPU the gap is ~2.7-3.7x, which overstates it in the other direction.
3. **Ours is the LESS optimised configuration** and still wins: the baseline runs
   DP-attention, EAGLE MTP and kvd; ours runs none of them. That is the one
   comparison-independent statement worth keeping — there is headroom we have
   not touched.

The mission's fixlen-alignment requirement is written against **GLM-5.3 (big)**,
which is the same architecture as GLM-5.2 and therefore the real apples-to-apples
comparison. That belongs to the big-model track.
