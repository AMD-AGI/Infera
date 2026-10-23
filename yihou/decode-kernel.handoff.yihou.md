# Handoff to the decode-kernel optimization work: GLM-5.2 PD best configs to plug kernels into (2026-10-23)

Two best cross-node 1P1D configs (prefill crsuse2-m2m-135 GPU0-3 -> decode crsuse2-m2m-136 GPU0-3, MI355X, AgentX,
sim acceptance 3.61). Both use image `infera-sglang:v0522-1020-yihou-dcp-mc4340-destpin` + sglang overlay `93034eba91`
(`/shared_nfs/yihou_glm52/dcp_pd_20261021/src/sglang-93034eba91`) on both legs.

| | DPA4 decode best | DCP4 decode best |
|---|---|---|
| round / score | dpax1, 26,321 tok/s/chip @C56 (running now: ppda1 = PP4 prefill + this decode @C80) | ppdc2, 31,128 @C72 (ppdc1 27,426 @C56) |
| pack-up | `yihou/glm52.dcpx-pack-fix-best.packup_20261023/` | `yihou/glm52.ppdc-pp4-dcp4.packup_20261023/` |
| env / dryrun | `scripts/envs/env.dpax1-135x136.yihou.sh`, `results/dryruns/dryrun.dpax1-*.txt` there | `scripts/envs/env.ppdc2-135x136.yihou.sh`, `results/dryruns/dryrun.ppdc2-*.txt` |
| decode leg | TP4/DP4 DPA, EP1, mem 0.95, max-running 96, graph max bs 64, EAGLE 5/1/6, index_share true, triton DSA, custom AR on | TP4 EP4 DP1 `--dcp-size 4`, KEEP_KV_B_BF16 + split triton indexer, mem 0.88 + empty_cache patch, max-running 96, graph bs 1..96 (13 sizes), EAGLE 5/1/6, index_share true, triton DSA, custom AR on |
| decode TPOT | 16.7 ms @C56 | 19.3 @C56, 22.5 @C72 |

How to measure a decode kernel change in this setup:
- Put the changed file(s) into a decode-only mount tsv (`DECODE_MOUNTS_ADD`, format `<src>\t<dst in container>`; the
  DCP config already uses `mounts.yihou.dcpx-ec-decode.tsv` for decode.py/metrics_reporter.py - add rows to a copy).
- One variable vs the env above, dryrun + token diff, A1 only, compare TPOT, decode step time (decode-0.a1.log
  `Decode batch` lines), and score at the same concurrency. Node-to-node / run-to-run spread is ~3 %.
- llying measured decode step ~30 % slower under DCP4 than DPA4 at equal load (4+4 C80: 92 vs 71 ms) - the main reason
  DCP trails DPA; per-kernel breakdown of the DCP decode step (RCCL, indexer, MoE, GEMM) has not been profiled yet.
- Correctness is waived by sim acceptance in these runs; a kernel change needs its own correctness check.
