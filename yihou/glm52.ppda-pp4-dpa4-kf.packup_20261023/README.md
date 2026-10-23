# GLM-5.2 PD cross-node 135 -> 136: PP4 prefill + DPA4 decode, and the KF decode kernels (2026-10-23)

**Result (AgentX C80, 3600 s, one A1 each, sim acceptance 3.61 - no correctness claim; per-GPU over 8 GPUs):**

| round | change | score tok/s/chip | qps | ISL mean | TTFT p50/p90 s | TPOT ms | GPU/CPU hit | KV xfer p50/p90 s |
|---|---|---|---|---|---|---|---|---|
| ppda1 | PP4 prefill (ppdc config) + DPA4 decode (dpax1: mem 0.95, max-running 96, graph 64) | 34,288 | 2.372 | 114.7k | 4.12 / 22.95 | 20.0 | 0.698 / 0.261 | 1.66 / 4.07 |
| **ppda2** | ppda1 + KF decode kernels (sparse MLA + indexer + MoE overlay from the kernel session) | **40,780** | 2.684 | 120.6k | 4.79 / 18.25 | **14.2** | 0.728 / 0.235 | 2.07 / 4.60 |
| ppdc4 (ref) | PP4 prefill + DCP4 decode | 31,076 | 2.208 | 111.7k | 8.48 / 21.28 | 23.5 | 0.711 / 0.249 | 1.74 / 4.25 |
| llying 4+4 C80 (ref) | his kit v6-pp4, image v0522-fdbafb60 | 37,310 | 2.553 | 115.9k | 3.75 / 10.83 | - | - | - |

- ppda1 vs llying (`gap-vs-llying-c80.analysis.md`): same protocol and scoring; TPOT, prefill queue/forward and KV
  transfer equal; the gap is TTFT (9.10 vs 5.53 s mean) spent in decode_bootstrap (pre-allocation wait, 3.6-7.5 s per DP
  rank vs 1.0-1.7 s): our router split requests exactly equally (2,379 per rank) while a full rank queued and others had
  room; llying runs router R2 (decode rank by input demand), plus custom AR off, top-k v2, decode max-running 256.
- ppda2: KF installed INDEXER/MOE/SPARSE on every decode rank, 0 fallbacks (RAISED), warm-up CHECK sparse_mla snr 100 dB.
  End-to-end correctness with KF (real acceptance / GSM8K) has NOT been run.
- Running at pack-up time: ppda3 = ppda1 + router R2 (`INFERA_R2_DECODE_DEMAND=on`). In preparation: a faithful
  reproduction of llying's 37,310 on 135/136 (his image rebuilt from Infera fdbafb60 + his kit), then our e2e options one
  at a time, then KF (user's order).

Files: `opt.readme.md`, `gap-vs-llying-c80.analysis.md`, `REPRODUCE.md`, `results/` (summary.tsv, rounds/, kf/, dryruns/),
`scripts/`, `patches/`. Previous: `../glm52.ppdc-pp4-dcp4.packup_20261023/`, `../glm52.dcpx-pack-fix-best.packup_20261023/`.
KF kernels themselves: kernel session pack-up `/home/yihou/kf-glm52-yihou/packups/glm52.kf-decode-kernels-e2e.packup_20261023`
(a16-19), overlay `/shared_nfs/yihou_glm52/kf_e2e_20261023/kf/e2e/overlay` (crusoe NFS, MD5SUMS in that pack-up).
