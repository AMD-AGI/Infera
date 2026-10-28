# GLM-5.2 PD 4+4 (135 -> 136): C104 confirmed, decode-flag / KF fused-indexer trials, and the garble-free settings sweep (2026-10-27/28)

Follows `yihou/glm52.llrepro-c104-conc.packup_20261027`. Same topology/harness (prefill crsuse2-m2m-135 GPU4-7 PP4,
decode crsuse2-m2m-136 GPU4-7 TP4/DP4, llying 4+4 kit + our hooks, AgentX 3600 s, A1, simulated acceptance 2.99 at K3).
Master table: `OPT_LEDGER.md`.

**Why the second half exists.** llying (10-27, branch commits c9ea79ce..19a9c82d) found that the ~48 k stack garbles
with REAL acceptance, mainly because of prefill `SGLANG_PP_COMM_OVERLAP=1`, with decode custom all-reduce adding
short-prompt garbling. 48 k is therefore not a deliverable number. llr44-llr48 measure what the stack does with both
settings removed ("garble-free settings"; correctness itself is still to be confirmed by llying's real-acceptance runs).

## Part 1: on the 48 k stack (overlap ON, custom AR ON)

| run | change vs llr37 (C104) | score | verdict |
|---|---|---|---|
| llr37 / llr37r | C104 / replicate | 48,713 / 48,103 (mean 48,408) | C104 confirmed (qps 3.209 / 3.213) |
| llr40 | decode `--enable-dp-lm-head` | 46,898 (-3.1 %) | rejected (step p50 42.4 -> 43.0 ms) |
| llr41 | decode `--enable-dp-attention-local-control-broadcast` | 48,027 (-0.8 %) | neutral |
| llr42 | KF gfx950 fused decode indexer (rows <= 48), live (CHECK 48/48 identical) | 47,674 (-1.5 %) | not kept (step p50 43.1 ms) |
| llr43 | C108 | 47,322 (-2.2 %) | rejected; peak C104 (C96 47.9 k, C108 47.3 k, C112 45.7 k) |

llr42 needed a module-path alias: this image keeps `Gfx950FusedIndexer` in
`sglang.kernels.ops.attention.dsa.hip_gfx950.fused_decode` (renamed upstream in 693d333), so the KF overlay target never
imported and the 10-25 try (llr22) was a no-op. `kfov_fused/module_path_alias.diff` = alias + double-patch guard.

## Part 2: garble-free settings (prefill PP comm overlap OFF, decode custom AR OFF)

| run | settings | CONC | score | qps | TTFT p50 s | 1800-s transfer timeouts |
|---|---|---|---|---|---|---|
| llr44 | overlap off only | 104 | 41,457 | 2.853 | 9.2 | 12 |
| llr45 / llr45r | both off | 104 | 42,186 / 37,919 | 2.896 / 2.639 | 7.6 / 11.3 | 14 / 17 |
| llr46 | both off | 96 | 42,543 | 2.831 | 5.9 | 5 |
| **llr48** | **both off** | **88** | **44,245** | 2.947 | 4.8 | **1** |
| llr47 | both off | 112 | 39,475 | 2.744 | 11.8 | 21 |
| (ref) llr37 / llr37r | both on | 104 | 48,713 / 48,103 | 3.21 | 6.0 | 3 / 4 |

Timeouts = decode log lines `Decode transfer failed ... timed out after 1800.0s in KVPoll.WaitingForInput`
(`analysis/transfer_timeouts.txt`, sample in `analysis/transfer_timeout.sample.txt`): a decode slot waits 30 min for KV
that the prefill never sends. They rise with concurrency and are more frequent with overlap off (correlation only; the
prefill-side cause is not measured). Without overlap the peak moves down: C88 44,245 (single run) is the best
garble-free number so far. In progress at pack-up: llr37r2 (overlap-on drift control), then llr49 (garble-free C80),
llr48r (C88 replicate).

Also measured: the garble-free stack is prefill-bound (prefill PP0 1.5-1.7 batches/s vs 1.93); its prefill hits 3,523
distinct untuned BF16 GEMM shapes (`analysis/prefill_untuned_gemm.llr45.csv`; M = arbitrary chunk lengths).

## Layout
`results/llr/<run>/`, `sites/` (llr37r..llr49, llr45r, llr48r), `kfov_fused/`, `analysis/`, `scripts/`, `OPT_LEDGER.md`.
Reproduce as in `yihou/glm52.llrepro-k3-fp8-gemm-mem.packup_20261027/REPRODUCE.md` with these site files.
