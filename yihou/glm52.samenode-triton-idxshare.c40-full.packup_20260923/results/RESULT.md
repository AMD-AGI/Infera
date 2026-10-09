# RESULT — same-node P4D4, triton DSA + index_share ON, AgentX CONC=40 full

**Completed 2026-09-23 on `crsuse2-m2m-137`.** Both legs on one machine.
`AgentX passed`, **0 / 4157 request errors**, **full 3,613.6 s** profiling window
reached (the previous attempt died at 89 %).

Raw artifact: `agentx_conc40.json`. Runtime snapshots: `spec_accept.decode.txt`,
`workers.json`, `server-info.29001.json`, `server-info.29257.json`.

## Headline, against the cross-node reference

| metric | `t2f` cross-node (tilelang, idx=false) | **this run** same-node (triton, idx=true) | ratio |
|---|---|---|---|
| **tput total** | 158,390 tok/s | **168,260 tok/s** | **1.062x** |
| **per-GPU** | 19,799 | **21,032** | **1.062x** |
| profiled requests | 4,025 | 4,157 | 1.033x |
| duration | 3,629.9 s | 3,613.6 s | 0.996x |
| input tok/s | 157,235 | 167,124 | 1.063x |
| output tok/s | 1,154 | 1,136 | 0.984x |
| TTFT p50 | 4.17 s | **3.60 s** | 0.864x |
| TTFT p90 | 14.80 s | **13.49 s** | 0.911x |
| TTFT p95 | 26.89 s | **20.10 s** | 0.747x |
| E2E p50 | 11.42 s | 11.12 s | 0.974x |
| ITL p50 | 13.2 ms | 13.3 ms | 1.012x |
| interactivity p50 | 76.1 | 75.1 | 0.988x |
| server GPU cache hit | 93.91 % | 94.22 % | 1.003x |
| theoretical hit | 97.45 % | 97.48 % | 1.000x |
| errors dropped | 1 | 4 | — |

Same-node is **6.2 % faster** than cross-node here, with a **25 % better TTFT
p95**. Crucially the cache hit rates are within 0.3 points of each other, so —
unlike the earlier 0.52x comparison — this difference is **not** a cache-hit
artefact.

## `spec_accept_length` — all four decode ranks moved

| dp_rank | accept length | accept rate |
|---|---|---|
| 0 | 3.575 | 0.515 |
| 1 | 3.500 | 0.500 |
| 2 | 3.525 | 0.505 |
| 3 | 3.650 | 0.530 |

Mean **3.5625**, consistent with the forced 3.61. No rank reads 0.0, so none is
the idle-gauge artefact.

## SCOPE LIMITS — read before quoting any number

1. **This is NOT a single-variable experiment.** Three things differ from the
   reference at once: **co-location**, **triton DSA**, and **index_share=true**.
   You may **not** attribute the 6.2 % to triton, nor to co-location, nor to
   index_share. Separating them needs one run per variable.
2. **IndexShare stability was NOT tested.** The reference packup records this
   shape faulting at ~1 h 26 m of sustained load, which maps to ~10:15 here; this
   run's profiling ended at 09:48. Zero `Memory access fault` was observed, but
   that only covers ~1 hour — **a clean finish here cannot be read as
   "IndexShare ON is stable."** We never reached the point where the reference
   failed.
3. **Correctness is waived by construction.** `SGLANG_SIMULATE_ACC_LEN=3.61`
   forces the accept *count*, not which tokens are right; output is garbled.
   These are timing numbers under forced acceptance.
4. **Errors rose 1 → 4** (`InvalidInferenceResultError`). Too small a base to
   mean anything, recorded rather than hidden.
5. **Triton-for-DSA is inference, not proof.** `dsa_*_backend=triton` is
   confirmed resolved on both legs and triton is confirmed loaded and compiling
   in the live processes, but process-level observables cannot separate
   DSA-triton from sglang's other triton users. See `analysis/config_audit.yihou.md`
   item 10, marked `pass with limit`.

## Two process findings from this run

- **The same-node start gate is insufficient — N=2.** It fired and decode still
  died in the RCCL race (08:13:11); a manual retry ~9 min after prefill went
  healthy came up clean (08:24:51). Combined with the previous workspace's run
  (~17 min), that is two-for-two on both halves. **UNSETTLED** whether the settle
  is causal or the retry merely wins a probabilistic race — do not turn "wait N
  minutes" into a harness fix on this evidence.
- **The monitoring worked.** `watchdog` caught the decode failure within minutes
  and reported the negative (`no Memory access fault`), which is what
  distinguished the known RCCL race from the IndexShare risk. The previous
  attempt died unnoticed for hours.
