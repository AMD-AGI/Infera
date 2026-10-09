# Same-node 1P1D P4D4 — triton DSA + index_share ON, AgentX CONC=40 full

GLM-5.2-MXFP4, **both the prefill leg and the decode leg on one machine**
(`crsuse2-m2m-137`, MI355X ×8), with **triton DSA backends** and
**`index_share_for_mtp_iteration = true`**, prefill HiCache on, full mode
(3600 s, warmup 10/lane), simulated acceptance 3.61. Run finished 2026-09-23
09:57 UTC.

**`AgentX passed`. 0 / 4157 request errors. The full 3,613.6 s window was
reached** — the previous attempt at this shape died at 89 % of its window and
produced no aggregate, so completing it was part of the deliverable.

| metric | value |
|---|---|
| **total throughput** | **168,260 tok/s** |
| **per-GPU throughput** | **21,032 tok/s/chip** (8 GPUs) |
| requests profiled | 4,157 |
| aiperf request errors | **0 / 4157 = 0.000 %** |
| `records_error_dropped` | 4 (`InvalidInferenceResultError`) |
| TTFT p50 / p90 / p95 | 3.60 s / 13.49 s / 20.10 s |
| E2E p50 | 11.12 s |
| ITL p50 | 13.3 ms |
| interactivity p50 | 75.1 |
| server GPU cache hit | 94.22 % |
| `spec_accept_length` (4 decode ranks) | 3.575 / 3.500 / 3.525 / 3.650 → mean **3.5625** |

## The headline comparison, and how to attribute it

| run | node | DSA | index_share | HiCache | mode | tput | per-GPU |
|---|---|---|---|---|---|---|---|
| `t2f` reference | cross-node | tilelang | false | on | full | 158,390 | 19,799 |
| prior same-node run | **same-node** | tilelang | false | on | full (89 %, self-computed) | ~149,601 | ~18,700 |
| **this run** | same-node | **triton** | **true** | on | full | **168,260** | **21,032** |

Against the cross-node reference this run is **+6.2 %** with **TTFT p95 25 %
better**, and the two have server GPU cache hit rates within 0.3 points
(94.22 % vs 93.91 %) — so, unlike an earlier comparison in this project that
came out at 0.52×, this difference is **not** a cache-hit artefact.

**Decomposition.** Using the middle row, the +6.2 % net splits into two opposing
effects:

- **co-location costs ~5.6 %** (149,601 / 158,390 = 0.944×),
- **triton + index_share gains ~12.5 %** (168,260 / 149,601 = 1.125×).

So the gain attributable to triton + index_share is roughly **twice the net
figure**. Caveat: the middle row is **self-computed from a partial (89 %)
window**, not an official aggregate, so the split is approximate. The two
changes are **not** separated from each other — that would need one run per
variable.

## SCOPE LIMITS — read before quoting any number

1. **triton and index_share are not separated from each other.** They changed
   together. Do not attribute the gain to either one alone.
2. **IndexShare stability was NOT tested.** `glm52.p8d8.agentx-sweep.packup_20260920`
   records this shape with IndexShare **ON** faulting at ~1 h 26 m of sustained
   load, versus 13 h 33 m fault-free with it **off**. This run's profiling ended
   at ~1 hour. Zero `Memory access fault` was observed, but **a clean finish here
   cannot be read as "IndexShare ON is stable"** — we never reached the point
   where the reference failed.
3. **Correctness is waived by construction.** `SGLANG_SIMULATE_ACC_LEN=3.61`
   forces the accept *count*, not which tokens are right; the deployment emits
   garbled text. The acceptance gauge ≈3.56 reports the value it was told to
   report and carries no correctness information.
4. **Errors rose 1 → 4** (`InvalidInferenceResultError`) versus the reference.
   Too small a base to mean anything; recorded rather than hidden.
5. **Triton-for-DSA is inference, not proof.** `dsa_*_backend=triton` is
   confirmed resolved on both legs and triton is confirmed loaded and compiling
   in the live processes, but process-level observables cannot separate
   DSA-triton from sglang's other triton users (`mamba_backend`,
   `linear_attn_backend`). See `analysis/config_audit.yihou.md` item 10.

## Two process findings

**The same-node start gate is insufficient — N=2.** The gate (added after the
previous experiment) fired, and decode still died in the RCCL start race at
08:13:11. A manual retry begun 08:21:19 — ~9 min after prefill went healthy —
came up clean at 08:24:51. The previous experiment's retry was ~17 min after.
Two-for-two on both halves. **Still unsettled** whether the settle is causal or
the retry merely wins a probabilistic race; `notes.md` states the experiment
that would distinguish them.

**The monitoring earned its keep.** A `watchdog` teammate polling every ~3 min
caught the decode failure within minutes and — critically — reported the
*negative*, "no `Memory access fault` present", which is what separated the known
RCCL race from the IndexShare risk this run was deliberately carrying. The
previous attempt at this shape died unnoticed for hours.

## Navigation

| path | what is there |
|---|---|
| `results/RESULT.md` | the numbers and the scope limits |
| `results/agentx_conc40.json` | the raw AgentX aggregate |
| `results/spec_accept.decode.txt`, `workers.json`, `server-info.*.json` | live runtime snapshots |
| `REPRODUCE.md` | ordered, copy-pasteable reproduction |
| `environment.md` | hardware + software env |
| `notes.md` | gotchas, corrections, and what is still open |
| `working_process.md` | every round in order, including the failures |
| `spec/mission.md` | the task of record |
| `analysis/config_audit.yihou.md` | proof the four config deltas were live, 9 pass + 1 `pass with limit` |
| `scripts/bench-harness/` | the patched harness that ran this |
| `patches/` | the three diffs vs the tracked repo |
| `logs/key-excerpts.md` | the exact log lines each conclusion rests on |

**Large artifacts are deliberately not packed** (the user's standing decision):
the 1.3 GB `server_metrics_export.json`, the 119 MB timeslices CSV, and the full
engine logs (16 MB decode, 7.7 MB prefill) stay in the gitignored workspace at
`bench/glm5p2_pd/results/yihou-triton-idxshare/rounds/`.
