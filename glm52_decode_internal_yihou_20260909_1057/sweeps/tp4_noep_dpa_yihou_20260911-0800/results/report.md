# TP4 / EP1 (no expert parallel) / DP-attention internal decode sweep

## Result
All eight points pass: TP4, `ep_size=1`, `dp_size=4`, `enable_dp_attention=true`, ISL 70000,
OSL 10000, simulated acceptance 3.61, EAGLE steps=5 / draft=6 / topk=1, mem-fraction-static 0.85.
Every run reports `complete: true`, exact `useful_output_tokens = C * 10000`, realized acceptance
3.6134393063583814 and 2768 verify iterations. The collector
(`scripts/collect_noep_sweep_yihou.py`, which asserts the topology and token counts) exits 0.

**EP1 + DP attention needed no source change** — it ran correctly on the first attempt. The
`--ep-size 1` path is a plain TP MoE configuration (`--moe-a2a-backend none` is already forced by
`bench/profile_decode.py:server_cli`). **No mem-fraction adjustment was needed either**, at any
concurrency (see "KV headroom" below).

## Measurements
`tpot_ms` is `effective_token_latency_ms_per_user` = full synchronous decode-loop elapsed * C /
useful tokens. `decode_seconds` excludes load, pool allocation, graph capture, bootstrap and warmup.

| C | local batch | TPOT ms | output tok/s | decode s | launch wall s |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 6.245638818759471 | 640.4468967986998 | 62.456 | 958 |
| 8 | 2 | 7.814737337362021 | 1023.7068316745905 | 78.147 | 171 |
| 16 | 4 | 9.821627188567073 | 1629.0579649189801 | 98.216 | 190 |
| 20 | 5 | 10.714399154949934 | 1866.6469029913098 | 107.144 | 196 |
| 24 | 6 | 11.361972574423998 | 2112.309270489213 | 113.620 | 203 |
| 32 | 8 | 12.757144562341272 | 2508.3983209270114 | 127.571 | 219 |
| 40 | 10 | 14.3613898829557 | 2785.24574055834 | 143.614 | 237 |
| 48 | 12 | 15.429530137684196 | 3110.9178031784368 | 154.295 | 259 |

`--batch-size` is the GLOBAL concurrency; with DP attention `dp_size = tp_size = 4`, so the per-rank
batch is C/4 and the CUDA-graph bs is that local value (`bench/topology.py`).

**TPOT and throughput are not two independent measurements here.** The harness defines
`output_tokens_per_second = useful_tokens / elapsed` and
`effective_token_latency_ms_per_user = elapsed * 1000 * C / useful_tokens`, so
`tok/s = C * 1000 / TPOT` identically. Reading "latency fell AND throughput rose" as two
confirmations would be double-counting one number.

The C=4 launch wall of 958s is the cold first point: `phase_seconds.load_pool_capture` was 823.9s
there versus 62.2s at C=8, because the AITER JIT cache and page cache were cold. The other seven
points (171-259s) match the EP4 range.

## KV headroom (why mem-fraction was never touched)
The per-rank KV pool is fixed by mem-fraction-static, not by concurrency: `#tokens: 2436864` and
`memory.target_initialized_bytes = 108.26 GB` at every point. Reserved tokens scale with the LOCAL
batch, `reserved_tokens = (C/4) * 80064`, so the worst case C=48 uses 960768 / 2436864 = 39% of the
pool, and `Memory pool end. avail mem` was still ~41.6 GB at C=24. This was computed before running
C=32/40/48; the runs then confirmed it. No parameter was relaxed to make any point pass.

## EP1 vs EP4, same node, single variable
`control_iterations/` holds TP4 / **ep_size=4** / DPA-on runs of the same points on the **same node,
same container, same allocation**, so the only difference from the table above is expert parallelism.
Authoritative CSV: `ep1_vs_ep4_same_node_yihou.csv`.

| C | EP1 TPOT ms | EP4 TPOT ms | TPOT delta | EP1 tok/s | EP4 tok/s | tok/s delta |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 6.245639 | 7.205778 | **-13.32%** | 640.45 | 555.11 | +15.37% |
| 16 | 9.821627 | 10.968559 | **-10.46%** | 1629.06 | 1458.71 | +11.68% |
| 24 | 11.361973 | 12.321298 | **-7.79%** | 2112.31 | 1947.85 | +8.44% |
| 32 | 12.757145 | 13.397674 | **-4.78%** | 2508.40 | 2388.47 | +5.02% |
| 40 | 14.361390 | 15.248736 | **-5.82%** | 2785.25 | 2623.17 | +6.18% |
| 48 | 15.429530 | 16.430828 | **-6.09%** | 3110.92 | 2921.34 | +6.49% |

(C=8 and C=20 have no same-node EP4 partner; only the cross-node baseline covers them.)

Observations, stated as observations rather than explanations:
- Dropping EP is faster at **every** measured concurrency, by 4.8% to 13.3% in TPOT.
- The advantage shrinks steeply from C=4 to C=32 (-13.3% -> -4.8%), then stops shrinking and widens
  slightly at C=40 and C=48 (-5.8%, -6.1%). The first segment is consistent with a fixed per-step
  expert-dispatch cost amortised over a larger local batch, but the non-monotonic turn at C>=40 is
  **not explained by this sweep** — no per-stage attribution was measured, and three points are not
  enough to separate a real effect from run-to-run variation (no repeats were taken). Left open.

### Node-to-node variation is not the cause
The same-node EP4 control reproduces the earlier cross-node EP4 baseline
(`sweeps/tp4_ep4_dpa_yihou_20260910-0452/summary.csv`) to within 0.4%:

| C | same-node control TPOT | cross-node baseline TPOT | delta |
|---:|---:|---:|---:|
| 4 | 7.205778 | 7.199074 | +0.09% |
| 16 | 10.968559 | 10.980900 | -0.11% |
| 24 | 12.321298 | 12.276800 | +0.36% |

So the previously published cross-sweep comparison stands; the control removes the caveat rather than
changing the conclusion.

## Provenance and limits
- Node crsuse2-m2m-217, allocation 133750, container `yihou-noep-dpa-0911`, image
  `sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d`, pinned SGLang
  `402df1e1e453e1e85ec0f5ac4052d36598cc691a`, model `/shared_nfs/models/GLM-5.2-MXFP4`.
- Method is the internal scheduler-free harness: `scheduler_used: false`, `synthetic_prefix: true`,
  `simulated_acceptance: true`, `real_model_weights: true`, `real_moe_routing: true`. These are
  decode-loop performance numbers under simulated acceptance, not a serving benchmark and not a
  statement about output correctness.
- Every point is a single run. No repeats, so no variance estimate; small deltas (the C>=32 segment)
  should not be over-read.
- C=24 was measured twice. The first measurement's local driver shell was killed by host memory
  pressure after it had launched the remote job, so that run produced a valid `result_yihou.json`
  (TPOT 11.3441, 2115.63 tok/s) but no `launch_status.json` exit-code attestation. It is preserved
  unmodified at `../noep_dpa_on_c24_orphan_driver_yihou/` and excluded from these tables; the
  reported C=24 point is a clean rerun with `exit_code 0`. The two agree to 0.16% on TPOT.
- FlyDSL sparse MLA declines some verify shapes and falls back, matching the baseline sweep's
  behaviour.

## Files
- `summary_yihou.csv` — collector output for the eight EP1 points, authoritative.
- `ep1_vs_ep4_same_node_yihou.csv` — paired same-node comparison, authoritative.
- `../iterations/noep_dpa_on_c<C>_yihou/` — EP1 per-point `result_yihou.json`, per-rank reports,
  per-step JSONL, `command.txt`, code snapshot and hashes, `runtime.log`, `launch_status.json`.
- `../control_iterations/control_ep4_dpa_on_c<C>_yihou/` — same for the EP4 control.
- `../working_process.md` — iteration log including the incidents (foreign container on 217, driver
  shell kill, teammate API failures).
