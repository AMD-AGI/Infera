# TP4 / DP-attention high-concurrency decode: EP on vs EP off

Internal **scheduler-free** harness (`bench/profile_decode.py` — no server, no scheduler, no PD
fake-input path). GLM-5.2-MXFP4, 4x MI355X, ISL 70000, OSL 10000, simulated acceptance 3.61,
EAGLE steps=5 / draft=6 / topk=1, `kv-cache-dtype fp8_e4m3`, FlyDSL DSA backends.
Nodes `crsuse2-m2m-254` (job 136670) and `crsuse2-m2m-267` (job 136669), 2026-09-14.

## What was asked and what was delivered

Requested points: C = 128, 160, 192, 224, 256, 288, both with EP on and EP off.

| requested C | outcome |
|---:|---|
| 128 | **measured**, both arms, both nodes |
| 160 | **infeasible** — measured negative result, 12 attempts (see below) |
| 192, 224, 256, 288 | **infeasible by arithmetic** — KV pool alone exceeds the device |

With the user's agreement, C = **64** and **96** were added to fill the gap between the previous
sweep's ceiling (C=48) and C=128, so the EP trend has more than a single high-concurrency point.

## Results — 12 points, all verified

Every row below passed, mechanically: `complete: true`, `useful_output_tokens == C x 10000`,
`verify_iterations == 2768`, `realized_accept_length == 3.6134393063583814`,
`(tp, dp, dpa) == (4, 4, True)`, `ep_size` as requested, driver `exit_code 0`.

| C | local batch | node | EP off (ep1) TPOT ms | EP on (ep4) TPOT ms | EP off tok/s | EP on tok/s |
|---:|---:|---|---:|---:|---:|---:|
| 64 | 16 | 254 | 17.5528 | 17.9472 | 3646.1 | 3566.0 |
| 64 | 16 | 267 | 17.5190 | 17.9769 | 3653.2 | 3560.1 |
| 96 | 24 | 254 | 21.9358 | 22.1619 | 4376.4 | 4331.8 |
| 96 | 24 | 267 | 21.9994 | 22.3749 | 4363.7 | 4290.5 |
| 128 | 32 | 254 | 25.5301 | 25.7811 | 5013.7 | 4964.9 |
| 128 | 32 | 267 | 25.7918 | 25.9187 | 4962.8 | 4938.5 |

**`tok/s = C x 1000 / TPOT` is an identity in this harness.** The two throughput columns are the
first two columns in different units, not independent confirmation.

## The EP-off advantage decays with concurrency and is nearly gone by C=128

Same node, same container, same allocation — single variable:

| C | node 254 | node 267 |
|---:|---:|---:|
| 64 | **-2.198 %** | **-2.547 %** |
| 96 | **-1.020 %** | **-1.678 %** |
| 128 | **-0.973 %** | **-0.490 %** |

(negative = EP off is faster). Joined to the prior sweep, which used the same method and node type:

| C | 4 | 16 | 24 | 32 | 40 | 48 | 64 | 96 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EP off vs EP on | -13.32 % | -10.46 % | -7.79 % | -4.78 % | -5.82 % | -6.09 % | -2.2/-2.5 % | -1.0/-1.7 % | -1.0/-0.5 % |

**What can be claimed:** EP off was faster in **6 of 6** same-node pairs measured here, and in every
point of the prior sweep. The sign is consistent. The magnitude falls steeply with concurrency.

**What cannot be claimed:** that the C=128 magnitude is resolved. At C=128 the EP delta
(0.49-0.97 %) is the same size as the node-to-node spread (below). These are single runs.

## How large is the noise? (this sweep's own replication)

Two independent measurements of each configuration, one per node:

| C | EP off spread | EP on spread |
|---:|---:|---:|
| 64 | 0.193 % | 0.165 % |
| 96 | 0.290 % | 0.961 % |
| 128 | 1.025 % | 0.534 % |

Plus a within-node, near-repeat pair (the `expandable_segments` control at C=128, identical except
for the PyTorch allocator flag): **0.265 %** (254 / ep1) and **0.205 %** (267 / ep4), with opposite
signs — so the allocator flag does not move the measurement, and run-to-run repeatability on one
node is roughly 0.2 %.

**This resolves an open question from the previous packup.** That report flagged as "unexplained"
that the EP advantage stopped shrinking and *widened* between C=32 (-4.78 %) and C=48 (-6.09 %) —
a 1.3-point excursion. The node-to-node spread measured here reaches 1.03 % and the within-node
repeat 0.27 %, so an excursion of that size is within what unreplicated single runs produce. The
monotone decay is the signal; that bump is most plausibly noise. Not proven, but no longer unexplained.

## Why C >= 160 was not delivered

### The cap that was hiding in plain sight
The previous sweep stopped at exactly C=48 with every point at `local_batch_size <= 12`. That was
never a memory limit. The pinned SGLang prints, and `kv_cache_configurator.py` implements:

> `Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests.`

The per-rank request pool is `max_running_requests // attn_dp_size` = 48 / 4 = **12 slots**, so the
local batch could never exceed 12 and global concurrency could never exceed 48, whatever the KV pool
size. All points here therefore pass `--max-running-requests <C>` explicitly. At C=48 the default
equals the explicit value, so the prior sweep's numbers remain directly comparable.

### C = 192 / 224 / 256 / 288: arithmetic
Per rank: device 287.98 GiB, target weights 119.84 GiB, draft weights 6.92 GiB, leaving
**151.66 GiB**. KV costs 46.58 KiB/token and each request reserves 80064 tokens = **3.556 GiB**.
With DP attention each rank holds `C/4` requests:

| C | local batch | KV required | vs 151.66 GiB ceiling |
|---:|---:|---:|---|
| 192 | 48 | 170.7 GiB | over |
| 224 | 56 | 199.2 GiB | over |
| 256 | 64 | 227.6 GiB | over |
| 288 | 72 | 256.1 GiB | over |

`--mem-fraction-static` bounds the static fraction, not the device; it cannot buy this memory.

### C = 160: measured, not assumed
KV needs 3,202,560 tokens = 142.3 GiB, which *does* fit under 151.66 GiB — but nothing else does.
Twelve attempts across both arms, both nodes, five mem-fractions:

| mem-fraction | pool tokens (ep1 / ep4) | failure |
|---|---|---|
| 0.9700 | 3,177,984 / 3,158,336 | `KV capacity insufficient` |
| 0.9735 | below requirement | `KV capacity insufficient` |
| 0.9740 | 3,202,688 (margin 128 tokens) | 267: capture OOM on a **22 MiB** alloc. 254: capture and bootstrap **succeeded**, then died in warmup with NCCL `Failed to CUDA calloc 6291456 bytes` |
| 0.9772 | ~3,202,700 (ep4) | capture OOM |
| 0.9780 | 3,227,392 / 3,207,744 | capture OOM, 962 MiB block |
| 0.9800 (+/- expandable_segments) | 3,239,744 / 3,220,096 | capture OOM, 962 MiB block |

The failure walks down the pipeline as the pool shrinks: KV check -> CUDA-graph capture
(`eagle_draft_extend_cuda_graph_runner.py:270` -> `aiter_paged_mqa_logits`, a 962 MiB contiguous
workspace) -> NCCL buffers. On the single most favourable configuration it reached the warmup loop
and still could not obtain **6 MiB**. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` did not
help; fragmentation (`reserved but unallocated`) stayed at 1.36 GiB and grew to 2.41 GiB at 0.978,
so pool memory given back was absorbed by fragmentation rather than becoming a contiguous block.

**There is no window at TP4 between "KV pool large enough" and "enough memory left to capture and
communicate".** To reach C >= 160 at this ISL the parallelism must change (TP8 halves the per-rank
weight footprint and the per-rank batch), which would be a different, non-comparable curve.

## Provenance
- Method, scripts and pitfalls follow `packups/glm52_tp4_noep_dpa_sweep_yihou.packup_20260911-1100/`.
- Image pinned **by digest** `sha256:b9a83742f631...`, re-asserted before every point.
  SGLang `402df1e1e453e1e85ec0f5ac4052d36598cc691a`. No benchmark code was modified.
- `--mem-fraction-static`: 0.85 at C=64/96 (matching the prior sweep), 0.88 at C=128 (0.85 yields
  2,436,864 pool tokens, below the 2,562,048 required). Mem-fraction changes pool size only.
- Files: `summary_yihou.csv` (all 41 rows, 14 pass), `ep_delta_same_node_yihou.csv`,
  `node_replication_yihou.csv`, `../working_process.md` (raw timestamped log, including every
  failed attempt).

## Limits
Single runs, no repeats beyond the one cross-node replication and one near-repeat per node. This is
the internal harness: `scheduler_used: false`, `synthetic_prefix: true`, `simulated_acceptance: true`
(with real weights and real MoE routing). It measures decode-loop performance under simulated
acceptance — **not** a serving benchmark and **not** a statement about output correctness.
