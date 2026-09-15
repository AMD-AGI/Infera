# GLM-5.2-MXFP4 internal decode bench — TP4 / DP-attention, EP off vs EP on, C = 4 → 128

Consolidated results of **every completed run of the internal scheduler-free decode method**
(`bench/profile_decode.py`), across four campaigns between 2026-09-09 and 2026-09-14.

## Launch configuration (identical for every row unless the row says otherwise)

| Parameter | Value |
|---|---|
| Model | `/shared_nfs/models/GLM-5.2-MXFP4` (quark / MXFP4), `gfx950` = MI355X |
| Method | internal harness `bench/profile_decode.py` — intercepts `TpModelWorker` + `EAGLEWorkerV2` via `ParallelState`. **No server, no scheduler, no PD fake-input path** (`scheduler_used: false`) |
| Tensor parallel | `--tp-size 4` (4 of 8 GPUs, `HIP_VISIBLE_DEVICES=0,1,2,3`) |
| Expert parallel | `--ep-size 1` (**EP off**) vs `--ep-size 4` (**EP on**); `--moe-a2a-backend none` forced in both |
| DP attention | `--enable-dp-attention` ⇒ `dp_size = tp_size = 4`, local batch = `C/4` |
| Input / output | `--input-len 70000` / `--output-len 10000` |
| Speculative | EAGLE, `--speculative-num-steps 5 --speculative-num-draft-tokens 6 --speculative-eagle-topk 1` |
| Acceptance | `--accept-length 3.61`, `match-expected`, `real-draft-token` (simulated) |
| KV cache | `--kv-cache-dtype fp8_e4m3`, page size 64 |
| DSA backends | `--dsa-decode-backend flydsl --dsa-prefill-backend flydsl --dsa-topk-backend aiter` |
| CUDA graph | enabled; `--cuda-graph-bs-decode` = `--cuda-graph-max-bs-decode` = `C/4` |
| Other | `--enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope --warmup-steps 10 --random-seed 1234 --disable-radix-cache --skip-tokenizer-init --disable-overlap-schedule` |
| `--mem-fraction-static` | 0.85 for C ≤ 96, **0.88 for C = 128** (pool sizing only — see §5) |
| `--max-running-requests` | **C**, mandatory for C > 48 (see §5) |
| Image / SGLang | image digest `sha256:b9a83742f631…`, SGLang `402df1e1e453e1e85ec0f5ac4052d36598cc691a`, AITER `2c71811b…`, torch `2.9.1+rocm7.2.0`, HIP `7.2.26015` |

**Metric definition — read this before reading any table.** The harness defines
`output_tokens_per_second = useful_tokens / elapsed` and
`TPOT_ms = elapsed × 1000 × C / useful_tokens`. Therefore

> **tok/s = C × 1000 / TPOT is an identity.**

Latency and throughput columns are **one measurement in two units**. "Latency fell *and* throughput
rose" is not two findings.

**Validity gate applied to every row.** `complete: true`, `useful_output_tokens == C × 10000`,
`verify_iterations == 2768`, `realized_accept_length == 3.6134393063583814`, resolved topology as
requested, and driver `exit_code 0`. The last two quantities are deterministic given the parameters,
so a mismatch means the configuration is not the documented one regardless of how plausible the
timing looks. Rows that failed the gate are excluded here and retained as evidence in the source
workspaces.

---

## 1. Headline table — EP off vs EP on, same node, single variable

Both arms of every pair ran on the **same node, same container, same allocation**, back to back.

| C | local batch | node | EP off (ep1) TPOT ms | EP on (ep4) TPOT ms | **TPOT Δ** | EP off tok/s | EP on tok/s |
|---:|---:|---|---:|---:|---:|---:|---:|
| 4 | 1 | 217 | 6.2456 | 7.2058 | **−13.33 %** | 640.4 | 555.1 |
| 16 | 4 | 217 | 9.8216 | 10.9686 | **−10.46 %** | 1629.1 | 1458.7 |
| 24 | 6 | 217 | 11.3620 | 12.3213 | **−7.79 %** | 2112.3 | 1947.8 |
| 32 | 8 | 217 | 12.7571 | 13.3977 | **−4.78 %** | 2508.4 | 2388.5 |
| 40 | 10 | 217 | 14.3614 | 15.2487 | **−5.82 %** | 2785.2 | 2623.2 |
| 48 | 12 | 217 | 15.4295 | 16.4308 | **−6.09 %** | 3110.9 | 2921.3 |
| 64 | 16 | 254 | 17.5528 | 17.9472 | **−2.20 %** | 3646.1 | 3566.0 |
| 64 | 16 | 267 | 17.5190 | 17.9769 | **−2.55 %** | 3653.2 | 3560.1 |
| 96 | 24 | 254 | 21.9358 | 22.1619 | **−1.02 %** | 4376.4 | 4331.8 |
| 96 | 24 | 267 | 21.9994 | 22.3749 | **−1.68 %** | 4363.7 | 4290.5 |
| 128 | 32 | 254 | 25.5301 | 25.7811 | **−0.97 %** | 5013.7 | 4964.9 |
| 128 | 32 | 267 | 25.7918 | 25.9187 | **−0.49 %** | 4962.8 | 4938.5 |

Negative = EP off is faster. **EP off won 12 of 12 same-node pairs.** The advantage decays
monotonically in trend from ~13 % at C=4 to ~0.5–1 % at C=128.

> C = 8 and C = 20 are omitted from this table on purpose: EP-off data exists for them, but the only
> EP-on counterpart was measured on a **different node**, so the pair would not be single-variable.

---

## 2. EP-off scaling curve (the operating curve)

TP4 / DP-attention on / EP off. One row per concurrency; where two nodes measured the same point the
node-254 value is shown (the node-267 value is in §4).

| C | local batch | TPOT ms | output tok/s | tok/s per GPU | throughput vs C=4 |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 6.2456 | 640.4 | 160.1 | 1.00× |
| 8 | 2 | 7.8147 | 1023.7 | 255.9 | 1.60× |
| 16 | 4 | 9.8216 | 1629.1 | 407.3 | 2.54× |
| 20 | 5 | 10.7144 | 1866.6 | 466.7 | 2.91× |
| 24 | 6 | 11.3620 | 2112.3 | 528.1 | 3.30× |
| 32 | 8 | 12.7571 | 2508.4 | 627.1 | 3.92× |
| 40 | 10 | 14.3614 | 2785.2 | 696.3 | 4.35× |
| 48 | 12 | 15.4295 | 3110.9 | 777.7 | 4.86× |
| 64 | 16 | 17.5528 | 3646.1 | 911.5 | 5.69× |
| 96 | 24 | 21.9358 | 4376.4 | 1094.1 | 6.83× |
| 128 | 32 | 25.5301 | 5013.7 | 1253.4 | **7.83×** |

32× the concurrency buys **7.83×** the throughput at **4.09×** the per-user latency.

---

## 3. Secondary comparison — DP attention off vs on (EP on, same sweep, same node)

From the 2026-09-10 campaign. Note the two arms do **not** run the same per-rank shape: with DPA off
`dp_size = 1` and every rank carries the full batch `C`; with DPA on each rank carries `C/4`.

| C | DPA **off** TPOT ms (local bs = C) | DPA **on** TPOT ms (local bs = C/4) | Δ |
|---:|---:|---:|---:|
| 4 | 6.1505 (4) | 7.1991 (1) | −14.57 % |
| 8 | 7.9991 (8) | 8.8705 (2) | −9.82 % |
| 16 | 10.2861 (16) | 10.9809 (4) | −6.33 % |
| 20 | 11.6671 (20) | 11.8078 (5) | −1.19 % |
| 24 | 12.6322 (24) | 12.2768 (6) | **+2.90 %** |

DPA-off leads at low concurrency and the lines **cross between C=20 and C=24**; beyond that DPA-on
is ahead. Only five points exist and the crossing margin (1.2 % → 2.9 %) is close to the noise floor
in §4 — treat the crossover as indicative, not established.

---

## 4. Noise floor — what a difference has to beat to mean anything

Every number in this document is a **single run**. These are the only repeat-like measurements taken.

| Comparison | A (ms) | B (ms) | spread |
|---|---:|---:|---:|
| C=24 EP off, driver-killed run vs clean rerun (node 217) | 11.3441 | 11.3620 | 0.157 % |
| C=128 EP off, default allocator vs `expandable_segments` (node 254) | 25.5301 | 25.4625 | 0.266 % |
| C=128 EP on, default allocator vs `expandable_segments` (node 267) | 25.9187 | 25.9719 | 0.205 % |
| C=4 EP on, node 217 vs node A | 7.2058 | 7.1991 | 0.093 % |
| C=16 EP on, node 217 vs node A | 10.9686 | 10.9809 | 0.112 % |
| C=24 EP on, node 217 vs node A | 12.3213 | 12.2768 | 0.362 % |

Node-to-node spread on **identical** configurations, 2026-09-14 campaign:

| C | EP off spread | EP on spread |
|---:|---:|---:|
| 64 | 0.193 % | 0.165 % |
| 96 | 0.290 % | 0.961 % |
| 128 | **1.025 %** | 0.534 % |

**Working noise floor: ~0.2 % within a node, up to ~1.0 % between nodes** — and the between-node
figure grows with concurrency. A difference below ~1 % at C ≥ 96 is not resolvable from single runs.

---

## 5. Two ceilings that bound the whole curve

| Ceiling | What it is | Effect | Status |
|---|---|---|---|
| **48-request speculative cap** | SGLang resolves `max_running_requests` to 48 under speculative decoding and sizes the per-rank request pool as `48 / attn_dp_size` = **12 slots** | local batch could never exceed 12, i.e. **global C could never exceed 48**, regardless of KV pool size | **Removed** by passing `--max-running-requests C` explicitly (global; SGLang divides internally) |
| **TP4 memory wall** | Per rank: 287.98 GiB device − 119.84 GiB target weights − 6.92 GiB draft weights = **151.66 GiB** free. KV costs 46.58 KiB/token; each request reserves 80064 tokens = **3.556 GiB** | C=160 needs 142.3 GiB KV and leaves no room for graph capture + NCCL; C≥192 exceeds the ceiling outright | **Not removable at TP4** |

Memory arithmetic for the requested-but-undelivered points:

| C | local batch | KV required | verdict |
|---:|---:|---:|---|
| 128 | 32 | 113.8 GiB | fits at `--mem-fraction-static 0.88` — **measured** |
| 160 | 40 | 142.3 GiB | KV fits; **12 attempts all failed** — capture needs a 962 MiB contiguous block, NCCL then needs 6 MiB, neither is available |
| 192 / 224 / 256 / 288 | 48 / 56 / 64 / 72 | 170.7 / 199.2 / 227.6 / 256.1 GiB | **over the 151.66 GiB ceiling — arithmetic, not run** |

The C=48 endpoint of the earlier campaign was set by the **first** ceiling, not by memory — every
point there reported `local_batch_size ≤ 12`. `--mem-fraction-static` bounds the static fraction, not
the device, so it cannot buy past the second.

---

## 6. Impact analysis — what actually moves the number

Ranked by measured effect size, largest first.

| Factor | Direction | Measured magnitude | Confidence | Basis |
|---|---|---|---|---|
| **Concurrency C** | dominates everything | TPOT 6.25 → 25.53 ms (4.09×); throughput 640 → 5014 tok/s (7.83×) | **High** | 11 points, monotone, far above noise |
| **Expert parallelism off** | faster, shrinking with C | −13.3 % at C=4 → −0.5…−1.0 % at C=128 | **High on sign** (12/12 pairs), **low on magnitude at C ≥ 96** (≈ noise) | §1, same-node pairs |
| **DP attention** | off wins below C≈20, on wins above | −14.6 % at C=4 → +2.9 % at C=24 | **Medium** — 5 points, crossover margin near noise | §3 |
| **Node identity** | none intended | 0.17 – 1.03 % | — | §4; **this is the bar the EP effect must clear at high C** |
| **PyTorch allocator (`expandable_segments`)** | none detectable | ±0.2 %, opposite signs on two nodes | **High that it is negligible** | §4; run as a deliberate control, not assumed |
| **`--mem-fraction-static` (0.85 → 0.88)** | none on speed | pool size only | **High** | pool token counts scale linearly; kernels unchanged |
| **`--max-running-requests`** | none on speed, **gating on feasibility** | C>48 impossible without it | **High** | §5, traced in pinned source |

**Reading of the EP result.** The shrinking gap is consistent with a roughly fixed per-step
expert-dispatch overhead being amortised over a larger local batch: at local batch 1 that overhead is
13 % of the step, at local batch 32 it is ~1 %. **This sweep does not measure that overhead** — no
per-stage attribution was taken — so the mechanism is a hypothesis, not a finding. What is
established is the sign and the decay.

**One prior open question is downgraded.** The 2026-09-11 report flagged as "unexplained" that the EP
advantage stopped shrinking and widened between C=32 (−4.78 %) and C=48 (−6.09 %) — a 1.3-point
excursion. The node-to-node spread measured later reaches 1.03 % and within-node repeat 0.27 %, so an
excursion of that size is within what unreplicated single runs produce. Most plausibly noise;
**not proven**, because no repeats were taken at C=32/40/48.

---

## 7. Conclusions

1. **Turning expert parallelism off is the right default for this configuration**, but the reason to
   do it disappears as concurrency rises. At C ≤ 24 it is worth 8–13 %. At C = 128 it is worth
   0.5–1.0 %, which is the same size as the node-to-node spread — **at production-scale concurrency
   the two arms are not distinguishable by this method with single runs.**
2. **Throughput scales sub-linearly but usefully**: 32× concurrency → 7.83× throughput, 4.09× TPOT.
   Per-GPU output rises 160 → 1253 tok/s.
3. **The practical concurrency ceiling at TP4 / ISL 70000 is C = 128.** C = 160 fails not on KV
   capacity but on the memory left over for CUDA-graph capture and NCCL; C ≥ 192 fails on KV
   capacity alone. Reaching higher requires TP8 (halves both per-rank weights and per-rank batch),
   which is a different and non-comparable curve.
4. **`--max-running-requests` must be set explicitly for any C > 48.** Without it the harness
   silently caps at 12 request slots per rank. Any earlier result that stopped at C=48 stopped
   because of this, not because of memory.
5. **Do not read differences under ~1 % as real** at C ≥ 96, and under ~0.3 % anywhere. Everything
   here is a single run.

### What would raise confidence, in priority order
1. **Repeats at C = 96 and 128** (3 runs per arm per node). This is the single highest-value
   follow-up: it is what separates "EP off is ~1 % faster" from "EP no longer matters".
2. **Per-stage attribution** (MoE dispatch/combine vs attention vs draft) to test the amortisation
   hypothesis directly instead of inferring it from the decay.
3. **A TP8 curve** to cover C = 160 → 288, reported as its own curve and not spliced onto this one.
4. **A controlled CUDA-graph on/off A/B.** Not available today: the only graph-off run in the project
   (`iterations/002_eager_smoke_yihou`) used ISL 1024, `max_steps 2` and **`warmup_steps 0`**, so its
   99.5 s for 2 iterations is dominated by first-call compilation, and its graph-on neighbour
   (`003_graph_smoke_yihou`) used a different `max_steps`, `warmup_steps` and OSL. **Comparing them
   would produce a ~2000× figure that is an artefact, not a speedup.** A valid A/B needs identical
   ISL/OSL/C and identical warmup on both arms.

---

## 8. Provenance and limits

| Campaign | Date | Node(s) | Contents |
|---|---|---|---|
| `iterations/` | 2026-09-09 | — | method bring-up; one complete TP8 reference point (below) |
| `sweeps/tp4_ep4_dpa_yihou_20260910-0452/` | 2026-09-10 | node A | EP4, DPA on/off, C = 4…24 |
| `sweeps/tp4_noep_dpa_yihou_20260911-0800/` | 2026-09-11 | 217 | EP1 C = 4…48 + same-node EP4 control |
| `sweeps/tp4_highconc_yihou_20260914-0423/` | 2026-09-14 | 254, 267 | EP1 and EP4, C = 64/96/128, both nodes; C=160 boundary attempts |

Packaged reproduction kits: `packups/glm52_tp4_ep4_dpa_sweep_yihou.packup_20260910-062400/`,
`packups/glm52_tp4_noep_dpa_sweep_yihou.packup_20260911-1100/`,
`packups/glm52_tp4_highconc_ep_sweep_yihou.packup_20260914-0800/`.

**Reference point outside the TP4 matrix** (kept separate because its topology is not comparable):
`iterations/005_full_target_yihou` — **TP8**, C=16, same ISL/OSL/acceptance, TPOT **7.8175 ms**,
**2046.7 tok/s**. That run predates the harness's EP/DPA flags: its config records only `--tp-size 8`,
so **its EP and DP-attention state is not recorded** and it must not be read as a TP8-vs-TP4 result.

**Also excluded from every table above:** the `comparisons/fake_server_10k500_yihou_20260910-0650/`
baselines (C=16 → 9.2747 ms, C=32 → 11.9401 ms). They are the internal method but at
**ISL 10000 / OSL 500** with `dp_size 1`, built to mirror a server reference — a different workload,
not a point on this curve.

### Scope of the claim
This is the internal scheduler-free harness: `scheduler_used: false`, `synthetic_prefix: true`,
`simulated_acceptance: true`, with `real_model_weights: true` and `real_moe_routing: true`. It
measures **decode-loop performance under simulated acceptance**. It is **not** a serving benchmark
(no scheduler, no queueing, no streaming, no real prefill) and says **nothing** about output
correctness. All verification was single-reviewer.
