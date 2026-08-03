# sglang `bench_serving` sweep on the FULL-feature GLM-5.2 deployment (spur)

**Ran:** 2026-08-01, 08:34 – 11:35 UTC (bring-up + sweep; measured window
10:25:41 – 11:35:16)
**Author:** yihou
**Nodes:** `crsuse2-m2m-253` (prefill) + `crsuse2-m2m-236` (decode), 8× MI355X
gfx950 each, mlx5 RoCE
**Model:** GLM-5.2-MXFP4, two-node PD over mooncake RDMA
**Status:** **PASS** — all features proven on, 8/8 sweep points completed, zero
faults.

## What this is

Phase 2 of `spec/agentic.bench.kv.liying.mtp.md`: benchmark the most
feature-complete infera deployment of GLM-5.2 that exists — **kvaware + kvd +
MTP + PD + DPA together** — with sglang's own `bench_serving`, before running
Optimus-AgenticBench Case A against it.

Nobody had benchmarked this combination before. The merged branch was validated
**on vultr** (ionic + peermem, dma-buf off, ctx 32768); the previous spur Case A
run used the **kvaware+kvd-only** image with **MTP off**. This run is the first
time all five features are measured together, on this cluster.

**Case A itself is a separate kit** — see *Related*. This one covers the
deployment, the feature proofs, and the synthetic sweep.

## Result

| # | goal | result | verdict |
|---|---|---|---|
| 1 | every feature genuinely ON, not just flagged | PD/mooncake, DPA 8/8, MTP, kv-aware, kvd — each with a check that would go red | **PASS** |
| 2 | 8-point `bench_serving` sweep, one server | 8/8 completed, `real conc` ≈ requested at every point | **PASS** |
| 3 | MTP acceptance measured | **2.0–2.4** at the p50 point (≈55–60 % @ 4 draft tokens) | **PASS** |
| 4 | deployment health under load | `Traceback` **0**, `Memory access fault` **0**, `Scheduler hit an exception` **0**, both legs, 70 min | **PASS** |

Full numbers and analysis: **`RESULTS.md`**.

### The headline shape

**Prefill-bound, and saturated by conc=32.** Quadrupling concurrency 32 → 128
buys only **19 %** (p50 point) / **26 %** (p90 point) more input throughput,
while p50 TTFT rises **24×** / **19×** and TPOT rises only **1.4×** / **1.5×**.
Requests queue, then are served at close to full speed.

| point | ISL | OSL | conc | in tok/s | TTFT p50 | TPOT p50 | E2E p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| p50 | 74,000 | 320 | 1 | 5,108 | 10.8 s | 11.5 ms | 14.5 s |
| p50 | 74,000 | 320 | 128 | 32,907 | 259.1 s | 16.5 ms | 263.3 s |
| p90 | 155,000 | 3,300 | 1 | 1,583 | 24.5 s | 22.6 ms | 98.9 s |
| p90 | 155,000 | 3,300 | 128 | 29,433 | 467.7 s | 33.6 ms | 575.7 s |

> ⚠️ **These E2E figures are not an SLA verdict.** `bench_serving` fires
> `--request-rate inf` — every request offered at t=0 with no think time — so
> queue depth is maximal by construction. Case A is closed-loop with a 4 s median
> inter-turn delay. `e2e_p50_ms ≤ 4500` is answered by the Case A kit, not here.

### The three numbers that actually discriminate

A green run proves little on this stack. These would have gone red:

**1. kvd is *serving*, not merely wired.** A latency win proves nothing — the
in-GPU radix cache serves a repeated prefix without ever touching L3. Restart the
engine, keep the daemon, replay:

| | gets | hits | sets | misses |
|---|---:|---:|---:|---:|
| after restart | 0 | 0 | 40,798 | 0 |
| **after replay** | **11,250** | **11,250** | **40,798 (flat)** | **0** |

`sets` staying put is the load-bearing part: reads, not re-writes. And the count
is not merely non-zero — it is the *right* number:
`360,000 tok ÷ 64 page × 2 pools (KV + INDEXER) = 11,250`, exactly.

**2. kv-aware routing hashes the same prefix the engine does.** Not "the view is
non-zero" (it would be, for uninteresting reasons) but an exact identity between
two independent subsystems: router `cache_hits=937` × `block_size 64` =
**59,968** = the engine's independently reported `cached_tokens`.

**3. MTP is accelerating, not looping.** `accept len` **2.0–2.4** at the p50
point. `4.00` would be *bad* news — it means the draft model predicted every
token because the output is a repetition loop.

## Read this before trusting any number

* **`accept_length` is `null` in all 8 bench JSONs, by construction.**
  `bench_serving` reads it from `<base_url>/server_info`; our base-url is the
  router, which has none. The decode leg's own value is a **cumulative,
  per-DP-rank** mean, so it is not a per-point number either. The per-point table
  is binned from 12,654 timestamped `accept len:` samples in the decode log.
* **The p90 point's low acceptance (1.3–1.6) is the benchmark's prompt, not the
  model.** `--dataset-name random` builds long prompts by **repeating** one
  ShareGPT conversation's token ids (`datasets/random.py:131-134`) — hundreds of
  repeats at ISL 155K. Quote the p50 value.
* **The sweep exercises kvd's write path only.** Every prompt is unique, so
  there is no prefix for L3 to serve: `gets` flat, `sets` +173,270, evictions
  +172,857. Expected; the read path is proven separately.
* **The bigram kv-event fix is *not* exercised here** — measured on the wire, the
  prefill leg emits plain ints. Had that fix been absent, every number here would
  be unchanged. This corrects a claim in a predecessor kit.
* **No kvd-off A/B**, so no performance claim is made for kvd.
* **The needle probe reads 4/5, and that is not a defect** — the failures are
  stochastic (identical prompt + identical warm cache → 3/6), and the bench's own
  prompts cannot reach that failure mode.

## The one thing that had to be fixed to run here at all

The ROCm `hipHostRegister` fix is **not on the merged branch** — the branch was
validated on vultr at ctx 32768 with short prompts, a regime that never triggers
the fault. On spur at 120K–235K tokens with kvd on, the prefill leg dies with
`Memory access fault by GPU node-N` on the first write-back, because `gfx950` is
`xnack-` and has no page-migration fallback.

It is applied here as **two uncommitted working-tree changes, baked into the
image** (not patched into a running container), and deliberately left uncommitted
pending an operator decision. `patches/README.md` has the mechanism and the
bytecode verification.

## Folder map

| path | what |
|---|---|
| `REPRODUCE.md` | ordered, copy-pasteable: bare nodes → image → gates → sweep |
| `RESULTS.md` | the numbers, with the analysis of each |
| `environment.md` | hardware, fabric, image digests, SHAs, external paths, secrets |
| `notes.md` | index of the traps + the wrong turns, in the order they bite |
| `notes/` | the four long-form investigations |
| `spec/` | the originating task file + the Case A/B operational guide |
| `scripts/` | every script, verbatim |
| `patches/` | the ROCm hicache fix + the Dockerfile layer that bakes it in |
| `results/` | 8 bench JSONs, per-point kvd + server_info, the extracted tables, wire probes |
| `logs/` | engine + sweep + build logs, gzipped |

## Related

- **`agenticbench.mtp.caseA.packup_*`** — the Case A run against this same
  deployment. Independent kit; this one is its prerequisite and shares its
  bring-up.
- `glm52.merged_branch_image.packup_20260801/` — the merged branch and its
  vultr built-image validation.
- `../infera.glm5.2.experiment/agenticbench.glm52.spur.packup_20260731/` — the
  previous spur Case A, **MTP off**, kvaware+kvd-only image.
- `../infera.glm5.2.experiment/kvd.rocm.hostalloc.packup_20260731/` — where the
  ROCm hicache fix was root-caused.
