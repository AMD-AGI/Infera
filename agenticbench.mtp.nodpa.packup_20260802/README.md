# noDPA — what prefill DP-attention costs at concurrency 1

**Ran 2026-08-02, 12:41:42–13:14:47 UTC** (1,985.5 s) on the **spur**
(`crsuse2-m2m`) cluster. Same workload lat1 ran; **one server flag moves**:
`--enable-dp-attention` on the prefill leg.

## Goal

lat1 established the concurrency-1 latency floor of the full-feature deployment,
with DP-attention on both legs. This run asks what that flag is *worth* on prefill.
Concurrency 1 is the right place to ask: DP-attention exists to pack independent
requests across ranks, so with exactly one request in flight there is nothing to
pack — what remains is its cost, isolated.

## Result at a glance

**Three arms.** The middle one is the result; the third exists to prove the
comparison is not confounded by chunk size.

| arm | prefill DPA | **global** chunk/step | n | input p50 | **TTFT p50** | TTFT p90 |
|---|---|---:|---:|---:|---:|---:|
| **lat1** | **on** (dp8) | 65,536 | 124 | 83,048 | 2,042 ms | 4,757 ms |
| **noDPA-65K** ← the result | **off** | 65,536 | 115 | 84,404 | **1,162 ms** | **2,227 ms** |
| noDPA-8K (chunk control) | off | 8,192 | 175 | 71,640 | 916 ms | 2,059 ms |

### Bin-matched — DPA costs 1.65–1.93×, chunk costs nothing

| input bin | DPA on / 65K | DPA off / 65K | DPA off / 8K | **DPA effect** | chunk effect |
|---|---:|---:|---:|---:|---:|
| 0–40K | 877 | 455 | 454 | **1.93×** | 1.00× |
| 40–60K | 1,150 | 637 | 633 | **1.81×** | 1.01× |
| 60–80K | 1,674 | 938 | 907 | **1.78×** | 1.03× |
| 80–100K | 2,164 | 1,199 | 1,228 | **1.80×** | 0.98× |
| 100–130K | 2,896 | 1,634 | 1,541 | **1.77×** | 1.06× |
| 130–160K | 3,729 | 2,104 | 2,006 | **1.77×** | 1.05× |
| 160–200K | 4,517 | 2,733 | 2,773 | **1.65×** | 0.99× |

**Three findings worth the run:**

1. **DP-attention has a fixed per-request prefill cost of ~1.65–1.93× at N=1.**
   Marginal prefill rate 59,705 vs 37,736 tok/s. **This does not say DPA is
   slower** — it says DPA must earn this back through concurrency, and this run
   deliberately removed the concurrency.

2. **An 8× change in per-step token budget is invisible at concurrency 1**
   (0.98–1.06×, no direction). A 74K prompt costs the same cut into 2 chunks or
   10: per-step overhead is negligible against the compute, and with one request
   in flight there is nothing else to pack into the spare capacity. This could
   not be assumed — it had to be measured, and measuring it is what licenses
   reading the first row as a DPA effect.

3. **DP-attention off is not free: it costs activation headroom.** This arm
   **cannot boot** at lat1's `--mem-fraction-static 0.80` — it dies with
   `HSA_STATUS_ERROR_OUT_OF_RESOURCES` at `token usage: 0.04`, i.e. with the KV
   pool empty. Ran at 0.70.

> **`--chunked-prefill-size` is per-*global*, and DPA divides it by `dp_size`**
> (`server_args.py:4902` — a division, not a clamp). lat1 requested 65,536 at dp8
> and `server_args=` shows 8,192, but that 8,192 is **per rank across 8 ranks**,
> i.e. 65,536 globally. With DPA off there is no division. Matching the *global*
> per-step budget therefore means passing **65,536**, not 8,192 — the reverse of
> what the `server_args=` value suggests read naively.

> **† TPOT is a weak instrument here and no claim rests on it.** The driver
> records one mean per request (`generation_time / (gen_len - 1)`, where
> `generation_time = total_time - ttft`), so the ladder is over *requests*, not
> tokens; the numerator includes PD handoff and SSE transport; and with MTP
> accepting ~2.9 tokens per verify step the true inter-token interval is bimodal,
> which a per-request mean erases. TPOT p50 11.0 (noDPA-65K) vs 10.66 (lat1) is
> reported as an observation only — see `analysis/nodpa_vs_lat1.md` § TPOT.

## What had to differ, and why it does not explain the result

This arm carries **two** flag differences from lat1, not one. Stated plainly
rather than buried:

| | lat1 | this run | why |
|---|---|---|---|
| prefill `enable_dp_attention` | True | **False** | the variable under test |
| prefill `mem_fraction_static` | 0.80 | **0.70** | **forced** — 0.80 does not boot |

Does GMU explain the TTFT win? No, and it is checkable rather than arguable: GMU
sets how much VRAM is *statically reserved for KV*, and `token usage` peaked at
**0.04** on both arms. The smaller pool (2,821,248 vs 2,939,264 tokens/rank, −4 %)
was never within 25× of binding. A KV pool 96 % idle cannot make prefill 1.7×
faster.

Held fixed on purpose, against the stock leg script which would have moved them:
`--ep-size 8` (MoE expert parallelism — a different axis from attention) and the
effective chunk (8192 both arms).

## Navigation

| path | what |
|---|---|
| **`REPRODUCE.md`** | the ordered, copy-pasteable command sequence |
| **`analysis/nodpa_vs_lat1.md`** | the report: bin-matched tables, fits, outliers, what is not established |
| **`notes/nodpa_design.md`** | why each confounder was pinned, with the source read for each |
| **`notes/notes.nodpa.md`** | what went wrong during the run and what it cost |
| `environment.md` | hardware, fabric, image digests, git SHA, secrets (names only), gaps |
| `spec/nodpa_full.yaml` | the workload — byte-identical to lat1's except `random_seed` |
| `spec/lat1_full.REFERENCE.yaml` | lat1's workload — diff the two |
| `scripts/` | every script that ran, verbatim |
| `patches/` | the ROCm hicache patch + `GLM52_P1V3`, both load-bearing |
| `results/` | `summary.json`, `metrics.jsonl.gz`, kvd before/after, ladders, probe |
| `logs/` | driver transcripts + both legs' tails, gzipped |
| `env/` | live `collect_env.sh` output from both nodes |

## Read this before trusting any percentile

**n = 175** (lat1: n = 124). p50 and p90 are solid on both arms; **p99 is a handful
of observations and should not be read as a point estimate.** The mitigation is the
bin-matched table and the length fit, which use every point at once.

**3 requests (1.7 %) are unexplained TTFT outliers** — 12.2 s, 12.2 s and 14.6 s.
Checked and ruled out: no engine fault, no retraction, no scheduler exception,
`token usage` 0.02–0.04, both returned `200 OK`. No positive cause found; they are
excluded from the headline fit and both fits are published so the exclusion is
visible rather than trusted.

## What this run does not establish

- **Anything about DPA under load.** N=1 removes exactly what DPA optimises. Case A
  (N≈44) is the other side of the trade and was not re-run.
- **The TPOT direction.** Decode was byte-identical on both arms, yet TPOT is 8–16 %
  slower here. Observed, not explained — **and the metric itself is too weak to
  settle it** (per-request mean, wall-clock numerator, MTP bimodality). Settling
  it needs per-chunk SSE timestamps.
- **ITL, at all.** No inter-token series is persisted, and on an MTP decode leg
  ITL is not recoverable from TPOT — accepted tokens arrive at near-zero spacing
  and the gap lives between verify steps.
- **A GMU-matched comparison.** 0.80 does not boot without DPA, so it is not
  reachable. Argued immaterial from `token usage`, not proven by ablation.
- **The 3 outliers' cause.**
- **Multi-turn behaviour**, `turns_per_session: 1` removes it by design.
