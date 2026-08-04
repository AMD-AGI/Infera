# AgentX Case-A with prefill DP-attention ON — router policy A/B

**Ran:** 2026-08-04 04:42 – 06:03 UTC (single session)
**Status:** **PARTIAL** — both legs of the A/B produced clean data, but the
`kv-aware` leg ran with an **empty cache view** (`cache_hits=0` on every
request), so the policy under test was never actually exercised. The
**deployment** change is measured; the **policy** change is not.

Re-runs the same customer benchmark as
[`../agentx.caseA.customer.packup_20260803`](../agentx.caseA.customer.packup_20260803)
(ROCm/MAD PR #173, aiperf open-loop frozen trace, unmodified) against the same
two nodes, with the prefill leg reconfigured.

## What changed vs the 20260803 run

| knob | 20260803 (par8 posture) | here |
|---|---|---|
| prefill DP-attention | **off** (`dp_size=null`) | **on** (`--dp-size 8 --enable-dp-attention`) |
| prefill `mem-fraction-static` | 0.80 | **0.70** |
| prefill per-forward prefill work | 16384 | **8192** (see below) |
| prefill delayer | absent | **present** (returns with DPA — see `notes.md`) |
| router policy | kv-aware, pw=20 dw=2 | **round-robin**, then **kv-aware pw=2 dw=2** |
| decode leg | DPA on, gmu 0.85, MTP | **unchanged** |

**Four things moved at once relative to 20260803.** No single-knob attribution
is possible from this data and none is claimed.

## Goal and success criteria

This run has **no written spec.** It was an operator instruction, given verbatim
as:

> 你再跑一遍这个external测试吧，这次把prefill的dpa打开，gmu=0.7, kv-aware调度用
> 你自己分析之前的agentx负载分布结合infera的，设一个合适数值。只跑c=8

It extends the task in [`spec/external.agentic.bench.md`](spec/external.agentic.bench.md)
(the originating brief, copied here), which set **no numeric bar** — it asked
for the customer bench to run and be analysed. The criteria below are therefore
the instruction's own four deliverables.

| # | criterion | actual | verdict |
|---|---|---|---|
| 1 | prefill DP-attention **on** | `dp_size: 8`, 8 schedulers, verified in `/v1/workers` and `ps` | **PASS** |
| 2 | prefill `gmu = 0.7` | `--mem-fraction-static 0.70`, verified in the resolved cmdline | **PASS** |
| 3 | derive a suitable kv-aware weight from the AgentX load distribution | `pw=2.0`, derived in [`analysis/overlap_weight_derivation.md`](analysis/overlap_weight_derivation.md) from the cost function + all 1,778 trace requests | **DERIVED, NOT VALIDATED** — the run it was meant to test had an empty cache view |
| 4 | run c=8 only | 135 profiling requests, 915 s, 0 errors | **PASS** |

Criterion 3 is the reason this kit's status is PARTIAL. The number was computed
as asked and the deployment ran as asked; what failed is that the policy it
configures never received a cache view to act on.

## Results — concurrency 8

All three columns are the same estimand computed by the same `analyze.py`,
profiling phase only, cancelled excluded.

| | 20260803<br>DPA off, pw=20 | **round-robin**<br>DPA on | **kv-aware pw=2**<br>DPA on |
|---|---|---|---|
| profiling requests | 231 | 136 | **135** |
| window | 901 s | 903 s | **915 s** |
| **errors / cancelled / ctx-overflow** | 0 / 0 / 0 | 0 / 0 / 0 | **0 / 0 / 0** |
| in-flight max / mean | 8 / 5.1 | 8 / 6.05 | 8 / **6.38** |
| **TTFT p50** | **5,146 ms** | 22,236 ms | **25,188 ms** |
| TTFT p90 | 19,780 ms | 42,693 ms | 69,938 ms |
| **ITL p50** | **13.8 ms** | 19.46 ms | **14.85 ms** |
| E2E p50 | 12,556 ms | 34,966 ms | 36,135 ms |
| throughput | 0.256 req/s | 0.151 req/s | 0.148 req/s |
| **cache hit, token-weighted** | 50.3 % | 12.4 % | **27.4 %** |
| cache hit, per-request p50 | 88.1 % | 55.7 % | 69.1 % |

A c16 round-robin leg also ran but is **not comparable** — its window was
308 s, not 900 s, and a prefill rank had already died by then. It is archived
in `results/c16_roundrobin/` for completeness only.

## ⚠ The kv-aware leg did not test kv-aware

Read this before quoting the third column as a policy result.

From `env/router.kvaware_pw2.log.gz`, over the whole run, for the 37 prefill
picks that carried a non-empty prompt:

```
mean request_blocks : 1436     <- correct, matches the trace's 1,315
mean cache_hits     :    0.0   <- every single pick
max active_blocks   :    0     <- every single pick
```

With both terms of the cost function at zero the policy is a no-op: `min()`
returns the first target every time, and the observed rank skew
(dp0 65 / dp1 45 / dp2 26 / dp3 10 / dp4 4 / dp5 3) is that stable ordering
plus whatever the engine's own dispatch did — **not** the load/locality
trade-off the weight was chosen to produce.

What was ruled out, with evidence:

| hypothesis | check | result |
|---|---|---|
| router subscribed to the wrong port | `/v1/workers` endpoint vs the leg's own `tcp://*:N` | both **26803** — match |
| the per-rank publishers never bound | `ss -ltn` inside the container | **26803–26810** all listening, 8/8 |
| the ZMQ link is down | `ss -tn \| grep 26803` | **ESTAB** both directions |
| the engine has no cache to report | `#cached-token>0` in prefill batches | **60 / 1129** batches — non-zero |
| kvd (L3) is dead | `statctl` before → after | gets **576 → 1,864**, hits == gets |

So the engine *is* caching and the transport *is* up, yet the router's view
stays empty. **The cause is not determined.** The next probe is whether the
prefill leg emits `BlockStored` at all (its log shows `kv_events` only twice,
both at startup) and, if it does, whether the router's `BlockHasher` produces
the same hashes the engine publishes.

**Consequence: `pw=2.0` is untested.** The derivation behind it still stands
(`analysis/overlap_weight_derivation.md`) but no measurement here confirms or
refutes it.

## Two engine defects found and fixed

1. **`free_tcp_port_block` raises on a node whose ephemeral range starts at
   1024** — `ValueError: empty range for randrange() (1024, 1017, -7)`. Only
   `dp_size > 1` reaches this code, so prefill DPA-on could not start at all on
   chi2835 while DPA-off was fine. Patch + regression tests in `patches/`.
   This also retroactively explains the failed DPA-on attempt of 2026-08-02
   (`p8_prefill.log`).
2. **chi2835's `ip_local_port_range` was `1024 65535`** while chi2879 / chi2867 /
   chi2872 all read the kernel default `32768 60999`. Even with the patch above,
   a block scanned from *inside* the ephemeral range loses a port to the kernel
   between the probe and the engine's bind — measured: base 37059, DP7 died on
   37066 with `ZMQError: Address already in use`. Reset to the default;
   **runtime only, reverts on reboot** (`env/sysctl_change_chi2835.txt`).

## One crash, mid-run, unresolved

During the round-robin leg, prefill **DP6 aborted** with
`HSA_STATUS_ERROR_OUT_OF_RESOURCES` at 05:04 — with `token usage: 0.07`, i.e.
the KV pool nearly empty. This is DP-attention *activation* memory, not KV
exhaustion (same family as par8 `start_leg.sh:54-70`, but that was cured by
lowering gmu and this was already at 0.70).

Mitigated for the kv-aware leg by halving per-forward prefill work
(CHUNK 131072 → 65536, i.e. 16384 → 8192 after sglang's `//dp_size`), which did
not recur. **Whether 16384/forward is simply unusable at gmu 0.70 under DPA, or
whether this was load-dependent, is not established.**

## Navigate

| file | what |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | ordered, copy-pasteable steps |
| [`environment.md`](environment.md) | hardware, images, digests, resolved cmdlines |
| [`notes.md`](notes.md) | traps, wrong turns, what this run could not answer |
| [`analysis/overlap_weight_derivation.md`](analysis/overlap_weight_derivation.md) | **how pw=2.0 was derived** from the cost function + the trace |
| [`analysis/sli_percentiles.md`](analysis/sli_percentiles.md) | full ladders, TTFT-by-size, both legs |
| [`analysis/policy_ab.md`](analysis/policy_ab.md) | round-robin vs kv-aware, and why the A/B is inconclusive |
| `results/c8_kvaware_pw2/`, `results/c8_roundrobin/` | **primary artifacts** — one JSON per request |
| `patches/` | the `net.py` fix + its regression tests |
| `env/` | router logs (all three policies), kvd before/after, cmdlines, the sysctl change |
| `scripts/` | everything that ran; the customer's script is in `spec/`, unmodified |

## Deployment under test

```
router   http://10.2.122.78:8100     served-model-name glm5.2-mxfp4
prefill  chi2835  TP8, DPA 8, chunk 65536 (->8192/forward), gmu 0.70, ctx 262144, kvd + hicache
decode   chi2879  TP8, DPA 8, MTP (EAGLE, 4 draft), gmu 0.85, ctx 262144
```
