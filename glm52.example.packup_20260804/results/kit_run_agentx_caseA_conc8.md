# This kit, measured — AgentX Case-A at concurrency 8

The other two files in this directory report runs against the deployment *shape*
this kit encodes, launched by the per-leg scripts it was refactored from. This one
reports a run against **the kit itself**: `cluster/cluster.dmabuf.sh up`, nothing
edited outside that wrapper, then the customer's harness pointed at the resulting
router.

Its value is not a new performance number — it is that the kit reproduces the
recorded ones. Read it as a regression check on the packaging.

## Setup

Two MI355X nodes on a **mode-B fabric** (no peer-memory module, one ODP NIC at
200 Gb/s alongside eight faster non-ODP rails that dma-buf cannot safely use).
`preflight_rdma.sh mode` picked that mode and named the NIC and GID index; the
wrapper was filled in from its report and nothing else.

Load came from the customer's `replay_caseA.sh` at
[ROCm/MAD PR #173](https://github.com/ROCm/MAD/pull/173), **unmodified** —
`md5sum` checked against the upstream blob before the run — driven by environment
only, exactly as [the pointing-it-here table](customer_agentx_caseA_conc8.md#pointing-it-at-this-deployment)
describes. aiperf ran on a third node so load generation did not compete with the
deployment for CPU.

## Results

| | this kit | previously recorded, same fabric class |
|---|---|---|
| profiling requests | 235 | 231 |
| window | 954 s | 907 s |
| **errors / cancelled / context overflows** | **0 / 0 / 0** | 2 / 0 / 0 |
| request rate | 0.246 req/s | 0.255 req/s |
| output token rate | 262 tok/s | 265 tok/s |
| **TTFT p50** | **6,715 ms** | 6,698 ms |
| TTFT p90 | 24,394 ms | 23,871 ms |
| TTFT p99 | 38,348 ms | 33,972 ms |
| **E2E p50** | **15,783 ms** | 13,874 ms |
| E2E p90 | 42,825 ms | 42,501 ms |
| **ITL p50** | **14.0 ms** | 13.26 ms |
| ITL p90 | 19.2 ms | 18.52 ms |
| ISL p50 / mean | 70,633 / 80,802 | 69,911 / 80,811 |
| OSL p50 / mean | 206 / 1,061 | 230 / 1,050 |
| **server-reported cache hit** (per-request p50) | **88.1 %** | 88.1 % |

Every ladder is recomputed from aiperf's raw `profile_export.jsonl`, not copied
from the summary line.

TTFT is server-side: `http_req_sending` p50 is 0.18 ms.

## What it independently confirms

**1. The cache-rate methodology.** Only **175 of 235** records carry
`usage_prompt_cache_read_tokens`. Taking the ratio per request and then the median
gives **88.1 %**; summing both fields over the unequal sets gives **48.9 %**. The
gap is an artifact of the missing records, not a cache result — the same trap the
[customer-bench notes](customer_agentx_caseA_conc8.md#two-cache-numbers-and-only-one-of-them-measures-the-server)
call out, reproduced here on fresh data.

**2. The prefix cache is worth ~2.9× on TTFT**, priced by turn index:

| | n | ISL p50 | TTFT p50 |
|---|---|---|---|
| first turn (cold) | 55 | 76,997 | **15,378 ms** |
| turn ≥ 1 (cached) | 180 | 67,641 | **5,333 ms** |

2.9× faster on a prompt that is 12 % *smaller*. The earlier run measured 2.5× on
the same trace — same direction, same order.

**3. At c8 the deployment is computing, not queueing.** TTFT p50 by input bucket
is monotone with no stall bucket:

| 0–50K | 50–100K | 100–160K | 160–220K | 220–300K | spread |
|---|---|---|---|---|---|
| 3,553 | 5,077 | 11,240 | 19,174 | 21,584 | **6.1×** |

Prefill-shaped, matching the recorded c8 curve. The flattening that locates the
usable concurrency limit appears at c16, which this run did not cover.

**4. MTP stays healthy under real agentic load** — acceptance length 2.25–2.90 on
the decode leg during the run. Not the degenerate steady 4.00.

## Long context, separately

`smoke`'s completion check uses a 25-token prompt. That proves the service answers;
it does **not** reach the sparse-attention path that Note 5's DSA env block exists
to correct, because at 25 tokens there is nothing sparse to index.

A needle-in-a-haystack pass over the same deployment closes that gap — **12/12**,
at depths 10 / 50 / 90 % across 8K, 32K, 127K and 238K-token prompts, the last
within 9 % of the configured 262,144 context. Requests went through
`/v1/chat/completions` so the model's own chat template applied, at the
checkpoint's `generation_config.json` defaults (`temperature 1.0`, `top_p 0.95`);
the haystack is varied prose rather than a repeated token, which a sparse indexer
would compress trivially.

## What this run does not cover

- **The peer-mem wrapper** (`cluster.peermem.sh`) — no such cluster was available.
- **`preflight_rdma.sh fabric`** — only `mode` was exercised.
- **`round-robin` routing**, and therefore the GMU coupling documented in the
  [infera-bench notes](infera_agenticbench_conc8.md). The shipped `kv-aware`
  default is what ran.
- **c16**, and any concurrency other than 8.
