# Phase 0 — every feature proven ON, with the check that would go red

Task spec item 1: *"before benching, verify each feature is genuinely enabled."*
A green run proves little on this stack, so each row below is the check that
would have gone **red** had the feature been absent or misconfigured.

## The image (both nodes)

Built from the branch on **each** node — not built once and shipped, because the
claim under test is that the Dockerfile reproduces the deployment.

| node | role | image id |
|---|---|---|
| `crsuse2-m2m-253` | prefill | `sha256:42a303e5…` |
| `crsuse2-m2m-236` | decode | `sha256:ff7b02eb…` |

**The two ids differ, and that is expected** — each node built independently, so
Rust objects and layer timestamps differ. Equivalence is established by the
bytecode gate below, never by comparing digests.

`BYTECODE_GATE OK` on both, 8/8 assertions, each read from **freshly compiled**
bytecode rather than from source (a stale `__pycache__` entry silently running
unpatched code has invalidated a full experiment on this stack twice):

| assertion | prefill | decode |
|---|---|---|
| ROCm hicache host alloc (`GLM52_ROCM_HOST_ALLOC`) | 1 | 1 |
| mooncake early-send — `conn.py` / `utils.py` / `prefill.py` | 2 / 1 / 1 | 2 / 1 / 1 |
| **behavioural**: `ALLOC_MEMORY_FUNCS["cuda"] → alloc_with_pin_memory` under HIP | OK | OK |
| DSA p1 `_p1v2_trim` | 1 | 1 |
| DSA p2b `_glm52_match_page_table_rows` | 1 | 1 |
| DSA p3 `requires_dp_attention_eager_forward` | 2 | 2 |

The behavioural row matters more than the marker rows: a marker proves the file
was edited, the dispatch check proves the edit does what it claims.

## The leg gate

| row | prefill | decode |
|---|---|---|
| `ready to roll` | 1 | 1 |
| `Memory access fault` | **0** | **0** |
| `Scheduler hit an exception` | **0** | **0** |
| `Traceback` | **0** | **0** |
| `Errno 98` *after* the ready line | 0 | 0 |
| `context_length=262144` | ✓ | ✓ |
| `dp_size=8` | ✓ | ✓ |
| `speculative_algorithm='EAGLE'` | **0** (correct) | **1** |
| `disable_custom_all_reduce` | False | **True** |
| `enable_hierarchical_cache` | **True** | False |
| live `sglang::scheduler_DP` procs | 8 | 8 |
| both workers in `/v1/workers` | ✓ | ✓ |

Cold start: prefill 315 s, decode 606 s. The decode leg is slower by design —
it also loads the EAGLE draft model and captures its draft graphs.

`Memory access fault: 0` on the prefill leg is itself a result: that is the
failure the ROCm hicache patch exists to prevent, at 120K-token prompts with
kvd on, in the exact regime that faulted before the fix.

## The five discriminating proofs

### 1. PD over mooncake RDMA — would go red on a silent TCP fallback

`MC_FORCE_TCP` **0** on both legs; `mlx5_0` present **26×** per leg;
`MOONCAKE_DISABLE_HIP_DMABUF=0`. Spur has no peermem, so dma-buf via mlx5 is the
only GPUDirect path — a misconfiguration here drops to TCP, which is *correct
but slow* and looks completely healthy.

### 2. DP-attention — would go red at `dp_size=1`

`enable_dp_attention=True`, `dp_size=8`, and **8 live scheduler processes** per
leg. The process count is the part that cannot be faked by an arg echo.

### 3. MTP / EAGLE — would go red with no acceptance signal

`speculative_algorithm='EAGLE'`, `num_steps=3`, `eagle_topk=1`,
`num_draft_tokens=4`, on the **decode leg only**; prefill carries none.

Server-reported **`avg_spec_accept_length = 2.749`** — in the healthy 2.1–2.6+
band, *not* 4.00. That distinction is load-bearing: `accept len: 4.00` means the
draft model predicted every token, which happens when the output is a **loop**.
It is a symptom, not a health signal. Measured here, `accept len` is bimodal:
**1.75–2.80** on healthy requests, **3.85–4.00** on the looping ones described
in `needle_resolved.md`.

### 4. kv-aware routing — would go red at cache view 0

Router policy `kv-aware`, weights `prefill 20.0 / decode 2.0`.
`infera_policy_cache_view_size` on the prefill worker: **14,723 blocks**.

But the *discriminating* number is not that it is non-zero — it is an exact
cross-check between two independent subsystems:

| router pick log | engine `usage` | check |
|---|---|---|
| `cache_hits=937` | `cached_tokens=59,968` | 937 × 64 = **59,968** ✓ |

The router's own block hashes chain to the same prefix the engine independently
reports as cached, to the token. A router hashing a wrong view could not produce
that identity.

**Correction carried:** the bigram code path is **not exercised** here — measured
on the wire, the prefill leg emits plain ints. See `bigram_not_exercised.md`.
This run's kv-aware evidence is about the plain-int path only.

### 5. kvd serving, not merely wired — the hardest one

`gets 0 → 11,250`, `hits 0 → 11,250`, `misses 0`, **`sets` flat**, and the read
count matches the replayed volume exactly (360,000 tok ÷ 64 × 2 pools = 11,250).
Full write-up, including a false negative that had to be diagnosed with an
instrumented round: `kvd_serving_proof.md`.

## kv-aware weights — the setting, and why (spec item 1.1)

Kept at the product defaults, `--kv-prefill-overlap-weight 20.0` /
`--kv-decode-overlap-weight 2.0`, justified by arithmetic rather than by a sweep.

The policy minimises `cost(w) = w·(request_blocks − hits) + active_blocks`
(`infera/router/policy/kv_event_aware.py`). At Case A's 0.89 hit rate with
`page_size 64`:

| ISL | blocks | hit blocks | miss blocks | prefill overlap term (w=20) |
|---:|---:|---:|---:|---:|
| 74,000 | 1,156 | 1,028 | 128 | 2,560 |
| 155,000 | 2,421 | 2,154 | 267 | 5,340 |
| 235,000 | 3,671 | 3,267 | 404 | 8,080 |

against an `active_blocks` load term in the low hundreds — locality dominates by
roughly 10×, which is the regime these weights are designed for. The asymmetry
is principled: prefill workers are compute-bound (a hit skips an entire prefill
pass) while decode workers are memory-bound on KV (a prefill-time hit does not
help the decode loop, so routing by load keeps latency consistent).

A weight sweep would cost a full run per point and perturb the very measurement
it informs; the spec asks for weight evidence only where it does not affect
performance. So instead the run **records** `infera_router_pick_cache_hits`,
`infera_router_pick_request_blocks` and `infera_policy_cache_view_size` from
`/metrics` at zero cost — which is what produced the exact cross-check above.

## Correctness

Short factual **4/4**. Needle 4/5 → investigated and closed as a **sampling
excursion, not KV corruption**: the identical prompt against an identical warm
cache passes 3/6, which corruption cannot do. The load generator sends uniformly
random ASCII with `ignore_eos: True` and grades nothing, so this cannot touch any
benchmark number. Full analysis: `needle_resolved.md`.

Raw evidence: `results/feature_evidence_g1.txt`, `results/wire_*.txt`,
`results/kvd_*.json`, `logs/`.
