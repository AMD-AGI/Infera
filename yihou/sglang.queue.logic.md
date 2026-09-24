# sglang PD-disaggregation: queue semantics, pool accounting, and transition gates

Reference notes for reading `Prefill batch` / `Decode batch` log lines and the
`sglang:*` Prometheus metrics in a 1P1D deployment.

All statements below are read from upstream source at
`github.com/sgl-project/sglang`:

- `python/sglang/srt/disaggregation/prefill.py`
- `python/sglang/srt/disaggregation/decode.py`
- `python/sglang/srt/arg_groups/fields/disagg.py`
- `python/sglang/srt/environ.py`

Measured values quoted as illustration come from the T2 sweep
(`yihou/glm52-agentx-t2sweep-simacc.packup_20260920/`), GLM-5.2 P4D4,
CONC 40 → 96.

---

## 1. The two life cycles

### Prefill server — `prefill.py` module docstring, verbatim

```
1. Bootstrap Queue
    a. Initialize a sender for each request
    b. Use the queue to store requests whose bootstrap (handshake and
       preallocation) has not finished
    c. Poll senders to check bootstrap state
    d. Once bootstrap is complete, move request to Waiting Queue

2. Waiting Queue
    a. Use PrefillAdder to pop requests
    b. Run forward
    c. Add the request to Inflight Queue

3. Inflight Queue
    a. Poll (non-blocking) the sender of the request
    b. Once the transfer has finished, return the request
```

### Decode server — `decode.py` module docstring, verbatim

```
1. PreallocQueue:
    a. Initialize a receiver for each request
    b. The request handshakes first, and pre-allocate kv once there is
       available kv.
    c. Move the request to TransferQueue.

2. TransferQueue:
    a. Poll the receiver to check the transfer state
    b. If the transfer has finished, move the request to waiting queue

3. WaitingQueue:
    a. Use the requests in the queue to construct a PrebuiltExtendBatch
    b. Skip the prefill forward but only populate metadata

4. RunningBatch:
    a. Merge the resolved PrebuiltExtendBatch into running batch to run decoding
```

Note 3.b: the decode leg **never runs a prefill forward**. The KV arrives over
the wire; the "extend" batch only populates metadata.

---

## 2. Log field → queue mapping

### `Prefill batch` line

| field | queue | request state |
|---|---|---|
| `#bootstrap-req` | Bootstrap Queue | handshaking, **waiting for the decode side to preallocate** |
| `#queue-req` | Waiting Queue | bootstrap done, waiting for local KV/budget to run forward |
| `#running-req` | — | forward in progress |
| `#inflight-req` | Inflight Queue | forward done, **KV being pushed to decode** |
| `#optimistic-req` | bypass | admitted while still `Bootstrapping` (see §4) |
| `#new-token` / `#cached-token` | — | per-batch prefix-cache split |
| `#pending-token` | — | tokens queued but not yet scheduled |

### `Decode batch` line

| field | queue | request state |
|---|---|---|
| `#prealloc-req` | PreallocQueue | handshook, **waiting for free KV**; holds **no** space yet |
| `#transfer-req` | TransferQueue | space allocated, **KV arriving**; **cannot be retracted** |
| `#queue-req` | WaitingQueue | KV complete, waiting to join the batch |
| `#running-req` | RunningBatch | decoding |
| `#retracted-req` | retracted_queue | evicted under pressure, awaiting `resume_retracted_reqs()` |
| `token usage` | — | fraction of the KV pool in use |
| `pre-allocated usage` | — | fraction of the pool **reserved for TransferQueue** |

### The two legs are coupled pairwise

```
prefill #bootstrap-req   ←─ handshake ─→   decode #prealloc-req
prefill #inflight-req    ←─ transfer  ─→   decode #transfer-req
```

A prefill request leaves the Bootstrap Queue only when the decode side reaches
`KVPoll.WaitingForInput`, i.e. **after decode has preallocated**. Decode KV
pressure therefore shows up directly as prefill bootstrap latency.

---

## 3. The KV-holding set

Only three decode queues occupy KV:

```
KV-holding set = #transfer-req + #queue-req + #running-req
```

`#prealloc-req` is *waiting for* space and holds none — "pre-allocate kv **once
there is available kv**".

Because per-request footprint is fixed by the workload, the holding set is
capped by `pool_size / tokens_per_request`. Measured on GLM-5.2 P4D4:

| CONC | holding set /rank | run | transfer | prealloc (waiting) | token usage |
|---|---|---|---|---|---|
| 40 | 5.58 | 3.87 | 1.71 | 0.00 | 0.432 |
| 56 | 10.60 | 3.73 | 6.87 | 0.57 | 0.751 |
| 72 | 12.58 | 2.58 | 10.00 | 6.17 | 0.840 |
| 96 | **12.16** | **1.72** | **10.43** | **14.58** | 0.833 |

Client concurrency ×2.4 from 40→96, holding set ×2.18 and **flat from 56 on**.
Overflow lands in PreallocQueue (×3,446). Within the capped set the running
share falls **69% → 14%** — the cap is consumed by requests that are merely
receiving KV, which is why decode compute goes idle without any queue building
in `#queue-req` (measured 0 throughout).

### Reading the counters

| rising counter | bottleneck |
|---|---|
| `#prealloc-req` | decode KV pool too small |
| `#transfer-req` | transfer/network slow |
| `#queue-req` (decode) | decode compute |
| `#running-req` at `max_running_requests` | genuine compute saturation |

---

## 4. Transition gates

### PreallocQueue → TransferQueue — the only real admission control

`decode.py: pop_preallocated()`, FIFO. Per request:

```python
required_tokens_for_request = required_alloc_tokens + num_reserved_decode_tokens
full_required_for_admission = max(
    required_tokens_for_request,
    origin_input_len - prefix_len + min(max_new_tokens, CLIP_MAX_NEW_TOKEN),
)
```

checked against `_allocatable_token_budgets()`:

```python
allocatable_tokens = available_size - max(reserved_tokens, need_space_for_single_req)
```

where

```python
reserved_tokens          = num_reserved_decode_tokens * n_active
need_space_for_single_req = max over running reqs of (
    min(max_new_tokens, CLIP_MAX_NEW_TOKEN)
    + len(origin_input_ids)
    - retractable_tokens                      # Σ over running (input + output)
)
```

and, only when decode radix cache is enabled:

```python
if get_disagg().disaggregation_decode_enable_radix_cache:
    available_size += self._radix_full_evictable()
```

**Why `need_space_for_single_req` exists** — comment in `pop_preallocated()`:

> We need to make sure that the sum of inflight tokens and allocatable tokens is
> greater than maximum input+output length of each inflight request. Otherwise
> it is possible for one request running decode out of memory, while all other
> requests are in the transfer queue that cannot be retracted.

It is deadlock avoidance, and it is **unconditional** — no flag disables it.
Note it enters via `max(...)`, not a sum, and `retractable_tokens` is a *sum*
over running requests while `len(origin_input_ids)` is a *single* one, so with
≥2 running requests the expression goes negative and `reserved_tokens` wins.
In the T2 run (`n_active ≈ 12`) the binding term was `512 × 12 = 6,144` tokens
= **0.28%** of a 2,226,688-token pool — effectively inert. Not worth disabling.

### Other transitions — polling only, no admission gate

| transition | gate |
|---|---|
| TransferQueue → WaitingQueue | `poll() == KVPoll.Success` |
| WaitingQueue → RunningBatch | merged as `PrebuiltExtendBatch`; **prefill forward skipped** |
| RunningBatch → retracted | out of space during decode |
| retracted → RunningBatch | `resume_retracted_reqs()` when space returns |

### Prefill Bootstrap Queue → Waiting Queue

`prefill.py: pop_bootstrapped()`. Two accepting states:

- `KVPoll.WaitingForInput` — decode has preallocated; normal path, calls
  `finalize_bootstrap(req)`.
- `KVPoll.Bootstrapping` — **optimistic path**, taken only while
  `req.prefill_attempt_count < get_disagg().optimistic_prefill_attempts` and the
  request is not retracted, and only if `ensure_metadata_buffer(req)` succeeds.
  Lets prefill start computing before decode has confirmed space.

`KVPoll.Failed` → `handle_bootstrap_failure(req)`.

Measured in T2: `#optimistic-req` was **0 at every point**, so the optimistic
path never engaged and every request waited for decode preallocation.

---

## 5. Knobs

| knob | where | default | effect |
|---|---|---|---|
| `--num-reserved-decode-tokens` | `arg_groups/fields/disagg.py` | **512** | KV reserved per request admitted to the running batch |
| `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` | env, `environ.py` | **4096** | caps the `max_new_tokens` term in admission and in `need_space_for_single_req` |
| `--disaggregation-decode-enable-radix-cache` | disagg args | **false** in T2 | when on, evictable radix entries count toward `available_size` |
| `--disaggregation-decode-extra-slots` | disagg args | 0 / 2× running batch | extra `req_to_token` slots for in-transfer requests |
| `optimistic_prefill_attempts` | disagg config | — | attempts allowed on the optimistic bootstrap path |

### Sizing arithmetic (GLM-5.2 `glm_moe_dsa`, 78 layers, `fp8_e4m3`)

| component | per layer | × 78 |
|---|---|---|
| MLA latent (`kv_lora_rank` 512 + `qk_rope_head_dim` 64) | 576 B | 44,928 B |
| DSA index cache (`index_head_dim` 128) | 128 B | 9,984 B |
| **total** | 704 B | **54,912 B/token** |

Measured `114.52 GiB / 2,226,688 tokens = 55,225 B/token` — within **0.57%**.

With the T2 workload (ISL p50 ≈ 109k, OSL p50 ≈ 400) one request costs ~6 GB of
KV, so a 114.52 GiB pool holds ~20 p50-sized requests, ~13 at the observed
length-biased mix, ~9.5 at p90. Across 4 ranks that is a **decode ceiling of
roughly 52 concurrent requests** — which is why CONC=40 fits, CONC=56 is the
first point to show `#prealloc-req > 0`, and CONC=96 stalls.

---

## 6. Caveat — decode-side Prometheus gauges were frozen

In the T2 run every `sglang:*` series scraped from the **decode** endpoint had
`std = 0` across 46,322 scrapes (`num_running_reqs` reported 0.00 for the whole
hour while the log showed 1.4–4.9). Prefill series on the same export varied
normally. Cause not established; the endpoint *was* refreshing
(`unique_updates == total_fetches`) and the frozen value differed between runs,
so it is not frozen since boot.

**Read decode-side load from the `Decode batch` log lines, not from
`/metrics`.** Verify with `std`/`min`/`max` before trusting any decode gauge.
