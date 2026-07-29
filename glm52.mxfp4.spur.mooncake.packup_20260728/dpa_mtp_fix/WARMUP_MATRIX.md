# Warmup outcome matrix — which runs cleared PD warmup, which hung in it

Question this answers: for every configuration tried, did the decode leg's
PD-disaggregation warmup **complete** or **hang**, and what concurrency does
that warmup actually exercise?

## What the warmup actually is

Source: `python/sglang/srt/entrypoints/http_server.py:2102-2132`.

The decode leg, once the HTTP server is up, sends **one** `/generate` POST to
itself whose payload is a **batch of `dp_size` requests** — for our config,
**8 concurrent requests, one per DP rank**:

```python
json_data = {
    "sampling_params": {"temperature": 0.0, "max_new_tokens": 8, "ignore_eos": True},
    "bootstrap_host": [FAKE_BOOTSTRAP_HOST] * server_args.dp_size,
    "bootstrap_room": [i * (2**63 // server_args.dp_size) + (i % server_args.tp_size)
                       for i in range(server_args.dp_size)],
    "input_ids": [[10, 11, 12, 13]] * server_args.dp_size,
}
```

So the warmup's parallelism is:

| property | value |
|---|---|
| concurrent requests | **8** (`dp_size`) |
| requests per DP rank | 1 — one distinct `bootstrap_room` per rank |
| prompt | 4 tokens (`[10, 11, 12, 13]`) |
| output | 8 tokens, `ignore_eos=True`, greedy |
| transfer | **fake** (`FAKE_BOOTSTRAP_HOST`) — no real RDMA KV pull |
| client timeout | 1800 s |

This matters for interpreting the results below. The warmup is the **only test
in this investigation that loads all 8 DP ranks simultaneously**. Every
post-warmup probe we run through the router is a *single* request that lands on
*one* dp_rank, leaving the other 7 idle.

That is exactly the busy/idle mix that triggers the rank-divergence bug — which
is why the warmup is a sharper test than the 4-request probe, and why "warmup
passed" and "requests hang" can and do coexist.

Note also that warmup uses fake transfer, so a warmup hang is **not** a
mooncake/RDMA problem; it is purely a compute-path deadlock.

## Results

`--disable-cuda-graph` runs are the eager control pair (jobs 10922/10923); the
rest are the graph-enabled pair (jobs 9005/9006).

| # | config | graphs | warmup | post-warmup 4-req probe |
|---|---|---|---|---|
| 1 | Fix A/A2 only | ON | **PASSED** 11:33:32 (8/8 ranks, spec metrics live) | **HUNG** (req1 timeout 180 s) |
| 2 | Fix A/A2 + Bug 4 "uniform event" | ON | **PASSED** | **HUNG** — all 8 ranks in `overlap_utils.py:292`, the wait that fix itself hoisted |
| 3 | Fix A/A2, Bug 4 reverted, probes on | ON | **PASSED** 11:47:34 | **HUNG** (req1+req2 timeout 150 s) |
| 4 | + probe 2 (graph-path logging) | ON | **PASSED** 12:1x | **HUNG** (req1+req2 timeout 100 s) |
| 5 | + Bug 2b naive fix (`not is_idle()` removed) | ON | **HUNG** 12:29:01 | not reached |
| 6 | same, after purging stale `.pyc` | ON | **HUNG** 12:40:14 | not reached |
| 7 | **eager control** (`--disable-cuda-graph`) | **OFF** | **PASSED** 12:28:47 (1 m 56 s) | **4/4 HTTP 200**, plus 2×512-token at 200 |
| 8 | Variant B (draft graph forced off, others ON) | partial | *running* | *pending* |

Runs 1–4 differ only in instrumentation and the (reverted) Bug 4 patch; they are
the same underlying configuration and all show the same signature.

## What the matrix says

1. **Warmup passing does not mean the config is healthy.** Runs 1, 3 and 4
   cleared an 8-way concurrent warmup and then deadlocked on a *single*
   request. The failure needs a sustained busy/idle mix across ranks, which a
   short warmup can get through by luck; over hundreds of iterations the counts
   drift apart and it wedges.

2. **The failure is racy.** The same patch set cleared warmup in runs 1/3/4 and
   hung inside warmup in runs 5/6. A single passing run therefore proves
   nothing — this is why the differential (对拍) method is mandatory.

3. **Only the eager control passed end to end** (run 7): warmup *and* 4×24-token
   *and* 2×512-token requests, with `spec_accept_length > 1` confirming MTP was
   genuinely active. This is the evidence that CUDA graphs cause the hang.

4. **The Bug 2b naive fix made things worse** (runs 5/6): it moved the failure
   *earlier*, from the request path into warmup. Removing the `not is_idle()`
   term did not make the guard rank-uniform — the surviving terms
   (`can_cuda_graph` from `prepare_for_draft`, and `dsa_topk_indices is None`)
   are themselves rank-divergent, measured as dp0 16/3 vs dp7 14/4 graph/eager
   splits *with the fix applied*.

## Methodological note

Run 5 was initially recorded as a valid failure. It was not: `shutil.copy2`
preserved the source mtime, so CPython reused a `__pycache__` entry compiled
from the **unpatched** file and the fix never executed. Confirmed by
`strings eagle_worker_v2.cpython-310.pyc | grep -c GLM52_BUG2B` returning 0.
Run 6 is the valid repeat, with all bytecode purged first. Both hung, so the
conclusion is unchanged — but run 5 on its own was worthless.
