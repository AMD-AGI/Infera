# `--max-running-requests 128` is a GLOBAL cap — it bounds both sweeps

Found 2026-09-18 12:55 UTC while the P8D8 probe deployment was capturing CUDA
graphs. Recorded before it can silently distort the sweep.

## What was observed

The decode engine logged, identically on all 8 DP ranks:

```
Capture target verify CUDA graph begin. backend=full, num_tokens_per_req=6,
bs=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16], avail mem=42.6 GB
```

The captured batch sizes stop at **16**, even though the engine was launched with
`--cuda-graph-max-bs-decode 128`. 16 = 128 / 8 is too neat to be a coincidence.

## Why — read from the source in the running image, not inferred

1. `model_executor/runner/base_cuda_graph_runner.py:64` — `get_batch_sizes_to_capture`
   clamps the list: `capture_bs = [bs for bs in capture_bs if bs <= num_max_requests]`,
   where `num_max_requests = model_runner.req_to_token_pool.size`.
2. `model_executor/pool_configurator.py:847` — that pool is sized

   ```python
   num_reqs = get_schedule().max_running_requests // kvc.ps.attn_dp_size
   ```

   Confirmed by three sibling sites: `pool_configurator.py:966`,
   `mem_cache/kv_cache_configurator.py:2449,2473,2530`, and
   `arg_groups/moe_hook.py:571` (`per_rank_pool_bs = max(1, view.max_running_requests
   // attn_dp_size)`).

So **`--max-running-requests` is a global budget divided across the DP ranks**, and
the CUDA-graph list inherits that per-rank ceiling. `--cuda-graph-max-bs-decode 128`
never had a chance to take effect on the decode leg.

| shape | per-rank ceiling | total concurrent requests the server will run |
|---|---|---|
| P8D8 (dp 8) | 128 / 8 = **16** | **128** |
| P4D4 (dp 4) | 128 / 4 = **32** | **128** |

## Consequence for the planned runs

**The two `fast` points are unaffected.** CONC 80 on P8D8 is 10 per rank and CONC 40
on P4D4 is 10 per rank — both comfortably inside the ceiling, and, usefully, *equal*,
so the screening comparison is not distorted by this at all.

**Both sweeps run into it, and the P8D8 one runs into it early.**

| sweep point | P8D8 per rank | inside? | P4D4 per rank | inside? |
|---|---|---|---|---|
| 80 / 40 | 10 | yes | 10 | yes |
| 112 / 56 | 14 | yes | 14 | yes |
| 144 / 72 | 18 | **NO** | 18 | yes |
| 192 / 96 | 24 | **NO** | 24 | yes |
| 256 / 128 | 32 | **NO** | 32 | **exactly at the cap** |

## Why this matters more than it looks

The mission says a top-point **out-of-memory failure is an acceptable end state**.
That expectation assumes the top point actually stresses the hardware. It would not.
Above 128 total in-flight requests the server simply **queues** — no error, no OOM,
no crash. The benchmark would keep running and report a number, and that number would
describe **the scheduler's cap, not the shape's capability**. A flat or gently
declining tail across the top three P8D8 points would look like a genuine saturation
curve and would be nothing of the sort.

That is the dangerous failure mode here: not a crash, but a plausible wrong answer.

## Options, none yet chosen

1. **Raise `*_MAX_RUNNING` with concurrency** (e.g. to 2x the sweep's top point).
   Costs VRAM for the larger request/token pools and lengthens CUDA-graph capture,
   which eats the same budget the intended OOM end-state is about — so the OOM may
   simply move to a different point, which is arguably the honest outcome.
2. **Cap the sweeps at the last point inside 128** — P8D8 at 112, P4D4 at 128 — and
   report the ceiling as the reason rather than pretending to a wider curve.
3. **Run the top points anyway and label them** as cap-limited. Cheapest, and
   honest, but spends hours of cluster time on points that measure the cap.

**Not decided. To be raised with the user once the branch is known**, together with
the measured `fast` numbers, since the right answer differs for P8D8 (hits the cap at
its 3rd point) and P4D4 (only reaches it at the last).
