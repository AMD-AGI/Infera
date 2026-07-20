# serve_standard — canonical cross-team serving benchmark

`bench_vllm_serve.sh` is the **single tool we report externally**. It wraps
`vllm bench serve` (the de-facto industry-standard serving benchmark) so our
numbers line up with what other teams quote.

## Why not `bench/kvcache/kv_cache_cliff/run_cliff.py`?

`run_cliff` is an **internal** cliff-detection / A/B tool. For the same server
and load it reports throughput **systematically ~25% higher** than
`vllm bench serve`, because:

- its throughput = tokens / `asyncio.gather` window only (prompts are pre-built
  before the timer starts → tightest possible wall), takes the **median of warm
  iterations**, and excludes errored requests from the token sum;
- it **always warms the cache first**, so it only ever measures the
  warm / cache-hit regime — never a cold prefill.

`vllm bench serve` instead measures tokens / full request duration (includes
client-side tokenization, dispatch, tail straggler and failed-request penalty)
and reports a single run. Neither is "wrong", but only one is comparable to
other teams — so we standardise on `vllm bench serve`.

## Protocol

- **dataset**: `random`, fixed `--random-input-len` (ISL) / `--random-output-len` (OSL)
- **per-round seed = the concurrency value N** (run1 and run2 of a given N share
  the seed; different N values get different seeds). Each N run **twice**:
  - **run1 = COLD** — no cache; real prefill compute throughput (baseline)
  - **run2 = WARM** — identical prompts → prefix / L3 cache hit; the KV-offload win
  - *Why per-round seed:* with one fixed seed, `vllm bench serve` generates
    prompts deterministically, so a larger N's prompt set is a **superset** of a
    smaller N's. Running N=256 then N=192 would leave N=192's prompts already
    cached → its "cold" run is actually warm (we hit exactly this: N=192 cold
    read 159k instead of ~25k). A distinct seed per N keeps each prompt set
    disjoint, so run1 is genuinely cold. Override with `SEED=<int>` only if you
    deliberately want one fixed seed.
- **concurrency == num-prompts** (one wave), swept over a fixed set
- model / GPU / image tag / vllm + hipfile versions / server config → `manifest.txt`

## Usage

```bash
# from the repo root, inside the engine container, against a running server:
BASE_URL=http://localhost:8833 \
IMAGE=infera/engine-vllm:dev \
GPU=7 \
  bash bench/serve_standard/bench_vllm_serve.sh
```

Env overrides: `BASE_URL MODEL TOKENIZER ISL OSL SEED CONCURRENCIES IMAGE GPU OUTDIR`.
Defaults: `ISL=60000 OSL=1 SEED=42 CONCURRENCIES="16 32 64 128 196 256"`.
`TOKENIZER` auto-resolves to the server's local model path if unset.

## Output (`bench_results/vllm_serve_<UTC-stamp>/`)

| file | contents |
|---|---|
| `manifest.txt` | full run metadata (reproducibility) |
| `summary.csv` | parsed `concurrency,run,phase,successful,failed,duration_s,total_tok_s,mean_ttft_ms,p99_ttft_ms` |
| `raw_N*_run*.json` | raw `vllm bench serve` result per point |

## What to report

Quote **two numbers**, both labelled with the protocol (model / ISL / OSL / GPU / image):

- **Cold (run1)** — real no-cache prefill throughput
- **Warm-L3 (run2, concurrency > L1 capacity)** — the KV-offload cache-hit
  throughput; the feature's headline
