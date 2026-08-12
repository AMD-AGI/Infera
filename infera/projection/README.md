# Infera Projection & Tuning

Inference/serving **performance projection** and an **LLM tuning agent** for
recipe search, ported from Primus. The core idea: *measure sparsely, transport
analytically*. Benchmark one cheap sub-scale anchor on a single GPU, then
project TTFT / ITL / throughput / KV-cache for every serving recipe
(TP/EP/PP, batch, concurrency, dtype) — no full-node sweep per recipe.

## What's here

- `cli.py` — `infera-projection` entry point (inference/serving projection).
- `_vendor/` — the vendored Primus projection engine and tuning agent. Imports
  are rewritten to `infera.projection._vendor.primus.*`; nothing depends on an
  installed `primus`. The Megatron/training closure is intentionally **not**
  vendored (the inference path never needs it).
- `examples/exp_pretrain.yaml` — an example model/experiment config. The model
  preset is selected via `PRIMUS_MODEL` and resolved from the vendored
  `_vendor/primus/configs/` preset tree.

## Install

```bash
pip install ".[projection]"          # projection only
pip install ".[projection-tuning]"   # + the DSPy tuning agent
```

`torch` and the serving engine (`vllm`) come from the engine base image, not pip.

## Project a recipe from a saved anchor (no GPU)

```bash
PRIMUS_MODEL=gpt_oss_120B PRIMUS_TP=2 PRIMUS_EP=2 \
infera-projection inference \
  --config infera/projection/examples/exp_pretrain.yaml \
  --inference-mode performance --serving-model continuous \
  --input-len 1024 --output-len 1024 --inference-batch-size 32 \
  --gpu-arch mi355x --hbm-capacity-gb 288 \
  --load-benchmark anchor.json
```

## Harvest a GPU-calibrated anchor (needs a ROCm GPU + vLLM)

```bash
infera-projection anchor --model gpt_oss_120B --benchmark-gpus 1 --save anchor.json
```

The anchor JSON is engine-neutral (`"backend"` field + per-batch decode/prefill
measurements), so an SGLang/ATOM harvester on Infera's own engine layer can be
added later without touching the projector.

## Prefix-cache reuse (`--prefix-cache-hit-rate`)

Agentic and multi-turn traffic shares long prefixes (system prompts, tool
schemas, conversation history) that Infera / vLLM / SGLang keep resident via
automatic prefix caching. Model that reuse with `--prefix-cache-hit-rate R`
(alias `--prefix-hit-fraction`), `R` in `[0, 1)`:

```bash
infera-projection inference ... --input-len 4096 --prefix-cache-hit-rate 0.8
```

Semantics: the cached prefix (`R * input_len` tokens) skips prefill compute;
only the non-cached suffix is run through the network, and it still attends over
the full context (cached prefix + suffix). So **TTFT and the prefill share of
continuous-batching pollution scale with `(1 - R)`**, while decode / KV sizing
are unchanged. `R = 0` (default) is a cold cache and is identical to prior
behaviour. At least one token is always prefilled, so `R` is clamped below 1.

Example (gpt_oss_120B, TP2, input=4096, batch=32, MI355X, analytical):

| hit rate | TTFT |
| --- | --- |
| 0.0 (cold) | 52.5 ms |
| 0.5 | 44.0 ms |
| 0.8 | 38.8 ms |
| 0.95 | 23.5 ms |

Works in both **analytical** and **benchmark-calibrated** (`--load-benchmark`)
mode. Analytically the suffix is re-projected over the full context; with a
measured anchor the discount is a proportional token scaling of the measured
prefill (prefill is ~linear in tokens), so `hit=0.8` gives 5x lower measured
TTFT continuously from `hit=0`. The multi-instance DES below also honors a
loaded anchor — its step durations already come from the (measured) cost kernel.

The discount shrinks each recipe's prefill/TTFT component monotonically, so
TTFT-ordered rankings are stable in the common case — though compute-bound
recipes benefit more than comm-bound ones, which is the intended (physical)
effect. Set it per workload in a YAML `inference:` block
(`prefix_cache_hit_rate: 0.8`) so the tuning agent scores every recipe under the
same reuse assumption.

## Multi-instance KV-aware routing (DES)

The discrete-event simulator can model a **fleet of `N` engine replicas behind a
router** sharing a **prefix pool**, so prefix-cache hits are *derived from
routing + locality* instead of the static `--prefix-cache-hit-rate` above. Each
request draws one of `P` shared prefixes (a system-prompt / template of `L`
tokens); each instance keeps its own resident-prefix set; a request that lands
on an instance already holding its prefix is a hit (only the suffix is
prefilled).

```bash
infera-projection inference ... \
  --request-rate 40 --arrival-model poisson --des-num-requests 400 \
  --des-instances 8 --des-routing prefix_aware \
  --des-num-prefixes 32 --des-prefix-len 3072
```

Routing policies (`--des-routing`):

- `prefix_aware` — **KV-aware routing**: a prefix's requests always hit the same
  home instance, so it is warmed once and reused (misses ≈ `P`, independent of
  `N`).
- `round_robin` / `random` — scatter prefixes, so every instance re-warms them
  (misses ≈ `P · N`).

Example (gpt_oss_120B TP2, input=4096, 8 instances, 32 prefixes × 3072 tok,
Poisson @ 40 req/s, analytical):

| routing | prefix-cache hit rate | TTFT mean / p99 |
| --- | --- | --- |
| round_robin | 48% | 55.0 / 62.9 ms |
| prefix_aware | 92% | 49.0 / 58.3 ms |

The model also exposes the real **cache-locality vs. load-balance** trade-off:
KV-aware routing maximizes reuse but can imbalance load across instances, which
the pooled fleet metrics reflect. Set `N=1` with a prefix pool to model
single-engine temporal reuse (automatic prefix caching) alone. Knobs:
`--des-instances`, `--des-routing`, `--des-num-prefixes`, `--des-prefix-len`,
`--des-prefix-zipf` (popularity skew), `--des-cache-slots` (per-instance LRU
capacity).

## Confidence ladder

A 1-GPU anchor cannot observe cross-GPU communication (EP all-to-all, TP
all-reduce). The **confidence ladder** climbs the benchmark GPU count per recipe
until per-GPU decode is flat within ±5% across an adjacent GPU-count pair,
bounding the restore error (≤10% decode MAPE target) and preserving recipe rank
ordering. Pure-TP recipes self-anchor at the target TP.

## Tuning agent

`infera-tuning` runs a deterministic seed sweep then a DSPy planner/RLM loop
that proposes recipes and scores them through the projector — no GPU in the
default path. See `_vendor/primus/agents/tuning_agent/`.
