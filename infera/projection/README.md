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
