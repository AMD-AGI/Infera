# Run DSv4-Pro with only the first N transformer layers

Truncate DeepSeek-V4-Pro to its first **N** layers (e.g. N=4) for fast
per-layer inspection of the forward path — **config-only, no SGLang source
change**. Verified 2026-07-27 on chi2879 (8×MI355X gfx950), image
`infera/engine-sglang:pd-mcgate` (sglang 0.5.15.post1).

## Why it works (no code change needed)

- The model builds exactly `num_hidden_layers` decoder layers via
  `make_layers(...)` (`deepseek_v4.py:1923`).
- The weight loader **auto-skips** any checkpoint tensor with
  `layer_id >= end_layer` (`deepseek_v4.py:2596`), and the MTP/nextn layer
  (id == `num_hidden_layers`, i.e. 61) is skipped too. So a full 61-layer
  checkpoint loads cleanly into an N-layer model.
- **Do NOT touch `compress_ratios`** (len = 62 = 61 layers + 1 MTP, last = 0).
  It is indexed by `layer_id` (`config.compress_ratios[layer_id]`), so leaving
  it at full length is correct: indices `0..N-1` are used, the rest ignored.
  No length assertion exists. `num_hash_layers` is also unused by the model.
  **Only `num_hidden_layers` matters.**

## How to run

On the node (chi2879), with GPUs free (`rocm-smi --showpids` → No KFD PIDs):

```bash
# 1. build the N-layer model dir (symlinks weights, rewrites one config.json)
bash scripts/make_first_n_config.sh 4

# 2. launch + probe (throwaway container, eager, dsv4 backend, R4 env)
bash scripts/run_first_n.sh 4 30000
```

`make_first_n_config.sh N` creates
`/mnt/vast/d_huggingface/models/DeepSeek-V4-Pro-{N}L`: symlinks every file from
the real model dir except `config.json`, which it rewrites with
`num_hidden_layers = N`. `run_first_n.sh N PORT` spins up a container, launches
the server, waits for ready, and fires 2 probe requests.

Archived config: [`../configs_4layer/config_4L.json`](../configs_4layer/config_4L.json).

## Verified evidence (N=4)

- Build: `[[DSV4-SUBSET]]`-free normal path; server ready, **0 errors**.
- Weight load: **`mem usage = 11.45 GB` / GPU** (vs ~105 GB for full 61 layers,
  ≈ 1/9 — embedding + lm_head dominate the 4-layer footprint). Proves only 4
  layers' weights were materialized.
- Forward: prefill + decode both fire end-to-end; 2 requests return 16/12
  tokens.
- **Output is gibberish by design** — 4 layers cannot language-model, but the
  first-N-layer forward path runs cleanly. Exercising that path (not producing
  coherent text) is the point.

## Note on non-contiguous subsets (first-4 + last-2) — NOT shipped

Loading a *non-contiguous* subset (orig layers `[0,1,2,3,59,60]`) was
prototyped via a `DSV4_LAYER_SUBSET` env + `deepseek_v4.py` patch (remap orig
layer id → contiguous local id for KV-pool indexing, keep orig id for
compress_ratio + weight-name remap). Build and weight-load **succeeded**
(`built 6 layers from orig ids [0,1,2,3,59,60]`, compress_ratios
`[128,128,4,128,128,4]`), but the first forward hit
`AssertionError: kv_score_buffer.shape[-1] == last_dim` in
`compressor_v2.py:447`. Root cause: the DSv4 KV pool's `layer_mapping` is built
from `config.compress_ratios[0:N]` by **local** id, so local layer 4 gets the
pool buffer sized for `compress_ratios[4]=4` while the actual layer (orig 59)
needs ratio 128 — a pool/layer compress-ratio mismatch. Making it work would
require threading the subset's orig ids into the KV-pool `layer_mapping`
construction too. Deferred — the contiguous first-N path above is the shipped,
verified method.
