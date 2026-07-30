# MoE-experts op optimize loop (issue #40)

The closed loop for iterating custom kernels behind the [Infera vLLM op-injection
plugin](../../infera/engine/vllm/ops/): **profile → tune → inject → re-profile**,
on a real model's op dimensions (Kimi-2.6 by default). A new kernel is a
`@register_experts_variant` in `infera/engine/vllm/ops/moe.py`; these scripts
measure, diagnose, and tune it.

Run inside the vLLM ROCm image with the repo mounted and `PYTHONPATH` set, e.g.
`docker run --device=/dev/kfd --device=/dev/dri -v <repo>:/work -w /work
-e PYTHONPATH=/work <vllm-rocm-image> bash -lc "<cmd>"`.

## The loop

```
                ┌────────────────────────────────────────────────┐
                ▼                                                 │
   profile_op.py  ──►  tune_op.py --inject  ──►  (plugin updated) ┘
   where's the time?   search configs,          _TUNE_DEFAULTS
   BW / roofline /     keep correct ones,        rewritten →
   which kernel        bake the winner in        re-profile confirms
```

| Ring | Script | What it does |
| --- | --- | --- |
| **measure** | `moe_experts_loop.py` | built-in vs plugin-op vs torch-reference: latency + max abs/rel error. The A/B + correctness gate. |
| **profile** | `profile_op.py` | roofline: achieved HBM BW vs peak → *bandwidth-bound* (near optimal) vs *launch/occupancy-bound* (headroom); `--kernels` adds the per-kernel device-time split. |
| **tune** | `tune_op.py` | coordinate-descent over block/warp configs, keeping only correct ones; `--inject` rewrites `_TUNE_DEFAULTS` in the plugin (the 植入 step). |

## Example cycle

```bash
# 1. measure the current op vs the built-in (baseline)
INFERA_MOE_EXPERTS=infera_decode python moe_experts_loop.py --tokens 1

# 2. profile it — is there headroom, and which kernel to attack?
python profile_op.py --impl infera_decode --tokens 1 --kernels

# 3. tune and inject the winner into the plugin
python tune_op.py --tokens 1 --inject

# 4. re-profile to confirm (new default is now baked in)
python profile_op.py --impl infera_decode --tokens 1
```

Measured on MI355X (vLLM 0.23 ROCm), Kimi-2.6 dims, decode `T=1`: the built-in
experts kernel = 0.171 ms; `infera_decode` after this loop = **0.135 ms
(1.31× )** at **~5.2 TB/s (65% of HBM peak)** — bandwidth-bound, so further wins
need less traffic (dtype / dedup), not more tuning.

## End-to-end (does the op win show up in a serve?)

`../_moe_decode_e2e_serve.sh` + `../_moe_decode_e2e_client.py` measure single-stream
(batch-1) decode ITL with the kernel wired into the serving path (the plugin's MoE
seam runs `infera_decode` on genuine small-M bf16 decode steps, verified via a
fire counter). Measured on **Qwen3.5-35B-A3B (bf16 MoE)**, MI355X, TP=1:

| config | ITL (ms/tok) | decode tok/s |
| --- | --- | --- |
| aiter **on** — production default | 5.754 | 173.3 |
| aiter off — builtin (triton) | 5.650 | 176.7 |
| **aiter off — `infera_decode`** | **5.264** | **188.8** |

So `infera_decode` is the **fastest config, ~8.5% better ITL than the aiter-on
default** — because at batch-1 aiter gives no decode benefit here (slightly
negative), so turning it off is free and the MoE kernel then wins. The op-level
~1.3× dilutes to single-digit e2e because routed experts are ~⅓ of the
bandwidth-bound batch-1 decode step.

**Limitations (by design, all self-delegating — never a regression):**
- **bf16/fp16 only** — MXFP4/quantized weights delegate (so Kimi-2.6 MXFP4 sees no effect).
- **batch ≤ 16** — larger batches delegate and aiter's large-M wins take over.
- **plain weight layout** — aiter pre-shuffles MoE weights, so the seam delegates when
  `VLLM_ROCM_USE_AITER` is set; the kernel runs with aiter's MoE path off (or
  `INFERA_MOE_ASSUME_PLAIN=1`). Mixing aiter-prefill with this decode kernel needs
  weight un-shuffling — future work.

## Config precedence

`_decode_tunables()` reads, in order: per-key env (`INFERA_MOE_GU_BLOCK_I`, …) →
`INFERA_MOE_TUNE_FILE` (JSON) → the baked `_TUNE_DEFAULTS`. So `--inject`
(rewrites `_TUNE_DEFAULTS`) makes a tuned config permanent, while env / tune-file
let you A/B without editing source.
