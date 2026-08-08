# Op optimize-loop scaffold (issue #40)

An **op-agnostic** `measure → profile → tune → inject` loop for iterating custom
kernels behind the [Infera vLLM op-injection plugin](../../infera/engine/vllm/ops/).
The scaffold is the deliverable; a kernel (e.g. `infera_decode`) is just a
candidate plugged in. **Adding an op is writing one `OpSpec` — no new script.**

```
framework.py   OpSpec + registry + generic measure/profile/tune
loop.py        one CLI over any registered op
ops/<name>.py  one OpSpec per op (self-registers). moe_experts.py = the template.
```

Run inside the vLLM ROCm image with the repo mounted and `PYTHONPATH` set (so the
edited plugin is used):
`docker run --device=/dev/kfd --device=/dev/dri -v <repo>:/work -w /work
-e PYTHONPATH=/work <vllm-rocm-image> bash -lc "cd bench/op_loop && <cmd>"`.

## The loop

```bash
python loop.py list                                   # registered ops
python loop.py measure --op moe_experts               # baseline vs candidate vs oracle
python loop.py profile --op moe_experts --kernels     # roofline + per-kernel split
python loop.py tune    --op moe_experts --inject      # autotune, bake winner into plugin
python loop.py measure --op moe_experts -d tokens=1 -d experts=64   # override dims
```

| Ring | What it does |
| --- | --- |
| **measure** | baseline (built-in) vs candidate (plugin op) vs reference (oracle): latency + rel error. The A/B + correctness gate. |
| **profile** | roofline from `traffic_bytes`: achieved HBM BW vs peak → bandwidth-bound (near optimal) vs launch/occupancy-bound (headroom); `--kernels` adds the per-kernel device-time split. |
| **tune** | sweep the op's `tune_env` configs, keep only correct ones; `--inject` calls the op's `inject` to bake the winner into the plugin. |

## Adding an op

Write `ops/<name>.py` with an `OpSpec` and `register_op` it — the CLI picks it up
by name. Provide what applies (the loop skips the rest):

| Field | For |
| --- | --- |
| `make_inputs(dims, dev)` | build the op's tensors at a model's dims |
| `baseline(*inputs)` | the engine's built-in op (A) |
| `candidate(*inputs)` | the plugin's op — the selected variant (B) |
| `reference(*inputs)` | correctness oracle (optional) |
| `traffic_bytes(dims)` | bytes moved, for the roofline (optional) |
| `tune_env` / `tune_grid` / `inject` | the tune ring (optional) |

`ops/moe_experts.py` is the reference implementation (Kimi-2.6 dims, the
`infera_fused_experts` candidate, a torch-SwiGLU oracle, block/warp tuning).

## Worked example: `moe_experts` / `infera_decode`

Measured on MI355X (vLLM 0.23 ROCm), Kimi-2.6 dims, decode `T=1`: built-in
experts = 0.171 ms; `infera_decode` after this loop = **0.135 ms (1.31× )** at
**~5.2 TB/s (65% of HBM peak)** — the profile verdict (bandwidth-bound, reads each
expert once = traffic floor) says it's near its roofline, so further wins need
*less traffic* (dtype), not more tuning.

### End-to-end (does the op win reach a serve?)

`../_moe_decode_e2e_serve.sh` + `../_moe_decode_e2e_client.py` measure batch-1
decode ITL with the kernel wired into the serving path (fire-counter verified).
**Qwen3.5-35B-A3B (bf16 MoE)**, MI355X, TP=1:

| config | ITL (ms/tok) | decode tok/s |
| --- | --- | --- |
| aiter **on** — production default | 5.754 | 173.3 |
| aiter off — builtin (triton) | 5.650 | 176.7 |
| **aiter off — `infera_decode`** | **5.264** | **188.8** |

Fastest config, **+8.5% ITL over the aiter-on default** (at batch-1 aiter gives no
decode benefit, so disabling it is free and the MoE kernel wins). Self-delegating
limitations: bf16 only, batch ≤ 16, plain (non-aiter-shuffled) weights — never a
regression.
