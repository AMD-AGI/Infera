# Baseline: the non-profiled C=256 EP8 point we are profiling

Source (first-hand, copied out of the published packup, not re-run):
`packups/glm52_tp8_dpa_ep8_vs_ep1_sweep_yihou.packup_20260914-0740/evidence/points/tp8_ep8_dpa_on_c256_yihou/result_yihou.json`

This is the **performance** number. Any profiled run is slower by construction and its TPOT must
never be quoted as a result.

| Field | Value |
|---|---|
| `complete` | true |
| global `batch_size` (C) | 256 |
| `local_batch_size` | 32 (= C / dp_size 8) |
| `input_len` / `output_len` | 70000 / 10000 |
| `verify_iterations` | **2768** |
| `useful_output_tokens` | 2 560 000 (= C × 10000, exact) |
| `realized_accept_length` | 3.6134393063583814 |
| `elapsed_seconds` (measured decode loop) | 262.51939422101714 |
| **`effective_token_latency_ms_per_user` (TPOT)** | **26.251939422101714 ms** |
| `output_tokens_per_second` | 9751.660472920015 |
| `output_tokens_per_second_per_gpu` | 1218.957559115002 |
| topology | tp=8, dp=8, ep=8, dpa=true, `moe_a2a_backend=none` |
| `target_graph_iterations` | 2768 — i.e. **every** measured iteration ran on the target CUDA graph |
| `max_memory_allocated_bytes` | 255 453 225 472 (237.9 GiB) |

Phase breakdown (max across ranks, seconds):

| phase | seconds |
|---|---|
| `load_pool_capture` | 54.957 |
| `physical_initialization` | 0.409 |
| `bootstrap` | 0.897 |
| `warmup_and_reset` | 9.337 |
| `decode` (the measured loop) | 262.519 |
| `rank_total_to_result` | 327.936 |

## Derived quantities that size the profiling window

- **One decode iteration ≈ 262.519 s / 2768 = 94.84 ms.** Each iteration emits ~3.61 accepted
  tokens per request, hence TPOT 94.84 / 3.6134 = 26.25 ms — consistent, so the iteration cost is
  the right unit to attribute.
- A 5-iteration profiling window is therefore ≈ **0.47 s** of GPU work. That is the right order for
  a trace: long enough to average over the EAGLE steps=5 structure, short enough not to produce an
  unmanageable file.
- `target_graph_iterations == verify_iterations` means the graph-ON path is the one that produced
  the published number. **Attribution for this point must come from the graph-ON profile.** The
  graph-OFF profile is a cross-check on kernel identity, not a substitute.
