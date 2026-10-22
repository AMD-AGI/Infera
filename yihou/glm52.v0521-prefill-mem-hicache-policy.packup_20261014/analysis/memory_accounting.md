# Prefill GPU memory accounting (answer to "这些空间确实被挤压的一滴不剩了么？")

All numbers first-hand unless marked [inferred]. Units: GiB (sglang prints "GB" but divides by 2^30; rocm-smi bytes / 2^30).

## 1. Startup accounting, prefill DP0 TP0 (server logs, `logs/server/<round>/prefill-0.log.gz`)

| round | mem_fraction_static | avail before weights | weights | avail after weights | KV pool (fp8_e4m3) | tokens/rank | "Memory pool end" avail | `available_gpu_mem` at max_total_num_tokens line |
|---|---|---|---|---|---|---|---|---|
| pf88 (274) / wts48 (136) / pc64 (089) | 0.88 | 278.42 | 119.84 | 158.58 | 125.14 | 2,433,088 | 33.18 | 32.96 |
| pf90 (089) | 0.90 | 278.42 | 119.84 | 158.58 | 130.71 | 2,541,376 | 27.70 | 27.47 |
| pf92 (085) | 0.92 | 278.42 | 119.84 | 158.58 | 136.28 | 2,649,600 | 22.21 | 21.99 |

- Post-pool free ≈ 278.42 × (1 − frac): 33.41 / 27.84 / 22.27 predicted vs 33.18 / 27.70 / 22.21 logged (within 0.25 GiB).
- Device total per GPU 287.98 GiB (rocm-smi); 287.98 − 278.42 = 9.56 GiB was already in use before weight load.
- KV per token: 125.14 GiB / 2,433,088 = 55,226 B (≈ 53.9 KiB)/token/rank (same ratio at 0.90/0.92).
- Code: the headroom term `pre_model_load_memory * (1 - mem_fraction_static)` appears in
  `sglang/srt/model_executor/model_runner_components/kv_pool_runtime.py:58-130` (`compute_post_capture_kv_resize`; source copy
  `/shared_nfs/yihou_glm52/v0521cap_20261012/src/image-9d755dd6/sglang/...`). Whether this post-capture resize path changed the
  pool in these runs is **not established**: the logged `max_total_num_tokens` equals the startup "KV Cache is allocated" count.

## 2. Runtime VRAM (results/vram/vram.089.tsv; `scripts/session/vram_peak.yihou.sh`, rocm-smi every 5 s)

Node 089, round pc64 (prefill mem 0.88, prefill_complete, HiCache 370 write_back, C64), 07:29:26-07:50:55 UTC, 255 samples
(end of A1 + start of the aborted A2):

| GPU | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| role | prefill | prefill | prefill | prefill | decode | decode | decode | decode |
| peak used GiB | 277.44 | 276.98 | 274.98 | 276.30 | 285.01 | 285.00 | 285.01 | 284.91 |
| never used at peak (287.98 − peak) | 10.54 | 11.00 | 13.00 | 11.68 | 2.97 | 2.98 | 2.97 | 3.07 |

- Prefill "Memory pool end" used = 287.98 − 33.18 = 254.80 GiB → runtime use above the pool at peak = 20.2-22.6 GiB at 0.88.
- Samples after 07:51 belong to the pc64 stop / cx90 bring-up on the same node and are not part of the table.
- The sampler covers only pc64 (a different config from pf90/pf92: prefill_complete + write_back + C64). **No VRAM trace exists
  for pf90 or pf92.**

## 3. What this does and does not say

- At 0.88 on 089/pc64, 10.5-13.0 GiB per prefill GPU was never touched at peak → the pool was **not** squeezed "to the last drop" at 0.88.
- At 0.92 the post-pool free is 22.21 GiB; if the runtime overhead were the same 20.2-22.6 GiB as in pc64, the margin would be
  −0.4 to +2.0 GiB [inferred; overhead at 0.92 / C48 was not measured]. pf92 A1 (C48, full 3600 s) completed without an OOM line in its logs;
  pf92 A2 died at 05:07 from the job time limit, not from OOM (NODE_FAIL at 1-00:00:24 RunTime per scontrol in poll.md; time limit as the cause is [inferred]; sacct (read 10-14 ~08:10) shows 211169 `COMPLETE -1:0`, Elapsed 1-00:00:23, TimeLimit UNLIMITED).
- Untested: whether the runtime overhead depends on concurrency / chunked-prefill size / HiCache policy; how far above 0.92 the
  pool can go. Probe prepared (env.pf93-136 / env.pf94-136: 0.93 / 0.94; expected 0.94 pool ≈ 2.758 M tok and ≈ 16.7 GiB
  post-pool free [inferred from the linear fit]) — in flight, no result.
- Not done: the "增加 kv_cache_token, 调整 block 对齐" route via `--max-total-tokens` at 0.92 (no round was started for it).
