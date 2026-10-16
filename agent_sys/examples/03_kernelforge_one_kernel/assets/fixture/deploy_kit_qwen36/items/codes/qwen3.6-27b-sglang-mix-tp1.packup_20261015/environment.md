# Environment (rendering of ../environment.yaml)

This is a human-readable rendering of `items/codes/environment.yaml`. The yaml
is the record and this file is not. Values are copied verbatim.

## Fixed

| field | value |
|---|---|
| node | `smc300x-ccs-aus-a16-19` |
| node_ip | `127.0.0.1` |
| image | `infera:sglang-gfx942-glm52` |
| image_id | `sha256:c05d2bebe442e8e0ea7c0d25a446ef6984a73c2eac5c5f048b77c708fd080818` |
| model_name | `Qwen3.6-27B` |
| served_model_name | `Qwen3.6-27B` |
| model_path | `/data/yihou_hf_cache/models/Qwen3.6-27B` |
| deploy_mode | `mix` |
| tp_size | `1` |
| context_length | `32768` |
| gpu_arch | `gfx942` |
| gpu_count | `8` (cards present on the node, counted from `rocm-smi --json`) |
| gpu_devices | `[6]` (the card this deployment took, named by the operator) |
| dockerfile | not set. The image was already present and was not built |

## Runtime (the producing run, now torn down)

| field | value |
|---|---|
| transport | `local` (slurm_jobid `local`) |
| container | `yihou_l2a_r16aea3` (+ `yihou_l2a_r16aea3_etcd`) |
| endpoint | `http://127.0.0.1:18101` (infera router) |
| ports | router 18101, worker 18102, etcd 18103 (kit band also uses 18104 etcd peer, 18105 kv-events, 18106 kv-snapshot) |
| started_at | `2026-10-15T07:02:06Z` (when the record was rendered; bring-up began 06:52:53Z) |

## Software inside the image, read from the running system

- infera at `/opt/infera` (the image's installed package), sglang `0.5.16`, torch
  `2.9.1+rocm7.2.0`, ROCm 7.2.0 in the image. Host ROCm `6.3.2-66`, kernel
  `6.5.0-45-generic`, docker `29.1.3` with no authorization plugin.
- The image is also tagged `infera:sglang-gfx942-dsv4flash`. Despite `glm52` in
  the tag, its sglang serves `Qwen3_5ForConditionalGeneration`, so no build was
  needed.
- etcd image `quay.io/coreos/etcd:v3.5.14` at
  `sha256:42106a001d1be7fd22e48830071ceae0fe760ff2c5df8c76c0e3b586d8a1f0fc`.

## Engine settings as the engine reported them (`/get_server_info`)

| setting | requested | resolved |
|---|---|---|
| mem_fraction_static | 0.85 | 0.7225 (rescaled by sglang for the hybrid mamba model) |
| cuda_graph_backend_decode | full | full |
| cuda_graph_max_bs_decode | 32 | 32 (captured bs 1, 2, 4, 8, 12, 16, 24, 32) |
| context_length | 32768 | 32768 |
| reasoning_parser | qwen3 | qwen3 |
| max_running_requests | (engine default) | 57 (worker log) |
| KV pool | bf16 | 751,733 tokens, 22.94 GB each for K and V |
| mamba state pool | | 288 slots, ssm 40.64 GB, conv 0.79 GB |
