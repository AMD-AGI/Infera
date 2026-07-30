# Environment

## When
Bring-up + verification: **2026-07-23**. Single session, ~6-7 min bring-up.

## Hardware / node
| role | node | GPUs used |
|------|------|-----------|
| single-node mix (prefill+decode) | **chi2879** | card0-3 |

- **GPU:** AMD Instinct **MI355X** (gfx950), 8 per node, ~283 MB/card idle baseline.
  chi2879 was fully free (all 8 cards) for this run.
- **No RDMA fabric used.** Single-node PD-mix is one server on one node; no ionic /
  MoRIIO / Mooncake / cross-node KV transfer. No `--kv-transfer-config` at all.

## Software
- **Docker image:** `infera/engine-vllm:test-local` (id `e91a6d7d3a91`, 36.3GB),
  built 2026-07-23 from repo branch `yihou.dev.vllm.glm` (vLLM **v0.25.1** base,
  digest-pinned `vllm/vllm-openai-rocm:v0.25.1`, torch 2.11.0, ROCm 7.2.3) + the
  Dockerfile patch loop (includes `patch_moriio_pagelen.py`). The pagelen fix is a
  **no-op for single-node mix** (no KV-transfer path); this run just confirms the
  image serves GLM correctly.
- **Run as:** privileged container, `--ipc host --shm-size 32g`,
  `-v /mnt/vast:/mnt/vast`. Server launched via plain `vllm serve` (OpenAI server).
- For the infera-native launch (`python -m infera.engine.vllm`) see the e2e case
  added alongside this kit (`tests/e2e/pd_mixed/vllm/matrix.py`).

## Per-run env vars
```
HIP_VISIBLE_DEVICES=0,1,2,3
VLLM_USE_V1=1
VLLM_ROCM_USE_AITER=1
AITER_BF16_FP8_MOE_BOUND=0
VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
PYTHONHASHSEED=0
```

## Server config
```
vllm serve --tensor-parallel-size 4 --trust-remote-code --kv-cache-dtype fp8
  --reasoning-parser glm45 --no-enable-prefix-caching --gpu-memory-utilization 0.85
  --max-model-len 9472 --max-num-batched-tokens 8192 --distributed-executor-backend mp
CUDA graphs captured in 181s (8.29 GiB); ready ~6-7 min; no OOM at 0.85.
aiter active via env (NOT --moe-backend). NO kv-transfer connector.
```

## External dependencies (absolute paths, not in repo)
- **GLM weights:** `/mnt/vast/xiaobo/models/GLM-5.1-FP8` (VAST shared mount, same
  path in-container).
- **Shared work dir:** `/mnt/vast/c_huggingface/` (scripts, `glm_vllm_mix.log`).
  `/mnt/vast` is shared VAST; `/tmp` is NOT shared.

## Required secrets (names only — no values)
- **Cluster SSH:** ProxyJump preconfigured in `~/.ssh/config` (`ssh chi2879`).
- **Docker image:** present on the node (built from the repo).

## Not captured (honest gaps)
- Exact host kernel / ROCm driver point-versions not snapshotted. Correctness
  bring-up (no perf number), so kernel drift should not change the verdict.
- No throughput/latency numbers — correctness bring-up only.
- Host port 8000 collided with a pre-existing `glm_pd` container on the node, so the
  probe went via `docker exec` localhost rather than a published port (cosmetic).
