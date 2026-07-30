# Environment

## When
Debug + fix + verification: **2026-07-22** (root cause + DSv4 fix) and
**2026-07-23** (final GLM verification / pack-up). Single continuous session.

## Hardware / nodes
| role | node | data-plane IP | GPUs used |
|------|------|---------------|-----------|
| prefill (P) / etcd / router | **chi2879** | 10.2.122.10 | card0-3 |
| decode (D) | **chi2866** | 10.2.122.47 | card4-7 |

- **GPU:** AMD Instinct **MI355X** (gfx950), 8 per node, ~298 MB/card idle baseline.
- chi2866 is ALSO the slurm jump host; card0-3 held foreign `titan` training +
  a stale GLM Mooncake decode throughout — untouched. That is why D ran on card4-7
  and DSv4 decode used port 30012 (30002 was taken).
- **RDMA fabric:** 8× **ionic** rails per node (IB, not TCP), `MC_GID_INDEX=1`
  `MORI_IB_GID_INDEX=1`. Measured chi2879↔chi2866 ≈ 354 Gb/s (healthy same rail).

## Software
- **Docker image:** `inferaimage/infera:vllm-v0.25.1-20260721`
  - image id `sha256:368cadb4d9838e548c3024bd65d7f30973615238612ece1bcc3e73b1f779a71d`
  - also tagged locally `infera/engine-vllm:test-local`.
  - **base:** `vllm/vllm-openai-rocm:v0.25.1@sha256:84459732ca98b40fe2f5338a3f050be6d522504e47a484a5180d58fb75956f86`
    (vLLM 0.25.1, torch 2.11.0, ROCm 7.2.3).
- **Repo:** branch `yihou.dev.vllm.image.update` @ `29021d303c1a6dc7b26ab24ba168fc855c9a40d5`
  ("Upgrade vLLM base image to stable v0.25.1 (digest-pinned)"). Note: the image was
  built BEFORE `patch_moriio_pagelen.py` existed, so the live runs applied the fix as a
  runtime hot-patch (see REPRODUCE §2b). A fresh image build from this repo state bakes
  the fix in via the Dockerfile patch loop — no hot-patch needed.
- **Connector:** `MoRIIOConnector`, **WRITE mode** (default; `read_mode` extra_config
  flips to READ). V1-only.

## Per-node env vars (in the launch scripts)
```
VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ENGINE_READY_TIMEOUT_S=3600
AITER_BF16_FP8_MOE_BOUND=0 PYTHONHASHSEED=0 VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
VLLM_HOST_IP=<dataplane-ip> MC_GID_INDEX=1 MORI_IB_GID_INDEX=1
HIP_VISIBLE_DEVICES=0,1,2,3 (P) / 4,5,6,7 (D)
```

## External dependencies (absolute paths, not in repo)
- **DSv4 weights:** `/mnt/vast/d_huggingface/models/DeepSeek-V4-Pro-fixed`
  (must be the `-fixed` copy; `model_type=deepseek_v4`, fp8 KV mandatory —
  `fp8_ds_mla` layout). VAST shared mount, same path in-container.
- **GLM weights:** `/mnt/vast/xiaobo/models/GLM-5.1-FP8`
  (`GlmMoeDsaForCausalLM`, 78 layers, fp8 dynamic, DSA indexer index_head_dim=128).
- **Shared work dir:** `/mnt/vast/c_huggingface/vllm_patch_verify/` (scripts, logs,
  the live `moriio_fix_loop/working_process.md`). `/mnt/vast` is the shared VAST
  filesystem (compute nodes see it; `/tmp` is NOT shared).
- **libionic:** host-ABI-specific, injected at container start (infera convention),
  never baked. Privileged `--device=/dev/infiniband` run gives rail access.

## Required secrets (names + source only — no values here)
- **Cluster SSH:** ProxyJump preconfigured in `~/.ssh/config` (direct `ssh chiXXXX`).
- **Docker image:** locally present on both nodes; if re-pulling, `inferaimage`
  creds from team vault.
- **etcd:** unauthenticated on the data plane (no creds needed).

## Not captured (honest gaps)
- Exact host kernel / ROCm driver point-versions per node were not snapshotted this
  session. If reproducing on different nodes, run `collect_env.sh` (from the
  experiment-result-packup skill) on each and compare; the fix is host-agnostic
  (it's a byte-layout correction) so kernel drift should not matter.
