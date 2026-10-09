# Actual environment and external dependencies

## Historical measured hardware

- **Measured host:** `crsuse2-m2m-036`, owned replacement Spur job130891 (2026-09-10 05:01:03–11:01:03Z scheduled hold). Only GPUs0–3 were selected: four AMD Instinct MI355X, gfx950; host had eight GPUs. TP4 / EP4 / DP1, no DP attention.
- **Actual run times:** C16 launcher05:08:32Z, ready05:18:43Z; C32 launcher05:22:14Z, ready05:26:23Z. Final runtime capture05:29:01Z and cleanup05:30:59Z. Per-round timestamp, readiness and console/log records are preserved.
- **Kernel:** `Linux 6.8.0-107-generic #107-Ubuntu SMP PREEMPT_DYNAMIC Fri Mar 13 19:51:50 UTC 2026 x86_64`.
- **Driver reported by rocm-smi:**6.14.14. Torch/Triton builds below identify their ROCm7.2.0 build lineage; no separate complete host ROCm package inventory was captured.
- **Memory:** `free -b` total2,954,723,254,272bytes, swap0, captured on036. This is the observed `/proc` total, not a cgroup reservation.
- **CPU model:** not captured; unknown. No login-node CPU substituted.
- **RDMA:** fake handoff performs no real P→D transport. NIC/driver/GID/rail details were not captured and are not needed for this local fake measurement. This kit does not reproduce real RDMA PD.
- Host Docker context, host network/IPC,32GiB shared memory, `/dev/kfd` and `/dev/dri`, video/render groups, memlock unlimited, seccomp unconfined. Full historical container options are in `rounds/*/container-inspect.json`.

First-hand evidence: `logs/final-environment.txt.gz`, `rounds/*/container-inspect.json`, `rounds/*/server-info.json`, `rounds/*/server-full.log.gz`. Host249/job130740 was an unsuccessful resource attempt, **not** the measured host. Peer056/job130737 was never accessed.

## Exact image and software

| Item | Captured identity |
|---|---|
| Executed immutable Docker image ID |`sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d`|
| Historical image alias |`rocm-llm-bench:latest`|
| Dockerfile base tag |`lmsysorg/sglang:v0.5.18-rocm720-mi35x`|
| Base registry digest |**Not captured. The executed image ID is not a base-image registry digest.**|
| SGLang commit |`402df1e1e453e1e85ec0f5ac4052d36598cc691a`|
| SGLang package |`0.0.0.dev16896+g402df1e1e.d20260908`, editable `/sglang/python`|
| AITER commit |`2c71811b32c8ce2e1266aedaec199df7d90f597d`|
| AITER package |`0.1.1.dev2938+g2c71811b3`, editable `/aiter`|
| Torch |`2.9.1+rocm7.2.0.lw.git7e1940d4`|
| Triton |`3.7.0+amd.rocm7.2.0.git89002410`|
| Python |3.10 library paths observed; exact interpreter patch version not captured|

`reference/Dockerfile` is the exact earlier build recipe with the two source SHAs. It clones the `xiaobochen-amd/sglang` and `xiaobochen-amd/aiter` forks and checks out commits. Branch names were not independently captured for this short run; prior notes associate the SGLang commit with `dev_glm52_0907`, but the immutable commit/image is authoritative. No full dependency lock or exact FlyDSL package version was captured. A rebuild from the mutable base tag is not guaranteed binary-identical.

Three full files are overlaid read-only on SGLang; their byte identities and patch application are audited. No diagnostic bounds backend or serialization environment flags were mounted/enabled for these points. No Git repository reconstruction is required.

### Runtime environment, including inherited flags

Full environment arrays are preserved in each container inspect JSON. Besides launch script variables (`HIP_VISIBLE_DEVICES=0,1,2,3`, acceptance3.61/match-expected/real-draft-token, cache paths), inherited performance variables include `SGLANG_USE_AITER=1`, `HIP_FORCE_DEV_KERNARG=1`, `HSA_NO_SCRATCH_RECLAIM=1`, `NCCL_MIN_NCHANNELS=112`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT8`, `SGLANG_SET_CPU_AFFINITY=1`, `SGLANG_ROCM_FUSED_DECODE_MLA=1` and TorchInductor autotune settings. Do not infer effective configuration from launch arguments alone.

The inherited `AITER_COMMIT=d9e5ef7...` environment string describes the base image build setting, **not** the later installed editable AITER checkout. The package/version plus pinned Dockerfile commit identify the replacement AITER version.

## External inputs: intentionally not bundled

- **Model/tokenizer:** `/shared_nfs/models/GLM-5.2-MXFP4`, MXFP4 weights, FP8 e4m3 KV. No model shard checksum manifest was captured; path/model name is not a byte identity guarantee.
- **Preferred exact image archive:** `/shared_nfs/yihou/playground/glm52_mix_repro_20260908/image/rocm-llm-bench.tar.zst`. Prior package records23,912,216,852bytes and SHA256 `a6d047de9e3791b3c819f9035b0c3a4f632663b5cf8f85c3147b4f5b8204f9a2`. This archive hash is inherited provenance, not newly rehashed during this bounded packup. Verify before load; require the exact final image ID afterward. Archive and layers excluded.
- **Dataset:** local `reference/synthetic_prompts.json` is included verbatim. Client tokenizes/repeats/truncates to exactly10000 integer input IDs and requests500 output IDs; range ratio1. No ShareGPT download needed.
- **Caches:** original run copied exact-image-compatible cache products from the previous experiment before launch. All compiled/JIT/HF caches are excluded here. Cold startup may exceed30minutes and the default2400s readiness wait; budgets are not performance guarantees.
- No host-injected external library file is named by the measured launcher. No external prefill host/etcd service is required.

## Required access, no secret values

A currently authorized Spur allocation and host Docker access are required for GPU replay. The model and image archive require shared-filesystem access. No API/HF token is needed for the local model and synthetic dataset. Network rebuild may need GitHub and Docker Hub authentication per site policy; obtain credentials through your own configured account/registry login, never from this kit. No credential file or secret value is packaged. The offline audit uses standard-library Python3.10+, GNU patch and bash only; it does not contact the cluster or run a container.
