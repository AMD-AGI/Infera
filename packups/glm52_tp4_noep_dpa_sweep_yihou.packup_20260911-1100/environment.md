# Environment and provenance

Raw probe output: `env/host_crsuse2-m2m-217.txt` (captured 2026-09-11T11:26Z, after the runs).

## Hardware — node crsuse2-m2m-217 (single node, the only node used)

First-hand, probed on the node:

| Item | Value |
|---|---|
| Hostname | `crsuse2-m2m-217` |
| OS | Ubuntu 24.04.4 LTS |
| Kernel | `6.8.0-107-generic` (x86_64), uptime 61 days at probe time |
| CPU | 2× AMD EPYC 9575F 64-Core, 59 cores/socket, 2 threads/core, 236 logical CPUs |
| RAM | 2.7 TiB total (1.5 TiB free, 1.2 TiB page cache at probe time) |
| GPU driver | ROCm driver `6.14.14` |
| GPU arch | `gfx950` (read from the run's own `runtime.log`) = AMD Instinct MI355X |
| GPUs used | 4 of 8, `HIP_VISIBLE_DEVICES=0,1,2,3` |
| Allocation | spur/Slurm job `133750`, `crsuse2-m2m-217`, 2026-09-11T07:54:28Z → 15:54:28Z |

**RDMA fabric — captured but irrelevant to this result.** 8× `ionic` rails (400 Gb/s, Ethernet,
`enP2p0s9..12` / `enP3p0s9..12`) plus `mlx5_0` (200 Gb/s, Ethernet, `ens3`), all ACTIVE / LINK_UP.
This experiment is **single-node**; no cross-node traffic occurred, so no network claim is made and
the fabric cannot explain any number here. Data-plane IPs were not recorded (not needed).

**Gap — GPU detail not re-probed.** `rocm-smi` inside a `spur exec` shell reports 0 GPUs because the
job cgroup presents an empty `/dev/dri`; real devices are only visible inside the container started
with `--device=/dev/kfd --device=/dev/dri`. By the time this packup was assembled (11:26Z) the
**Docker daemon on 217 had entered `failed` state**, so per-GPU VRAM/util and the container's package
inventory could not be re-probed. All measurements had completed by 10:56Z, so no data was lost — but
the container-side figures below are carried over from the packup of the identical pinned image
rather than freshly measured today, and are labelled as such.

## Software

First-hand, from this sweep's own artifacts (`evidence/points/*/config_yihou.json`,
`result_yihou.json`, `rank_*_yihou.json`):

| Item | Value | Source |
|---|---|---|
| Container image | `sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d` | pinned **by digest** in `create_container_yihou.sh` and re-asserted by `run_decode.sh` before every point |
| Image archive | `/shared_nfs/yihou/playground/glm52_mix_repro_20260908/image/rocm-llm-bench.tar.zst` | 23,912,216,852 B, SHA256 `a6d047de9e3791b3c819f9035b0c3a4f632663b5cf8f85c3147b4f5b8204f9a2` |
| SGLang commit | `402df1e1e453e1e85ec0f5ac4052d36598cc691a` | `expected_sglang_commit` in config **and** `pinned_sglang` in every result |
| SGLang source in image | `/sglang/python/sglang/...` | `module_source_paths` in every rank report |
| Model | `/shared_nfs/models/GLM-5.2-MXFP4` (read-only NFS mount) | `config_yihou.json` |
| Wrapper repo | branch `dev.yihou.sglang.bench.fast.script`, HEAD `4b2b3bc39cc019a9c651c1bfdfd35eb6e63dcbae` | `evidence/code_snapshot/git_head.txt` |
| Bench code hashes | `evidence/code_snapshot/code_hashes.sha256` | identical across all 14 points (verified) |

The `rocm-llm-bench:latest` **tag** is mutable and was *not* the selection criterion — the digest was.

### Carried over from the identical-image packup (not re-probed today)
Recorded in `packups/glm52_tp4_ep4_dpa_sweep_yihou.packup_20260910-062400/environment.md`, verified
there against the **same image digest**:
- Base tag `lmsysorg/sglang:v0.5.18-rocm720-mi35x`, base digest
  `sha256:6d68cd19206716cb3f1e31e2ad89cd0852d7ae614a792773c30a4277f8955c72`.
- AITER `xiaobochen-amd/aiter` commit `2c71811b32c8ce2e1266aedaec199df7d90f597d`.
- Torch `2.9.1+rocm7.2.0.git7e1940d4`, Python 3.10, Triton `3.7.0+amd.rocm7.2.0.git89002410`,
  FlyDSL `0.3.2`, transformers `5.12.1`.

These are second-hand **for this sweep** (same image, different day and node). Since the image is
pinned by digest and the runs re-assert that digest before executing, the stack is the same by
construction — but this packup does not claim a fresh probe of it.

## Runtime environment variables that affect the result
Set by `scripts/create_container_yihou.sh`:
`HIP_VISIBLE_DEVICES=0,1,2,3`, `AITER_JIT_DIR=/tmp/yihou-aiter-b9a83742f631`,
`AITER_USE_FLYDSL_MOE_SORTING=1`, `SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1`,
`SGLANG_OPT_USE_TOPK_V2=false`, `SGLANG_TIMEOUT_KEEP_ALIVE=900`, `HF_HUB_OFFLINE=1`,
`HF_HOME=/tmp/yihou-hf-cache`, `PYTHONNOUSERSITE=1`, `PYTHONUNBUFFERED=1`.
Container also gets `--network host --ipc=host --shm-size 32g --ulimit memlock=-1:-1` and
`--security-opt seccomp=unconfined`.

The `AITER_JIT_DIR` is keyed to the image digest and **persists between points**. This is why the
first point of a cold node costs ~824 s in `load_pool_capture` and later points ~62–74 s. It affects
startup only — `decode` timing is unaffected.

## Resolved benchmark configuration (identical for every point except `--batch-size` / `--ep-size`)
From `config_yihou.json` (`server_cli`), verbatim:
`--tp-size 4 --ep-size <1|4> --dp-size 4 --moe-a2a-backend none --enable-aiter-allreduce-fusion
--enable-fused-qk-norm-rope --mem-fraction-static 0.85 --enable-dp-attention
--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-num-draft-tokens 6
--speculative-eagle-topk 1 --kv-cache-dtype fp8_e4m3 --dsa-decode-backend flydsl
--dsa-prefill-backend flydsl --dsa-topk-backend aiter --cuda-graph-bs-decode <C/4>
--cuda-graph-max-bs-decode <C/4> --random-seed 1234 --disable-radix-cache --skip-tokenizer-init
--disable-overlap-schedule --trust-remote-code`

Benchmark-side: `input_len 70000`, `output_len 10000`, `accept_length 3.61`
(`accept_method match-expected`, `accept_token_mode real-draft-token`), `warmup_steps 10`,
`seed 1234`, `disable_cuda_graph false`, `max_steps 0`.

Memory, per rank, constant across concurrency: KV pool `#tokens: 2436864`,
`target_initialized_bytes` 108.26 GB, `draft_initialized_bytes` 1.61 GB, `page_size 64`,
`reserved_tokens_per_request 80064`, `max_memory_allocated_bytes` 237.17 GiB.

## Required access (names and sources only — no values anywhere in this packup)
- spur/Slurm account with an existing allocation and `spur exec` rights to your own job.
- Docker daemon access on the compute node.
- Read access to `/shared_nfs/models/GLM-5.2-MXFP4` and to the image archive path.
- **No** registry login, HF token, API key or SSH key is used by any script here
  (`HF_HUB_OFFLINE=1`; the image is loaded from a local archive, never pulled).
