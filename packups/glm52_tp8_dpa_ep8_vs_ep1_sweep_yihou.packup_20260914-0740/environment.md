# Environment and provenance

Raw probe output: `env/host_smci355-ccs-aus-n06-25.txt` (captured 2026-09-14T08:23Z, after the runs)
and `env/container_yihou-glm52-tp8ep8-0914.txt` (captured 2026-09-14T08:24Z, from the **same
container instance that executed all 20 points** — it was still running).

## Hardware — `smci355-ccs-aus-n06-25` (single node, the only node used)

First-hand, probed on the host:

| Item | Value |
|---|---|
| Hostname | `smci355-ccs-aus-n06-25.prov.aus.ccs.cpe.ice.amd.com` |
| OS | Ubuntu 22.04.5 LTS |
| Kernel | `6.8.0-107-generic` (x86_64), uptime 4 d 23 h at probe time |
| CPU | 2× AMD EPYC 9575F 64-Core, 64 cores/socket, 2 threads/core, **256 logical CPUs** |
| RAM | 3.0 TiB total (1.8 TiB free, 1.0 TiB page cache at probe time) |
| GPU | **8× AMD Instinct MI355X**, `gfx950`, 288 GB each |
| GPU driver | ROCm driver **`6.16.13`** |
| GPUs used | **all 8**, `HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` |
| Allocation | **none** — bare metal, no Slurm/spur. The host was verified idle (`docker ps` empty, `rocm-smi` 0 % on all 8) before the first run |
| Local disk | `/dev/nvme2n1p2`, 14 TB, 2 % used (Docker image store) |

**RDMA fabric — captured but irrelevant to this result.** 8× `ionic` rails, 400 Gb/s, Ethernet
link layer, all `Active` / `LinkUp` (`ionic_0` … `ionic_7`; node GUIDs in the raw probe). Every run
here is **single-node**; no cross-node traffic occurred, so no network claim is made and the fabric
cannot explain any number in this packup. Data-plane IPs were not recorded (not needed).

## Software

First-hand, from this sweep's own artifacts (`evidence/points/*/config_yihou.json`,
`result_yihou.json`, `rank_*_yihou.json`) and from the live container probe:

| Item | Value | Source |
|---|---|---|
| Container image | `sha256:b5aa5bd3d828285bff8dd01bb92e6106f569f5b1eab5de24558790b1b10a5dc0` | pinned **by digest** in `create_container_yihou.sh` and **re-asserted by `run_decode.sh` before every point** |
| Image local tag | `rocm-llm-bench:latest` (mutable tag; the **digest** was the selection criterion) | `docker images` |
| Image built / size | 2026-09-08T11:51:53Z, 66,304,135,225 B | `docker image inspect` |
| SGLang commit | `402df1e1e453e1e85ec0f5ac4052d36598cc691a` | `git -C /sglang rev-parse HEAD` **and** `pinned_sglang` in every result |
| SGLang version string | `0.0.0.dev16896+g402df1e1e.d20260908` | `pip list` in the container |
| SGLang source in image | `/sglang/python/sglang/...` | `module_source_paths` in every rank report |
| AITER commit | `2c71811b32c8ce2e1266aedaec199df7d90f597d` | `git -C /aiter rev-parse HEAD` |
| torch | `2.9.1+rocm7.2.0.lw.git7e1940d4` | `pip list` |
| triton | `3.7.0+amd.rocm7.2.0.git89002410` | `pip list` |
| transformers | `5.12.1` | `pip list` |
| FlyDSL | `0.3.2` | `pip list` |
| Python | 3.10.12 (venv at `/opt/venv`) | container probe |
| Model | `/perf_apps/data/models/GLM-5.2-MXFP4` (read-only NFS mount, 408 GB) | `config_yihou.json` |
| Wrapper repo | branch `dev.yihou.sglang.bench.fast.script`, HEAD `e38dadae464e558795e762a96033595e56a3e51a` | `evidence/code_snapshot/git_head.txt` |
| Bench code | unmodified; `code.diff` is **0 bytes** | `evidence/code_snapshot/` |
| Bench code hashes | `evidence/code_snapshot/code_hashes.sha256` | identical across all 20 points (verified) |

**Discrepancy recorded, not resolved:** the image carries a build-arg
`AITER_COMMIT=d9e5ef7ce08ee7045d583aed768cff41aa9210fe` in its environment, which does **not** match
the AITER tree actually checked out at `/aiter` (`2c71811b32c…`). The checked-out tree is what ran.
Why the build arg differs was not investigated.

## Relationship to the spur packups

The reference runs (`../glm52_tp4_noep_dpa_sweep_yihou.packup_20260911-1100/`) pin image digest
`sha256:b9a83742f631…`, whose archive does **not** exist on this host. The digest used here is
different, but the stack it carries was probed and is the same: SGLang `402df1e1e45…`, AITER
`2c71811b32c…`, torch `2.9.1+rocm7.2.0`. **A different build of the same sources — not the same
image.** Everything else that differs is listed in `notes.md` → "Deviations".

## Runtime environment variables that affect the result

Set by `scripts/create_container_yihou.sh`:
`HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, `AITER_JIT_DIR=/tmp/yihou-aiter-b5aa5bd3d828`,
`AITER_USE_FLYDSL_MOE_SORTING=1`, `SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1`,
`SGLANG_OPT_USE_TOPK_V2=false`, `SGLANG_TIMEOUT_KEEP_ALIVE=900`, `HF_HUB_OFFLINE=1`,
`HF_HOME=/tmp/yihou-hf-cache`, `PYTHONNOUSERSITE=1`, `PYTHONUNBUFFERED=1`, plus the host-specific
`HOME=/tmp/yihou-home`, `USER`/`LOGNAME`, and the explicit `PATH` (see `notes.md`).

Baked into the image and left untouched (they also affect the result — full list in
`env/container_*.txt`): `SGLANG_USE_AITER=1`, `SGLANG_MOE_PADDING=1`,
`SGLANG_ROCM_FUSED_DECODE_MLA=1`, `SGLANG_SET_CPU_AFFINITY=1`, `SGLANG_USE_ROCM700A=1`,
`HIP_FORCE_DEV_KERNARG=1`, `HSA_NO_SCRATCH_RECLAIM=1`, `NCCL_MIN_NCHANNELS=112`,
`ROCM_QUICK_REDUCE_QUANTIZATION=INT8`, `TORCHINDUCTOR_MAX_AUTOTUNE=1`,
`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`, `PYTORCH_ROCM_ARCH=gfx942;gfx950`.

Container also gets `--network host --ipc=host --shm-size 32g --ulimit memlock=-1:-1`,
`--security-opt seccomp=unconfined`, `--device=/dev/kfd --device=/dev/dri`, `--group-add video`,
`--group-add render`, and `--user <host uid>:<host gid>`.

The `AITER_JIT_DIR` is keyed to the image digest and **persists between points** for the life of the
container. This is why a cold first point is far slower than later ones. It affects startup only —
`decode` timing is unaffected.

## Resolved benchmark configuration

Identical for every point except `--batch-size` / `--max-running-requests` (= C) and `--ep-size`.
From `config_yihou.json` (`server_cli`), verbatim:

```
--model-path /perf_apps/data/models/GLM-5.2-MXFP4 --tp-size 8 --ep-size <8|1> --dp-size 8
--moe-a2a-backend none --max-running-requests <C> --enable-aiter-allreduce-fusion
--enable-fused-qk-norm-rope --mem-fraction-static 0.85 --enable-dp-attention
--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-num-draft-tokens 6
--speculative-eagle-topk 1 --kv-cache-dtype fp8_e4m3 --dsa-decode-backend flydsl
--dsa-prefill-backend flydsl --dsa-topk-backend aiter --cuda-graph-bs-decode <C/8>
--cuda-graph-max-bs-decode <C/8> --random-seed 1234 --disable-radix-cache --skip-tokenizer-init
--disable-overlap-schedule --trust-remote-code
```

Benchmark-side: `input_len 70000`, `output_len 10000`, `accept_length 3.61`
(`accept_method match-expected`, `accept_token_mode real-draft-token`), `warmup_steps 10`,
`seed 1234`, `disable_cuda_graph false`, `max_steps 0`.

Attention backend resolved to `DeepseekSparseAttnBackend` for both target and draft.
`scheduler_used = false`, `real_model_weights = true`, `real_moe_routing = true`,
`synthetic_prefix = true`, `simulated_acceptance = true`.

Memory, per rank: KV pool `#tokens 3,448,128` (EP8) / `3,471,936` (EP1), `page_size 64`,
`reserved_tokens_per_request 80,064`, `speculative_reserve 12`;
`target_initialized_bytes` 164.48 GB (EP8) / 165.61 GB (EP1);
`max_memory_allocated_bytes` 253.45 → 256.36 GB across C=48 → 320 (measured on the EP8 sweep).

## Required access (names and sources only — no values anywhere in this packup)
- Docker daemon access on the host (group membership).
- Read access to `/perf_apps/data/models/GLM-5.2-MXFP4` (world-readable NFS mount) and to the repo.
- **No** registry login, HF token, API key, SSH key or cluster account is used by any script here
  (`HF_HUB_OFFLINE=1`; the image is already local and is never pulled).

## Gaps
- **Per-GPU VRAM / utilisation during the runs was not sampled** into a time series; only the
  harness's own `max_memory_allocated_bytes` per point is recorded. A concurrent `rocm-smi` trace
  would be needed to say anything about transient memory behaviour.
- **No repeats.** Every one of the 20 points is a single run, so run-to-run variance is unmeasured.
- The `+2.9 GB` device-memory growth across concurrency was measured on the **EP8** sweep only; the
  same span was not checked on EP1.
