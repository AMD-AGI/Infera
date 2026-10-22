# Environment and provenance

Raw probes: `env/node254_yihou.txt` and `env/node267_yihou.txt`, captured 2026-09-14T06:21Z
**inside the running containers on both nodes** — first-hand, no carry-over.

## Hardware

| Item | node `crsuse2-m2m-254` | node `crsuse2-m2m-267` |
|---|---|---|
| OS | Ubuntu 24.04.4 LTS | Ubuntu 24.04.4 LTS |
| Kernel | `6.8.0-107-generic` | `6.8.0-107-generic` |
| RAM | 2751 GiB total | 2751 GiB total |
| GPUs | 8x, `309220868096` B each (**287.98 GiB**) | 8x, same |
| GPU idle usage at probe | ~298 MB/card | ~298 MB/card |
| GPUs used | 4 (`HIP_VISIBLE_DEVICES=0,1,2,3`) | 4 (same) |
| GPU arch | `gfx950` = AMD Instinct MI355X (from the runs' own `runtime.log`) | same |
| Allocation | spur/Slurm job `136670` | spur/Slurm job `136669` |

CPU is 2x AMD EPYC 9575F (236 logical CPUs) — taken from the prior packup for the same node class.
**Caveat, first-hand:** `nproc` inside a `spur exec` shell reports **1**, because that shell runs in
a CPU-restricted job cgroup. Do not record that number as the machine's core count.

Excluding transient values (free RAM, idle VRAM bytes) and environment-variable ordering, the two
nodes' probes are identical.

**RDMA fabric — captured, and irrelevant to this result.** Both nodes expose
`ionic_0 .. ionic_7` plus `mlx5_0`; every probed port is `PORT_ACTIVE (4)`, `link_layer: Ethernet`.
This experiment is **single-node per run** — no cross-node traffic occurred — so no network claim is
made and the fabric cannot explain any number here. Data-plane IPs were not recorded (not needed).

## Software — first-hand from inside the containers, both nodes identical

| Item | Value |
|---|---|
| Container image | `sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d` — pinned **by digest** in `create_container_yihou.sh` and re-asserted by `run_decode.sh` before every point |
| Image archive | `/shared_nfs/yihou/playground/glm52_mix_repro_20260908/image/rocm-llm-bench.tar.zst`, 23,912,216,852 B, SHA256 `a6d047de9e3791b3c819f9035b0c3a4f632663b5cf8f85c3147b4f5b8204f9a2` |
| SGLang | `0.0.0.dev16896+g402df1e1e` from `/sglang/python`; `git -C /sglang rev-parse HEAD` = **`402df1e1e453e1e85ec0f5ac4052d36598cc691a`**, matching `PINNED_SGLANG` in `profile_decode.py` |
| AITER | checkout **`2c71811b32c8ce2e1266aedaec199df7d90f597d`** (2026-09-04), from `git -C /aiter rev-parse HEAD` |
| torch | `2.9.1+rocm7.2.0.lw.git7e1940d4`, HIP `7.2.26015-fc0010cf6a` |
| triton | `3.7.0+amd.rocm7.2.0.git89002410` |
| transformers | `5.12.1` |
| flydsl | `0.3.2` |
| numpy | `2.2.6` |
| Model | `/shared_nfs/models/GLM-5.2-MXFP4` (read-only NFS mount) |
| Wrapper repo | branch `dev.yihou.sglang.bench.fast.script`, HEAD `e38dadae464e558795e762a96033595e56a3e51a` (identical across all 41 points) |
| Bench code hashes | `evidence/code_snapshot/code_hashes.sha256` — **one unique hash across all 41 points** (verified) |

The `rocm-llm-bench:latest` **tag** is mutable and was *not* the selection criterion — the digest was.

### Trap: `AITER_COMMIT` does not name the installed AITER
The image exports `AITER_COMMIT=d9e5ef7ce08ee7045d583aed768cff41aa9210fe` (and the same value as
`AITER_COMMIT_DEFAULT`). The **actual checkout** in `/aiter` is `2c71811b32c8ce2e1266aedaec199df7d90f597d`.
The env var is a build-time request, not the resulting state. Use `git -C /aiter rev-parse HEAD`.

This also **closes a gap in the published 2026-09-11 packup**, which could not probe the container
(that node's Docker daemon had failed) and carried the software stack over from an earlier packup,
explicitly labelled second-hand. The same image digest has now been probed directly; every carried
value is confirmed, including the AITER commit, so those figures can be upgraded to first-hand.

## Runtime environment variables that affect the result
Set by `scripts/create_container_yihou.sh`:
`HIP_VISIBLE_DEVICES=0,1,2,3`, `AITER_JIT_DIR=/tmp/yihou-aiter-b9a83742f631`,
`AITER_USE_FLYDSL_MOE_SORTING=1`, `SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1`,
`SGLANG_OPT_USE_TOPK_V2=false`, `SGLANG_TIMEOUT_KEEP_ALIVE=900`, `HF_HUB_OFFLINE=1`,
`HF_HOME=/tmp/yihou-hf-cache`, `PYTHONNOUSERSITE=1`, `PYTHONUNBUFFERED=1`.
Container also gets `--network host --ipc=host --shm-size 32g --ulimit memlock=-1:-1` and
`--security-opt seccomp=unconfined`.

Baked into the image (observed via `docker inspect`, both nodes): `BUILD_AITER_ALL=1`,
`AITER_USE_SYSTEM_TRITON=1`, `SGLANG_USE_AITER=1`, `SGLANG_USE_ROCM700A=1`,
`SGLANG_ROCM_FUSED_DECODE_MLA=1`, `SGLANG_MOE_PADDING=1`, `SGLANG_SET_CPU_AFFINITY=1`,
`SGLANG_DISABLE_CUDNN_CHECK=1`, `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`,
`SGLANG_INT4_WEIGHT=0`, `SGLANG_ROCM_DISABLE_LINEARQUANT=0`.

`AITER_JIT_DIR` is keyed to the image digest and **persists between points**, which is why a cold
node's first point costs ~14 min in `load_pool_capture` and later points ~1 min. Startup only —
`decode` timing is unaffected.

Two points (`*_mrr_xs_*`) additionally ran with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
injected per-run by `scripts/run_decode_env_yihou.sh`; each such point records it in
`docker_env_yihou.txt`.

## Resolved benchmark configuration
From `config_yihou.json` (`server_cli`), identical for every point except `--batch-size`,
`--ep-size`, `--mem-fraction-static`, `--max-running-requests` and the derived graph batch size:

`--tp-size 4 --ep-size <1|4> --dp-size 4 --moe-a2a-backend none --enable-aiter-allreduce-fusion
--enable-fused-qk-norm-rope --mem-fraction-static <MF> --max-running-requests <C>
--enable-dp-attention --speculative-algorithm EAGLE --speculative-num-steps 5
--speculative-num-draft-tokens 6 --speculative-eagle-topk 1 --kv-cache-dtype fp8_e4m3
--dsa-decode-backend flydsl --dsa-prefill-backend flydsl --dsa-topk-backend aiter
--cuda-graph-bs-decode <C/4> --cuda-graph-max-bs-decode <C/4> --random-seed 1234
--disable-radix-cache --skip-tokenizer-init --disable-overlap-schedule --trust-remote-code`

Benchmark-side: `input_len 70000`, `output_len 10000`, `accept_length 3.61`
(`accept_method match-expected`, `accept_token_mode real-draft-token`), `warmup_steps 10`,
`seed 1234`, `disable_cuda_graph false`, `max_steps 0`.

Per-rank memory, measured: KV costs **46.58 KiB/token**; each request reserves **80064 tokens
= 3.556 GiB**; target weights **119.84 GiB**, draft weights **6.92 GiB**, leaving **151.66 GiB** free.
Pool sizes actually allocated: 2,436,864 tokens at mf 0.85; 2,622,144 (ep1) / 2,602,432 (ep4) at
0.88; 3,202,688 at 0.974; 3,239,744 (ep1) / 3,220,096 (ep4) at 0.98.

## Required access (names and sources only — no values anywhere in this packup)
- spur/Slurm account with an existing allocation and `spur exec` rights to your own job.
- Docker daemon access on the compute nodes.
- Read access to `/shared_nfs/models/GLM-5.2-MXFP4` and to the image archive path.
- **No** registry login, HF token, API key or SSH key is used by any script here
  (`HF_HUB_OFFLINE=1`; the image is loaded from a local archive, never pulled).
