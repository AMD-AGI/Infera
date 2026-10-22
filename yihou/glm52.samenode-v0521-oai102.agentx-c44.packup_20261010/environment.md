# Environment

Captured 2026-10-10 ~13:50 UTC on the node that produced all three passing rounds, while the
e2c stack was still up. Raw captures: `results/env/env-102.yihou.txt` (skill `collect_env.sh`)
and `results/env/env-102-image.yihou.txt` (image + in-container versions).

## Hardware — `crsuse2-m2m-102` (both legs on one node)

| | |
|---|---|
| allocation | spur job **202546** `yihou-oai-go2`, partition `amd-spur`, account `amd-oai`, QOS `amd-oai-qos`, 24 h, started 2026-10-10T05:25:53Z. Hold script `scripts/hold_inf.yihou.sh` (`exec sleep infinity`). |
| roles | prefill = GPUs 0-3, decode = GPUs 4-7, router + etcd on the same host |
| GPUs | 8 × AMD Instinct MI355X (gfx950), 309,220,868,096 B VRAM each |
| GPU driver | amdgpu 6.14.14 (rocm-smi "Driver version") |
| CPU / RAM | 2 × AMD EPYC 9575F 64-core; 2.7 TiB RAM (MemTotal 2,885,471,972 kB) |
| OS / kernel | Ubuntu 24.04.4 LTS, `6.8.0-107-generic` |
| data-plane NIC | `ens3` (`mlx5_0` netdev), IP **10.245.147.130** — all engine/bootstrap/etcd/router traffic binds here |
| RDMA devices | `mlx5_0` + `ionic_0..7` (400 Gb/s, Active). **None used**: KV rides XGMI via HIP IPC (`MC_TE_FILTERS=<non-existent>` → `Found 0 HCAs`). |
| kernel knobs (first-hand) | `kernel.numa_balancing=0`; THP `[never]`; `pid_max=4194304`; ephemeral ports 32768-60999 (reserved 30000-32767) |
| docker policy | spur-managed dockerd with **`spur-authz` plugin: denies `docker run --pid=host`** (`denied [N1]`) |
| GTT (TTM) limit | 1,477,361,647,616 B = MemTotal/2, node-wide (only relevant to the optional PR #42553 arm) |

## Software

| | |
|---|---|
| image (used by both legs and router) | `infera-sglang:v0521-yihou-1010-mc4340`, image ID on 102 **`sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2`**, created 2026-10-10T06:36:03Z |
| base image | `lmsysorg/sglang-rocm:v0.5.21-rocm720-mi35x-20261008`, digest **`sha256:6c2c0b0e2afed9924d6f3bd621c554e83e6ffd37a1bf559f900cfbda7eb763ef`**, created 2026-10-08T14:41:32Z |
| recipe | `patches/image-recipe/files/Dockerfile.sglang.v0521-mc4340.yihou` (+ the 4 recipe commits in `patches/image-recipe/`) |
| infera repo for the build | private clone `/shared_nfs/yihou_glm52/agentx44_20261009/infera-src.yihou`, branch `yihou.agentx44.20261009` @ **`68f06e29b8bd8267330d042268201b4ab349a4e5`** = `infera.glm52.pd` @ `28121cb4` + `e3e46f32` + `fb8ae9c9` + `cdba2446` + `68f06e29` (clean tree) |
| sglang | `0.5.21.dev20261008+g11b1fb21de`; tree `/sgl-workspace/sglang` HEAD `9f5d8712514c2d9dd29c76c6080774db3fd50aa2` (2026-10-08) + recipe patches |
| sglang-kernel | 0.4.9 |
| ROCm (in image) | 7.2.0; torch `2.9.1+rocm7.2.0.git7e1940d4` (HIP 7.2.26015-fc0010cf6a); triton 3.7.0 |
| aiter | `amd-aiter 0.1.23.dev183+ge7d2453f2.d20261008` (`AITER_COMMIT=e7d2453f25e5aaeea72a3a40b284ba453e026624`) |
| mooncake (LOADED engine) | `/opt/venv/lib/python3.10/site-packages/mooncake/engine.cpython-310-x86_64-linux-gnu.so` sha256 **`3c96489a0ca4c8662a5f6512c1d9d44653835afc3d7cfaa9082b70f575bb289a`** = faae8dd4 + Mooncake PR #4340 head e8062249 port. (`/sgl-workspace/Mooncake` @ 01d1eb2a is the base's leftover source, not the loaded engine.) |
| HIP/ROCr libs | byte-identical across v0519 / v0521-1009 / mc4340 images (libamdhip64 `36fb0b13…`, libhsa-runtime64 `c534feb1…`) — repro, R020 |
| etcd | `quay.io/coreos/etcd:v3.5.14` |
| bench client | AgentX via InferenceX `https://github.com/SemiAnalysisAI/InferenceX.git` @ `918524ff94045b3f091115f1051c22a8588edf2b` (harness `config.sh`), dataset `semianalysisai/cc-traces-weka-062126` (HF, public, split train, 393 entries) |
| model | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (real 256-expert checkpoint, no overlay), served name `glm5.2-mxfp4` |
| sibling image | `infera-sglang:v0521-yihou-1009` (102 ID `sha256:3f917e9b…`) = same recipe without #4340; used by the failed e2b/e2b2 rounds, not by any pass |

## Container runtime options (from `scripts/harness/bench-harness/engine.sh` + actual docker cmd)

- `--network host --ipc host --shm-size 32g --device /dev/kfd --device /dev/dri --device /dev/infiniband
  --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --init`
- **Private PID namespace per container** (no `--pid host`, denied on spur nodes); decode command wrapped
  by the PID-offset loop (`DECODE_PID_OFFSET=30000`).
- Role-scoped `HSA_NO_SCRATCH_RECLAIM`: prefill 0, decode 1. `HIP_VISIBLE_DEVICES` 0-3 / 4-7.
- Per-node AITER/triton JIT caches under `/shared_nfs/yihou_glm52/v0521dbg_20261010/cache/102/…`
  (cold cache adds ~25 min to the first prefill bring-up).

## Gaps

- The exact `docker build` command line for the mc4340 image is recorded in `REPRODUCE.md §1`
  as reported by the builder teammate (see notes there); the build log is referenced by path,
  not copied.
