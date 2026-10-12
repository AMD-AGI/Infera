# Environment

Headline run e2g A2: 2026-10-11 07:16-08:35 UTC on `crsuse2-m2m-102`. Image IDs and versions re-captured first-hand
on 2026-10-12 04:4x UTC (`docker image inspect`, in-container `python3 -c 'import sglang,torch'`); the full node
captures from 2026-10-10 are in `results/env/` (skill `collect_env.sh` output + image/in-container versions) — the node
and image did not change between then and the headline run.

## Hardware — `crsuse2-m2m-102` (both legs on one node)

| | |
|---|---|
| allocation | spur jobs **202546** (`yihou-oai-go2`, `sleep 86400` hold, ended 2026-10-11 05:25:53Z) → **207495** (`yihou-oai-go3`, `scripts/alloc/hold_inf.yihou.sh` = `exec sleep infinity`, started 05:26:36Z, TimeLimit raised to 2 days); partition `amd-spur`, account `amd-oai`, QOS `amd-oai-qos`, `--gres=gpu:8 -c 160` |
| roles | prefill = GPUs 0-3 (NUMA node 0), decode = GPUs 4-7 (NUMA node 1); router + etcd on the same host |
| GPUs | 8 × AMD Instinct MI355X (gfx950), 288 GiB each; KFD: 24 compute HQDs per GPU, `sched_policy=0` (HWS), `mes=0`, CWSR on |
| GPU driver | amdgpu DKMS **6.14.14** (`/sys/module/amdgpu/version`) |
| CPU / RAM | 2 × AMD EPYC 9575F 64-core (SMT2, 236 logical CPUs online: node0 0-117, node1 118-235); 2.7 TiB RAM |
| OS / kernel | Ubuntu 24.04.4 LTS, `6.8.0-107-generic` |
| data-plane | `ens3` 10.245.147.130 (engines, bootstrap, etcd, router). RDMA (`mlx5_0`, `ionic_0..7`) present but **unused**: KV rides XGMI via HIP IPC (`MC_TE_FILTERS=<non-existent>` → `Found 0 HCAs`) |
| docker policy | spur-managed dockerd + `spur-authz`: denies `--pid=host`; allows `--cpuset-cpus/--cpuset-mems`, `--cap-add SYS_PTRACE` with `--pid container:<name>` |
| login hosts | runs driven from `crsuse2-slog-005` (until it became unreachable during e2j, 2026-10-11 ~14:31Z) and `crsuse2-slog-006` (same $HOME / NFS) |

## Software

| | |
|---|---|
| image (both legs + router) | `infera-sglang:v0521-yihou-1010-mc4340`, ID **`sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2`**, created 2026-10-10T06:36:03Z |
| base image | `lmsysorg/sglang-rocm:v0.5.21-rocm720-mi35x-20261008`, digest **`sha256:6c2c0b0e2afed9924d6f3bd621c554e83e6ffd37a1bf559f900cfbda7eb763ef`** |
| Dockerfile | `patches/image-recipe/files/Dockerfile.sglang.v0521-mc4340.yihou` (+ `build_mooncake_sglang.mc4340.yihou.sh`, `mooncake-pr4340-e8062249.faae8dd4-port.yihou.diff`) |
| build repo | clone `/shared_nfs/yihou_glm52/agentx44_20261009/infera-src.yihou`, branch **`yihou.agentx44.20261009`**: base `infera.glm52.pd` **`28121cb40ef0b03bf7b8481fde214562d634a9a0`** (on `origin/dev/pd_opt/glm_5.2_agentx`, `git@github.com:AMD-AGI/Infera.git`) + `e3e46f32` router fix + `fb8ae9c9` v0521 recipe + `cdba2446` mc4340 recipe + `68f06e29` pipefail fix (**the 22k image was built at `68f06e29`**) + `702af10f` (mc4340prio, later, unused). Local only (not pushed): patches + bundle in `patches/image-recipe/` |
| sglang | `0.5.21.dev20261008+g11b1fb21de` (tree `/sgl-workspace/sglang` HEAD `9f5d8712…`) + recipe patches + 3 runtime mounts (`patches/runtime-mounts/`) |
| torch / ROCm | `2.9.1+rocm7.2.0.git7e1940d4`, ROCm 7.2.0 (HIP 7.2.26015); triton 3.7.0 |
| aiter | `amd-aiter 0.1.23.dev183+ge7d2453f2.d20261008` |
| mooncake (loaded engine) | `…/site-packages/mooncake/engine.cpython-310-x86_64-linux-gnu.so` sha256 `3c96489a0ca4c8662a5f6512c1d9d44653835afc3d7cfaa9082b70f575bb289a` = faae8dd4 + Mooncake PR #4340 port |
| key server args (first-hand, server_args in `logs/e2g/*.gz`) | both legs: `kv_cache_dtype fp8_e4m3`, `page_size 64`, `tp_size 4`, `enable_dp_attention True`, `dsa_{prefill,decode}_backend triton`, `max_running_requests 64`, `json_model_override_args {"index_share_for_mtp_iteration":false}`; prefill: `mem_fraction_static 0.85`, `hicache_size 450`; decode: `mem_fraction_static 0.95` (2,762,176 KV tokens/rank), EAGLE `5/1/6`, `SGLANG_SIMULATE_ACC_LEN=3.61` |
| other images present (not used for the 22k) | `infera-sglang:v0521-yihou-1010-mc4340prio` `sha256:aacc9aeb46cebb411d29e99e6ce2afafe46349ca74f78d0d18d6e8b804624fcd` (mc4340 + MC_HIP_STREAM_PRIORITY gate, commit 702af10f) |
| etcd | `quay.io/coreos/etcd:v3.5.14` |
| bench client | AgentX via InferenceX `https://github.com/SemiAnalysisAI/InferenceX.git` @ `918524ff94045b3f091115f1051c22a8588edf2b`; dataset `semianalysisai/cc-traces-weka-062126` (HF, public); full mode 3600 s, warmup 10/lane, CONC 40 |
| model | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`, served name `glm5.2-mxfp4` |

## Container runtime options (engine.sh)

`--network host --ipc host --shm-size 32g --device /dev/kfd --device /dev/dri --device /dev/infiniband --cap-add SYS_PTRACE
--cap-add IPC_LOCK --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --init`; private PID namespace per container
with the decode command wrapped by a PID-offset loop (`DECODE_PID_OFFSET=30000`); `HIP_VISIBLE_DEVICES` 0-3 / 4-7;
per-node AITER/triton JIT caches under `/shared_nfs/yihou_glm52/v0521dbg_20261010/cache/102/…`. The base image sets
`SGLANG_SET_CPU_AFFINITY=1` (schedulers pinned by local gpu_id — R045; not changed for the 22k).

## Gaps

- `engine.so` sha256 was not re-hashed on 2026-10-12 (the `find` inside the image returned nothing on that attempt);
  the value above is the 2026-10-10 first-hand capture (`results/env/env-102-image.yihou.txt`), same image ID.
- `/tmp` scratch scripts on the lost login host 005 (e.g. `/tmp/yihou_decstat.py`, `/tmp/yihou_wf_c40/*.py`) could not be
  recovered; their successors are in `scripts/diag/` and `scripts/harness/tools/decode_stats.yihou.py`.
