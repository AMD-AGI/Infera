# Environment

GLM-5.2-MXFP4 + DCP (decode context parallelism) + SGLang on MI355X, mixed (single server) and PD-disaggregated
(Infera harness), 2026-10-21 12:38Z .. 2026-10-22 (ongoing at the time of writing, 02:3xZ).

Source abbreviations used below:
- `WS` = local workspace `/home/yihou/dev/git.16-19/sglang.pd.pp.dcp/yihou_ws/glm52_dcp_pd_20261021`
  (`mission.md`, `poll.md`, `watch.md`, `ledger.{pdrun,mixrun,benchrun,coder2,analyst}.md`, `RESULTS.md`).
- `CL` = cluster workspace `/shared_nfs/yihou_glm52/dcp_pd_20261021` (`pd/`, `mixrun/`, `bench/`, `src/`, `cache/`).
- "captured 10-22 02:29Z" = first-hand `docker image inspect` / `/sys/module/amdgpu/version` / `uname -r` /
  `rocm-smi` / `sacct` / `squeue` run by the pack-up on 2026-10-22 02:29Z on v3-060 and v3-070 via `spur exec`.

## Hardware

All nodes: 8 x AMD Instinct MI355X (gfx950) per node, AMD Crusoe "spur" cluster (`rocm-smi` Card Series on v3-060 /
v3-070: captured 10-22 02:29Z; 054: per the 22k pack-up node class, not re-captured). GPU NUMA on the v3 nodes:
GPUs 0-3 -> node0 (CPUs 0-117), GPUs 4-7 -> node1 (CPUs 118-235) (`WS/ledger.mixrun.md` "Node v3-045";
`WS/ledger.benchrun.md` header).

| Node | Spur job | Partition / QOS | Job window (UTC) | amdgpu | Kernel / OS | IP | Used for |
|---|---|---|---|---|---|---|---|
| crsuse2-m2m-054 | 228279 (OAI, lent by pool manager) | amd-spur / amd-oai-qos | 2026-10-21T01:20:41 .. 2026-10-22T01:21:19, TIMEOUT (sacct, captured 10-22 02:29Z). Our use ~12:38Z .. 00:31Z (containers stopped 00:29-00:31Z, `WS/ledger.pdrun.md` P015) | **6.14.14** (`WS/poll.md` 14:17Z; node gone, not re-captured) | not captured | 10.245.144.125 (`CL/pd/runs/pdBeperf-r4/agentx-c40-A2/runtime.env`) | mixed eager bring-up (mixrun R1-R3); PD rounds pdBe-r1, pdBeperf-r1/r2/r3/r4 (pdrun) |
| crsuse2-m2m-268 | 227779 (OAI) | amd-spur | 2026-10-21T01:11:47 .. 13:05:59, NODE_FAIL / NodeDown (sacct; `WS/ledger.pdrun.md` P007) | - | - | 10.245.145.242 (topology, `WS/ledger.pdrun.md` P006) | **nothing ran** (node failed before hand-over) |
| crsuse2-m2m-v3-045 | 229517 (burst, preemptible) | amd-spur-v2 / amd-burst-qos | 2026-10-21T13:15:01 .. 2026-10-22T01:15:14, TIMEOUT (sacct). Our use ~13:36Z .. ~00:05Z (`WS/ledger.mixrun.md` paired AgentX section) | **6.16.13** (`WS/poll.md` 14:17Z) | not captured (node gone) | not captured | mixed graph + AgentX (full1-dcp4-c40 26,277 etc.), custom-AR A/B, DP2+DCP2 bug chain, paired kvb run (mixrun) |
| crsuse2-m2m-v3-060 | 229524 (burst, preemptible) | amd-spur-v2 / amd-burst-qos | start 2026-10-21T13:22:49, end 2026-10-22T13:22:49 (squeue, captured 10-22 02:29Z); our containers stop by 13:02Z (`WS/watch.md` 02:24Z) | **6.16.13** (captured 10-22 02:29Z) | `6.8.12-322.crusoe.amd64-g275dd86317a7`, Ubuntu 24.04.5 LTS (captured) | 10.245.155.84 (`WS/ledger.pdrun.md` P016) | benchrun T1-T16 microbenchmarks (from 16:10Z); PD r5 / r5b (pdrun, from 00:30Z) |
| crsuse2-m2m-v3-070 | 230266 (burst, preemptible) | amd-spur-v2 / amd-burst-qos | start 2026-10-21T23:46:39, end 2026-10-22T23:46:39 (squeue, captured); our stop deadline 23:26Z (`WS/watch.md` 02:24Z) | **6.16.13** (captured 10-22 02:29Z) | `6.8.12-322.crusoe.amd64-g275dd86317a7`, Ubuntu 24.04.5 LTS (captured) | 10.245.148.77 (`WS/ledger.mixrun.md` E2 header) | PD E2 = DP2 x DCP2 decode (pdE2-r1 .. r6, pdE2-p1; mixrun) |

Pending OAI replacement jobs 229615, 229857, 230267 (amd-spur / amd-oai-qos, PENDING at 10-22 02:29Z, squeue) never ran.

**Burst v3 nodes (amdgpu 6.16.13) need `HSA_ENABLE_SDMA=0`** on every engine container: without it, weight load hangs
in HIP `memcpy_and_sync` (futex, `kfd_wait_on_events`, GPU idle). Found on v3-045 (`WS/ledger.mixrun.md` R5-R8, fixed
R8; `WS/poll.md` 14:17Z), reproduced on v3-060 (`WS/poll.md` 16:39Z, `WS/ledger.benchrun.md`). 054 (6.14.14) ran with
SDMA on. All v3 results are labelled "v3/SDMA=0". Also on v3-060, `DEBUG_AGENT=1` (ROCm debug agent) aborted a prefill
scheduler (`rocm-dbgapi: fatal error: os_queue_packet_id ...`), so r5b and later use `DEBUG_AGENT=0`
(`WS/ledger.pdrun.md` P016). Not seen on 054; the cause is not established.

Data plane: same-node PD, Mooncake over XGMI via HIP IPC (as the 22k pack-up). RDMA NICs present but not used.

## Software

### Docker images (IDs captured first-hand 10-22 02:29Z on v3-060 and v3-070; identical on both)

| Image | ID / digest | Created | Used by |
|---|---|---|---|
| `infera-sglang:v0522-1020-yihou-dcp-mc4340` | `sha256:5b6565ef3e4e7dcca02423d509f842a74cf7416b18ed94edbf12e2be100f7376` (same ID built on 054, `WS/ledger.pdrun.md` P003; `IMAGE_IDS` in `CL/pd/runs/pdBeperf-r4/agentx-c40-A2/runtime.env`) | 2026-10-21T12:48:22Z | all PD rounds, both legs + router (054, v3-060, v3-070). Locally built, no registry digest; transported as `CL/pd/images/infera-sglang_v0522-1020-yihou-dcp-mc4340.tar` (23 GB, `docker load`) |
| `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020` | `sha256:3a75c91365cc4f4e1194bf5d64b92f2f1ef134d8b9045da165fba28a8dada464` (= RepoDigest; also `WS/ledger.mixrun.md` M0, `WS/ledger.pdrun.md` P001) | 2026-10-20T14:45:04Z | base of the PD image; used directly (with our source mounted read-only) for all mixed runs (mixrun) and microbenchmarks (benchrun) |
| `quay.io/coreos/etcd:v3.5.14` | `sha256:42106a001d1be7fd22e48830071ceae0fe760ff2c5df8c76c0e3b586d8a1f0fc` | 2024-05-30 | PD harness etcd |

PD image build (`WS/ledger.pdrun.md` P003): own clone `CL/pd/build/infera-src.yihou`, branch `yihou.dcp.v0522.20261021`
= `68f06e29` (the 22k image recipe; `AMD-AGI/Infera` base `28121cb4...` + local commits, see the 22k pack-up
`environment.md`) + `0a54d223` (`Dockerfile.sglang.v0522-1020-mc4340.yihou`, base ARG bumped to v0.5.22-20261020) +
`80458804` (`scripts/build_mooncake_sglang.mc4340.v0522.yihou.sh`: installs yalantinglibs, absent in the v0522 base).
Mooncake rebuilt with the PR #4340 port; loaded `engine.so` sha256 prefix `dad1eeeb...` (full hash not recorded in the
ledger; the 22k v0521 image had `3c96489a...`). Verify scripts `verify_v0521.sh` + `verify_mc4340.sh`: fail=0
(`CL/pd/logs/verify.v0522-1020-mc4340.log`).

In-image versions of the v0.5.22 base (`WS/ledger.mixrun.md` M0): sglang `0.5.22.dev20261020+geae134d61f`
(`/sgl-workspace/sglang` HEAD `e262b44eef`; contains #43735), aiter `0.1.25.dev178+g2bb549102.d20261020`,
triton `3.7.0+amd.rocm7.2.0.git89002410`, torch `2.9.1+rocm7.2.0`, flydsl 0.3.4.1, sglang-kernel 0.4.10, ROCm 7.2.0.
The image's sglang `eae134d61f` is an ancestor of our base `fda84f80b7` (90 commits behind).

### sglang source actually run

The in-image sglang is shadowed by a read-only mount of `git archive python/` of our branch:
`CL/src/sglang-<sha>/python/sglang` -> `/sgl-workspace/sglang/python/sglang` (mixed: `mixrun/serve.sh`; PD:
`SGLANG_OVERLAY` / `PD_OVERLAY` in `CL/pd/harness/config.yihou.dcp.sh`). Always `cd /tmp` before `import sglang`
(`WS/ledger.mixrun.md` M0 gotcha).

Repo: worktree `/home/yihou/dev/git.16-19/sglang.pd.pp.dcp`, branch **`dev/yihou/glm52_dcp_pd`**, HEAD `93034eba91`
= origin/main `fda84f80b7` + PR #44322 merge (`4792dc0e52`) + our commits. DP2 fixes were first made on side branch
`dev/yihou/glm52_dcp_pd_dp2`, then cherry-picked (`WS/RESULTS.md` §4). Branch membership below checked with
`git branch --contains` on 10-22.

| sha | Branch | What it adds | Runs on it (source) |
|---|---|---|---|
| `4792dc0e52` | main branch | origin/main fda84f80b7 + #44322 merge | mixed eager bring-up on 054 R1-R3 (`WS/ledger.mixrun.md` M0); PD overlay `sglang-4792dc0e52-pdrecipe` (dryruns only, P004) |
| `ae529c96bf` | main branch | + 9 Infera recipe commits + PD DCP commits a6bdc7894d / 228263d116 / ae529c96bf | PD pdBe-r1, pdBeperf-r1/r2/r3/r4 (054; overlay grep of `CL/pd/harness/env.*`, `WS/ledger.pdrun.md` P008); mixed v3-045 R4-R11 incl. full1-dcp4-c40 26,277, ref c40, dcp4 c48, custom-AR A/B (`WS/ledger.mixrun.md`) |
| `6ee35f3617` | main branch | + PP non-final chunk wait | benchrun T1-T8 (`WS/ledger.benchrun.md` header) |
| `6ffb90ae6b` | main branch | + split-indexer owned pages/lens once | benchrun T9 |
| `31309dc4f1` | dp2 side branch | + batch-major pad fix | mixrun DP2+DCP2 retest |
| `fdf73b6fb2` | dp2 side branch | + ROCm eager-idle indexer short circuit | mixrun DP2+DCP2 |
| `bd6a337ab7` | dp2 side branch | + bounds check (debug only) | mixrun DP2 diagnostic |
| `fa810b9f69` | dp2 side branch | + `extend_seq_lens_cpu` new-list fix | mixrun DP2+DCP2 eager/graph gate and decodebench vs DCP4 |
| `636cc68dd0` | main branch | + `SGLANG_DSA_DCP_KEEP_KV_B_BF16` (+ #45483 idlefix 5fb03011d6) | benchrun T10, T11 |
| `9af2166f3f` | main branch | + `SGLANG_LOG_DECODE_SEQ_LENS` | mixrun index_share A/B |
| `da6e383513` | main branch | + DP2 fixes cherry-picked, triton split pack/merge 0347a6d47e | benchrun T12 |
| `5aacce1f1c` | main branch | + `extend_seq_lens_cpu` fix cherry-pick | mixed paired AgentX full-kvb-dcp4-c40 26,152 / DP2 (no score) |
| `a6ea78e322` | main branch | + radix-select merge (failed to compile on GPU) | benchrun T14 |
| `93034eba91` | main branch (HEAD) | + radix-select constexpr fix; hang fix c4eda0343e | benchrun T15-T16; PD r5 gate / r5 / r5b (v3-060, `WS/ledger.pdrun.md` P016); PD E2 pdE2-r1..r6, pdE2-p1 (v3-070, `WS/ledger.mixrun.md` E2) |

### Benchmark client

| | |
|---|---|
| InferenceX | `https://github.com/SemiAnalysisAI/InferenceX.git` @ **`918524ff94045b3f091115f1051c22a8588edf2b`**, `INFERENCEX_DIRTY=false` (`CL/pd/runs/pdBeperf-r4/agentx-c40-A2/runtime.env`; `git rev-parse` of `CL/cache/InferenceX`, captured 10-22) |
| aiperf | **0.12.0** (`aiperf-0.12.0.dist-info` in `CL/pd/cache/{054,060}/agentx/aiperf/venv`, python 3.11; venv created by InferenceX `install_agentic_deps`) |
| scenario | `aiperf profile --scenario inferencex-agentx-mvp ... --concurrency 40 --benchmark-duration 3600 --warmup-requests-per-lane 10 --random-seed 42 --use-server-token-count` (`CL/pd/runs/pdBeperf-r4/agentx-c40-A2/benchmark_command.txt`) |
| dataset | `semianalysisai/cc-traces-weka-062126` (HF, public; cached as `datasets--semianalysisai--cc-traces-weka-062126` under `CL/pd/cache/054/agentx/hf/hub` and `CL/cache/agentx-hf/hub`) |
| scoring | PD: total tok/s / 8 chips (same as the 22k pack-up). Mixed: total / 4 GPUs (InferenceX non-multinode convention). Not directly comparable (`WS/RESULTS.md` §2) |
| accuracy tool | sgl-eval 0.1.3, preset `zai-org/GLM-5.2-FP8` (partial eager GSM8K only; `WS/ledger.mixrun.md` R2) |

### Model

`/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`, served name `glm5.2-mxfp4`, also used as the aiperf tokenizer
(`CL/pd/harness/config.yihou.v0521-base.sh:74-76`; `benchmark_command.txt`).

### Container runtime notes

- PD: harness `run.v2` / `engine.sh` from the 22k kit (same docker options: host network/IPC, `/dev/kfd`, `/dev/dri`,
  private PID namespace, decode `DECODE_PID_OFFSET=30000`); container names `yihou-dcp-pd-<node>-<round>-*`
  (`WS/ledger.pdrun.md` P005).
- Mixed / bench: `mixrun/serve.sh` (`docker --rm -d`), `--cpuset-cpus` per GPU half + `SGLANG_SET_CPU_AFFINITY=0`,
  `PID_OFFSET=30000` for the second container on a node, one aiter JIT dir per GPU slot (`CL/cache/<node>/...`)
  (`WS/ledger.mixrun.md` Launcher, R1-R2, v3-045 section).

## External dependencies (absolute paths)

| Path | What |
|---|---|
| `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` | model weights + tokenizer (not ours; read only) |
| `/shared_nfs/yihou_glm52/dcp_pd_20261021/pd/harness` | PD harness (copy of `/shared_nfs/yihou_glm52/v0521dbg_20261010/harness` + `config.yihou.dcp.sh`, `config.yihou.dcp-e2.mixrun.sh`, `mk_env*`, `launch_pd`, `pd_seq`, `bench_retry`, ...) |
| `/shared_nfs/yihou_glm52/dcp_pd_20261021/pd/images/infera-sglang_v0522-1020-yihou-dcp-mc4340.tar` | PD image tarball (23 GB) |
| `/shared_nfs/yihou_glm52/dcp_pd_20261021/pd/build/infera-src.yihou` | Infera clone used to build the PD image |
| `/shared_nfs/yihou_glm52/dcp_pd_20261021/src/sglang-<sha>/python/sglang` | mounted sglang sources (one dir per sha above) |
| `/shared_nfs/yihou_glm52/dcp_pd_20261021/cache/InferenceX` | InferenceX checkout @ 918524ff (target of `pd/cache/InferenceX`) |
| `/shared_nfs/yihou_glm52/dcp_pd_20261021/pd/cache/<node>/agentx/{aiperf,hf}`, `/shared_nfs/yihou_glm52/dcp_pd_20261021/cache/agentx-hf` | aiperf venv, HF dataset cache |
| `/shared_nfs/yihou_glm52/dcp_pd_20261021/cache/<node>/`, `.../pd/cache/<node>/` | aiter / triton JIT caches (per node, per GPU slot or role) |
| `/shared_nfs/yihou_glm52/dcp_pd_20261021/{mixrun,bench}` | mixed and microbenchmark scripts, logs, results |
| `/shared_nfs/yihou_glm52/agentx44_20261009/cache/InferenceX` | older shared InferenceX cache (same commit); r1 bench ran `git` there (`WS/ledger.pdrun.md` P011) |

## Required secrets (no values)

- Cluster access: your own ssh login to the spur login nodes (`crsuse2-slog-00X`) and `spur exec` rights on the job.
  Allocation goes through the pool-manager session (`WS/mission.md` §2).
- None found in the run scripts: `grep` for `HF_TOKEN|HUGGING_FACE_HUB_TOKEN|GH_TOKEN|GITHUB_TOKEN` over `mixrun/*.sh`,
  `bench/*.sh`, `pd/harness/*.sh`, `pd/tools/*.sh` returned nothing (10-22). The base image (Docker Hub), InferenceX
  (GitHub) and the dataset (HF) are public; the model is on NFS.
- Only if re-cloning the Infera repo instead of using the NFS clone: read access to `git@github.com:AMD-AGI/Infera.git`
  (your GitHub ssh key).

## Gaps

- 054 and v3-045 kernel versions, OS and full driver info were not captured (jobs ended before the pack-up); amdgpu
  versions for them come from the leader's `poll.md` 14:17Z entry, not a pack-up capture.
- Mooncake `engine.so` full sha256 is not recorded (prefix `dad1eeeb` only).
- Pending at the time of writing: PD r5b A1/A2 (v3-060) and E2 perf pdE2-p1 (v3-070); their environment is as above.
