# Environment

Identical to `agenticbench.mtp.sweep.packup_20260801/environment.md`, which is
the authoritative copy — the same two held nodes, the same images, the same
fabric, the same repo state, in one continuous session. Only the run-specific
deltas are recorded here.

## Nodes

| role | job | host | ens3 | image |
|---|---|---|---|---|
| prefill + etcd + router + kvd | **24300** | `crsuse2-m2m-253` | `10.245.157.89` | `sha256:42a303e5…` |
| decode + kvd(off) | **24301** | `crsuse2-m2m-236` | `10.245.146.87` | `sha256:ff7b02eb…` |

8 × MI355X `gfx950` per node, ROCm 7.2.0 in-image, sglang 0.5.15.post1,
torch 2.9.1+rocm7.2.0. KV transport `mlx5_0`, GID 3, dma-buf ON.

Differing image ids are expected — each node built independently; equivalence is
established by the 8-assertion bytecode gate, which passed on both.

## Deployment under test

    two-node PD over mooncake RDMA        (mlx5_0, GID 3, dma-buf ON)
    DP-attention 8/8 both legs
    kv-aware routing                      ON  (prefill 20.0 / decode 2.0)
    kvd (infera HiCacheStorage)           PREFILL ON / DECODE OFF
    MTP (EAGLE)                           DECODE ON: steps 3, topk 1, draft 4
    --disable-custom-all-reduce           ON, BOTH arms (see REPRODUCE.md §7)
    --context-length                      262144
    --chunked-prefill-size                65536   (8192 per rank at dp 8)
    --hicache-size                        32      (absolute GB)
    index_share_for_mtp_iteration         True    (model default, NOT overridden)
    KV pool                               3,260,992 tokens/rank (167.72 GB, fp8_e4m3)

`index_share_for_mtp_iteration` is called out because `exp2_indexshare_off`
identifies turning it **off** as a route around this bug class. Confirmed here as
`json_model_override_args='{}'`, i.e. the model's own default of `True`.

## Load generator

| | |
|---|---|
| driver | `Optimus-AgenticBench @ 1cf01cbf`, branch `fix/realistic-profile-session-driver` |
| entry | `python3 -m agent.agent_throughput --mode realistic --dashboard-mode` |
| venv | `/shared_nfs/yihou_agentbench/venv` (login node) |
| workload | `workloads/caseA_full.yaml`, derived from `glm52_crxx_caseA.fix.yaml` |
| driven from | the **login node** — the bench is HTTP + tokenizer, and running it inside the prefill container would place the load generator on the host it measures |

Client-side constant that shapes the error counts: `aiohttp.ClientTimeout(total=240)`
is hardcoded at `agent_throughput.py:928`. Every error in every attempt here is
that timeout; there are no server-returned failures.

## Timeline of this session's runs

| UTC | what | outcome |
|---|---|---|
| 11:54–12:09 | Case A probe, rate 0.10, 900 s | **PASS** — 509 completed, 3 errors |
| 12:12:43 | Case A full, rate 0.15 | healthy to t=556 s, then **froze** |
| 12:22:06 | prefill `TCPStore recvValue failed` | first observable failure |
| 12:27 | attempt 1 aborted | 401 completed, 42 errors |
| 12:31:00 | Case A full, rate 0.115 | **void** — decode already dead |
| ~13:20 | decode leg killed, GPUs polled to 0 %, rebooted (MTP on) | decode healthy |
| 13:32:53 | Case A full, rate 0.15 (arm A) | **0 completions** — prefill now dead (see notes §3) |

## Artifacts outside this kit

| what | where |
|---|---|
| model weights | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` |
| bench outputs | `/shared_nfs/yihou_agbench_mtp/bench/caseA_*` |
| engine logs (full) | `/shared_nfs/yihou_agbench_mtp/logs/` |
| the successful sweep | `agenticbench.mtp.sweep.packup_20260801/` |

## Secrets

None beyond the sweep kit's: registry login via `DOCKER_CONFIG=/tmp/dockercfg`,
and spur job allocation (`spur exec`; ssh to compute nodes is blocked). etcd is
unauthenticated on a private data-plane IP. No secret value appears in this kit.

## Gaps

- **The MTP-off control arm was not run** — the single most important missing
  measurement. Procedure in `REPRODUCE.md` §6.
- **The live hang was not probed beyond py-spy.** ZMQ socket state, the prefill
  bootstrap queue, and `ss -tn` were not captured before the restart, on the
  operator's instruction to prioritise getting data. Those are unrecoverable for
  this occurrence.
- **No repeat of the probe** after the hang, so it is unknown whether a
  post-hang deployment still passes the 900 s calibration.
