# Environment

Captured on both nodes with `scripts/collect_env.sh`; raw output in
`env/<node>.txt`. Everything below is from that capture or from the run's own
artifacts, not from memory.

## When

2026-09-18. Image layer built 10:38–10:40 UTC; stack up 11:04–11:17; correctness
probe 11:17; AgentX C40 11:18–11:57 (profiling window 1229.3 s); teardown 12:03.

## Nodes

| role | host | data IP | GPUs used |
|---|---|---|---|
| prefill + control + builder | `crsuse2-m2m-135` | `10.245.148.209` | 2,3,4,5 |
| decode | `crsuse2-m2m-138` | `10.245.157.237` | 2,3,4,5 |

Topology file: `scripts/topology.yihou.tsv`. The repo's tracked `topology.tsv`
lists 136/137/140 and must be overridden with `TOPOLOGY=`.

## Hardware

| | |
|---|---|
| GPU | AMD Instinct **MI355X** ×8 per node, `gfx950`, model `0x75a3` |
| GPU driver | **6.14.14** |
| CPU | 236 logical cores |
| RAM | 2.7 TiB |
| OS | **Ubuntu 24.04.4 LTS** |
| kernel | **6.8.0-107-generic** |

**Not idle:** `crsuse2-m2m-135` GPU[1] carries a root-owned Kubernetes vLLM pod
(`pod5d84e491`, Qwen3-32B, ~96.9 GB). It is untouched by this run, which is one
of the two reasons GPUs 2,3,4,5 were chosen rather than 0-7.

## Fabric

Eight isolated 400 Gb/s RoCE rails per node, `ionic_0` … `ionic_7`, plus an
`mlx5_0`. `ionic_i` reaches only `ionic_i` — a mismatch between the two ends is
*unreachable*, not merely slow.

| | |
|---|---|
| rails used | `ionic_2`, `ionic_3`, `ionic_4`, `ionic_5` (PORT_ACTIVE, real netdev, non-zero sysfs GID[1] on both nodes) |
| GPU↔NIC pin | per-GPU JSON: local visible index `0`→GPU2→`ionic_2` … `3`→GPU5→`ionic_5` |
| GID index | `MC_GID_INDEX=1` |
| transport filter | `MC_TE_FILTERS=ionic_2,ionic_3,ionic_4,ionic_5` |
| host provider | `/lib/x86_64-linux-gnu/libionic.so`, bind-mounted into the engine containers |
| router affinity | `PD_DP_RANK_AFFINITY=1` |

**`crsuse2-m2m-135 ionic_7` is defective** — no netdev, sysfs GID index 1
all-zero. Its non-zero GID in `ibv_devinfo` is **stale**; sysfs is ground truth.
Avoiding GPU7 is the second reason for the 2,3,4,5 selection.

Measured with a pinned probe on today's image (`spec/preflight-report.md`), both
directions, GB/s: gpu2 29.9/28.9, gpu3 27.9/30.0, gpu4 40.4/41.3, gpu5 40.3/41.2.

## Software

| | |
|---|---|
| run image | `infera-sglang:v0519-yihou-0917-nextnfix-hicache` |
| image id, 135 | `sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35` |
| image id, 138 | `sha256:972d8fd952e97b3915d742072874b57a1796c10de1dc0cdcc85e92123167f5d8` |
| base | `infera-sglang:v0519-yihou-0917`, id `4190c3a99d0e` on both |
| base of base | `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260917`, digest `sha256:21c1cc9ab9b703cfe2bf4d62d5c2028654e5f54c810890d9d4b2019d1c31ca32` |
| in-image sglang | `0.5.19.dev20260917+ga9fb1c3238` |
| repo | `AMD-AGI/Infera`, branch `dev/pd_opt/glm_5.2_agentx`, HEAD `ad3b85d3838513010d7364309da3fcfebbf76518` |
| uncommitted | `bench/glm5p2_pd/engine.sh` — supplied as `patches/0001-engine-sh-extra-args-env.patch` |

**The two image ids differ by design.** Each node built its own layer on its own
locally-built base; nothing was pushed, so there is no shared digest and the tag
does not identify one blob across the two nodes. Reproduction rests on
`patches/Dockerfile.yihou.hicache` plus the base tag.

Image contents beyond the stock base:

1. the infera DSA patch set (7 patches; their `.orig` backups were compared
   base↔new on both nodes and are identical, so the thin layers did not disturb
   them);
2. the GLM NextN shared-experts-fusion fix (`patches/Dockerfile.yihou.nextnfix`,
   `patches/apply_nextn_fusion_fix.sh`);
3. upstream sglang PR **#37152** (`patches/pr37152.sources.yihou.diff`) — **open
   upstream, never merged, and upstream's own AMD ROCm CI on it is red**. Inert
   in this run, which has `PREFILL_HICACHE=0`.

## Model and data

| | |
|---|---|
| model | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (shared NFS, absolute path) |
| served name | `glm5.2-mxfp4` |
| KV dtype | `fp8_e4m3` |
| AgentX trace corpus | `semianalysisai/cc-traces-weka-062126` @ `23f152f6f0f9399a85901b89a6458def0ef16729`, 393 traces |
| InferenceX | fetched by `tools/ensure_inferencex.py` into `bench/glm5p2_pd/.cache/InferenceX` at the ref pinned in `config.sh` |

## Secrets

None are required for the deployment. AgentX pulls its trace corpus from Hugging
Face; if that dataset is gated for your account, `HF_TOKEN` must exist in the
control node's environment. **No secret value appears anywhere in this pack-up.**

## Run shape

P4+DPA / D4+DPA, TP4 / DP4 / EP1 on both legs, `mem_fraction_static 0.85`,
`chunked_prefill 32768`, max-running 64, CUDA-graph max BS 64, HiCache off on
both legs, `index_share_for_mtp_iteration=false`.

Decode speculative decoding: EAGLE, 5 steps, topk 1, 6 draft tokens,
**`--disable-custom-all-reduce`**, **no `SGLANG_SIMULATE_ACC_LEN`**.

Authoritative record of what each engine actually ran:
`results/server-info/{prefill,decode}-0.json`.
