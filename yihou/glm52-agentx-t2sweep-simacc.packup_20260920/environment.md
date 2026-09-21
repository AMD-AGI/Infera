# Environment

Hardware and fabric are identical to the T1 pack-up
(`glm52-agentx-c40-realacc.packup_20260918`) — same two nodes, same GPUs, hours
apart; `dsa.topk.indexer.bug.analysis.md` §5 records that a `collect_env.sh` diff
morning-vs-evening on the decode node showed only monotonic activity counters
differing (driver, kernel, OS, HCA inventory identical). The node captures in
`env/<node>.txt` are those Sept-18 snapshots, reused here for the unchanged HW.
**The software/run-shape section below is specific to T2 and is the part that
differs.**

## When

- **2026-09-19 to 2026-09-20** (UTC). Multi-run:
  - t2e A/B arm (custom-all-reduce ON): CONC=40 completed ~08:27, CONC=56 ~10:16.
  - t2f main arm (custom-all-reduce OFF): bring-up 11:52:46; CONC=40 ~13:27,
    CONC=56 ~15:09, CONC=72 ~17:26, CONC=96 ~21:57.
  - CONC=128: 1st attempt died ~23:42 (NFS-full), 2nd attempt hung 00:53:59,
    torn down ~01:16 on 2026-09-20.
- Per-run timestamps are in `spec/sweep-driver-log.md`.

## Nodes

| role | host | data IP | GPUs used |
|---|---|---|---|
| prefill + control + builder | `crsuse2-m2m-135` | `10.245.148.209` | 2,3,4,5 |
| decode | `crsuse2-m2m-138` | `10.245.157.237` | 2,3,4,5 |

Topology: `scripts/topology.yihou.tsv`. The repo's tracked `topology.tsv` lists
136/137/140 and MUST be overridden with `TOPOLOGY=`.

## Hardware

| | |
|---|---|
| GPU | AMD Instinct **MI355X** ×8 per node, `gfx950`, model `0x75a3` |
| GPU driver | **6.14.14** |
| CPU | 236 logical cores |
| RAM | 2.7 TiB |
| OS | **Ubuntu 24.04.4 LTS** |
| kernel | **6.8.0-107-generic** |

`crsuse2-m2m-135` GPU[1] carries a root-owned Kubernetes vLLM pod
(`pod5d84e491`, Qwen3-32B, ~96.9 GB), untouched — one of the two reasons GPUs
2,3,4,5 were used, not 0-7.

## Fabric

Eight isolated 400 Gb/s RoCE rails per node, `ionic_0`…`ionic_7`. `ionic_i`
reaches only `ionic_i` — a mismatch between ends is *unreachable*, not slow.

| | |
|---|---|
| rails used | `ionic_2..5` (PORT_ACTIVE, real netdev, non-zero sysfs GID[1] both nodes) |
| GPU↔NIC pin | per-GPU JSON: local visible index `0`→GPU2→`ionic_2` … `3`→GPU5→`ionic_5` |
| GID index | `MC_GID_INDEX=1` |
| transport filter | `MC_TE_FILTERS=ionic_2,ionic_3,ionic_4,ionic_5` |
| host provider | `/lib/x86_64-linux-gnu/libionic.so`, bind-mounted into the engine containers |
| router affinity | `PD_DP_RANK_AFFINITY=1` |

`crsuse2-m2m-135 ionic_7` is defective (no netdev, sysfs GID[1] all-zero; the
non-zero GID in `ibv_devinfo` is stale). Avoiding GPU7 is the second reason for
the 2,3,4,5 selection.

## Software — NOTE: T2 ran the BASE image, not the thin layer

| | |
|---|---|
| **run image** | **`infera-sglang:v0519-yihou-0917`** (the base — id `sha256:4190c3a99d0e…` on both nodes) |
| base of base | `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260917`, digest `sha256:21c1cc9ab9b703cfe2bf4d62d5c2028654e5f54c810890d9d4b2019d1c31ca32` |
| in-image sglang | `0.5.19.dev20260917+ga9fb1c3238` |
| repo | `AMD-AGI/Infera`, branch `dev/pd_opt/glm_5.2_agentx`, HEAD `c19bb2a8fbf70195b80cb2505960ea87f9d17d81` |
| uncommitted | `bench/glm5p2_pd/engine.sh` — `patches/0001-engine-sh-extra-args-env.patch` |

**Material scope caveat — the two "added optimisations" were NOT in the T2
image.** The mission's fixed-parameters table names the run image as
`…-nextnfix-hicache` (with the GLM NextN shared-experts-fusion fix + upstream
PR #37152 HiCache widening). But `config.yihou.full.sh` forces
`IMAGE=infera-sglang:v0519-yihou-0917` (the plain base) — a user decision on
2026-09-18, after a third GPU memory access fault, to align the T2 `server_args`
byte-for-byte with the C40 run. Confirmed from the emitted launch command
(`results/t2f-launch-command-lines.txt`): the image is the base.

Consequences, recorded honestly:
- **Neither the NextN shared-experts-fusion fix NOR PR #37152 was present in
  T2.** So HiCache ran on the **stock** base image, not the widened
  PR #37152 path. T2 therefore does **not** measure PR #37152; it measures
  HiCache-on timing on the stock kernels. (PR #37152 is open upstream, never
  merged, its AMD ROCm CI red — its patch is preserved in T1's pack-up.)
- Under simulated acceptance the missing NextN fix is immaterial (acceptance is
  forced, see notes.md), but it means the draft was degraded — irrelevant to
  timing, relevant only if anyone reads the (forced) acceptance gauge.

## Model and data

| | |
|---|---|
| model | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (shared NFS, absolute path) |
| served name | `glm5.2-mxfp4` |
| KV dtype | `fp8_e4m3` |
| AgentX trace corpus | `semianalysisai/cc-traces-weka-062126` @ `23f152f6f0f9399a85901b89a6458def0ef16729`, 393 traces |
| InferenceX | fetched by `tools/ensure_inferencex.py` into `bench/glm5p2_pd/.cache/InferenceX` at the ref pinned in `config.sh` |

## Secrets

None for the deployment. AgentX pulls its trace corpus from Hugging Face; if the
dataset is gated for your account, `HF_TOKEN` must exist in the control node's
environment. **No secret value appears anywhere in this pack-up.**

## Run shape (T2 — full mode, simulated acceptance)

P4+DPA / D4+DPA, TP4 / DP4 / EP1 on both legs, `mem_fraction_static 0.85`,
`chunked_prefill 32768`, **max-running 128, CUDA-graph max BS 128**.

- **Prefill HiCache ON**, `hicache-ratio 1.5`, write_through, kernel io-backend,
  page_first (stock base-image HiCache — no PR #37152).
- Decode HiCache OFF (`DECODE_HICACHE=0` — no decode host pool).
- Decode speculative: EAGLE, 5 steps, topk 1, 6 draft tokens.
- **`--disable-custom-all-reduce`** on the decode leg (the "corrected MTP config"
  — the t2f arm). The t2e A/B arm is identical but WITHOUT this flag.
- **`SGLANG_SIMULATE_ACC_LEN=3.61`**, method `match-expected`, token mode
  `real-draft-token` — acceptance FORCED (see notes.md §"forced acceptance").
- `index_share_for_mtp_iteration=false` via `--json-model-override-args` (the
  GPU-fault mitigation; both arms).
- DSA backends: `--dsa-prefill-backend tilelang --dsa-decode-backend tilelang
  --dsa-topk-backend sgl-kernel`, `SGLANG_OPT_USE_TOPK_V2=false`.

Authoritative record of what each engine actually ran:
`results/server-info/{prefill,decode}-0.json` and
`results/t2f-launch-command-lines.txt`.
