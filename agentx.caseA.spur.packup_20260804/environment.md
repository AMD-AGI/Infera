# Environment

Snapshots in `env/env_prefill.txt` and `env/env_decode.txt`, captured
**2026-08-04 ~04:55 UTC**, minutes after the last run, with both legs still
live — so the recorded command lines are the ones that served the runs.

## Digest

| | |
|---|---|
| cluster | **CruSoe / spur** (`crsuse2-m2m`), scheduler **Spur 0.5.1**, partition `amd-spur` |
| access | the login node **is** `crs-m2m-cpu-spur-012`; `/shared_nfs` and the spur CLI are local. `ssh` to compute nodes is blocked (`AllowUsers` whitelist) — use `spur exec <job>` |
| **prefill node** | **crsuse2-m2m-268**, job **35748**, ens3 `10.245.145.242/20` |
| **decode node** | **crsuse2-m2m-288**, job **35749**, ens3 `10.245.152.60/20` |
| GPUs | 8 × AMD Instinct MI355X `gfx950`, 288 GB/card, per node |
| CPU / RAM | 2 × AMD EPYC 9575F (236 logical) / ~2.8 TB, both nodes |
| kernel | `6.8.0-107-generic`, both nodes |
| **amdgpu driver** | **6.14.14**, both nodes |
| ROCm / sglang (in-image) | 7.2.0 / 0.5.15.post1 |
| image tag | **`infera/engine-sglang:final-pr`** |
| image id, 268 (prefill) | `sha256:3b71c8303ca710f8c80aeb2f2b09393834dcd1f0db575d76136b32f2c228a6ff` |
| image id, 288 (decode) | `sha256:81f05a634358988a2a036cb772af87c2b6c713a16a6140a11cdddee804b13560` |
| base image | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` @ `sha256:40e940a0c55b87105c773d8b484616616b3a91662bfa223c48ff721d9793dc8d` |
| repo | `AMD-AGI/Infera`, branch **`yihou.dev.glm52.merged.experiment.1`** |
| aiperf image | `aiperf-agentx:v1.0`, built here from `Dockerfile.aiperf` (see below) |

> **The two image ids differ by design.** Each node built the same Dockerfile
> independently from the same `src.tar`, so layer timestamps and Rust object
> files differ. Equivalence is established by the **9-assertion bytecode gate**
> in `scripts/start_ctr.sh` (`BYTECODE_GATE OK` on both), never by digest
> comparison.

## Why the image was rebuilt (and the constraint it violated)

The operator's instruction was **"avoid rebuilding the image"** — reuse the
`infera/engine-sglang:merged-mtp` already on the nodes. **This could not be
honoured.** The image is node-local (built in place; no registry, no exported
tar). Across three allocation rounds, ~25 nodes were screened:

- nodes **with** the image: 2 — both with GPUs fully occupied by other tenants
  (~278 GB/card in use)
- nodes with **free GPUs**: many — none carrying the image

Transferring 108 GB between nodes is slower than rebuilding. The rebuild took
**~4 min per node** from `build/src.tar` + layer cache, run in parallel.

It turned out strictly better: the rebuild produces `final-pr`, which has
**GLM52_P1V3 baked in**, so no in-container patching was required.
`merged-mtp` fails the gate at `_p1v2_rows pyc_hits=0` and needs
`scripts/apply_p1v3.py` + a `.pyc` purge + a relaunch.

## Fabric — spur is configured OPPOSITELY to vultr

    IBDEV=mlx5_0   MC_GID_INDEX=3   MOONCAKE_DISABLE_HIP_DMABUF=0  (dma-buf ON)
    MC_MS_AUTO_DISC=0   MC_MS_FILTERS=mlx5_0   NIC=ens3

Both nodes expose **9** RDMA devices: `ionic_0..7` **and** `mlx5_0`. Only
`mlx5_0` is correct here — the ionic set is the vultr fabric. A leg script that
auto-discovers ionic (as the vultr `glm52_leg.sh` does) will bind the wrong
fabric and silently fall back to TCP, which looks fine and is slow.

Verified on the wire, not assumed: `MC_FORCE_TCP` / `GID is NULL` occurrence
count = **0** on both legs.

`ibv_devinfo` inside the container reports `mlx5_0 PORT_ACTIVE (4)`.

> The container gate line `RDMA PORT_ACTIVE in container: 1` is the **correct**
> reading here. An earlier node pair read `9` because it counted the ionic
> devices too.

## Deployment under test

    two-node PD over mooncake RDMA          (mlx5_0, GID 3, dma-buf ON)
    prefill DP-attention                    OFF  (pure TP8)   <-- the variable
    decode  DP-attention                    ON   (dp8)
    kv-aware routing                        ON   (Rust router, prefill 20.0 / decode 2.0)
    kvd (infera HiCacheStorage)             PREFILL ON / DECODE OFF (by design)
    MTP (EAGLE)                             DECODE ON: steps 3, topk 1, draft 4
    --disable-custom-all-reduce             ON, both legs
    --context-length                        262144
    --kv-cache-dtype                        fp8_e4m3
    --hicache-size                          32          (absolute GB, never --hicache-ratio)
    --enable-cache-report                   ON          (or the cache column reads 0)
    --ep-size                               8, both legs

| | prefill | decode |
|---|---|---|
| `--chunked-prefill-size` (passed) | **65536** | 65536 |
| resolved by engine | **65536** (not divided — DPA off) | 8192 (65536 ÷ 8) |
| `--mem-fraction-static` | **0.70** | 0.85 |
| `--dp-size` | *(absent)* | 8 |
| `--max-running-requests` | 2048 | 2048 → 256/rank |

**The chunk value is not free.** `server_args.py:4902` divides
`chunked_prefill_size` by `dp_size` **only** under `enable_dp_attention`. With
DPA off, the same CLI number yields an 8× larger per-forward batch than a dp8
leg would see. GMU 0.70 (vs the spur Case-A kit's 0.80) is the paired
compensation for that activation peak. The vultr par8 kit takes the other route
— chunk 16384 with GMU 0.80.

## The two arms — what differs, and what is NOT controlled

Both arms share the same nodes, image, decode leg, corpus, and customer script.
Arm 2 ran 39 min after arm 1 on the same live decode leg.

| | arm 1 (C=2/8/16) | arm 2 (C=8) |
|---|---|---|
| window (UTC) | 04:00–04:53 | 05:32:39–05:49:12 |
| prefill DP-attention | **OFF** (pure TP8) | **ON**, dp8 |
| prefill `--mem-fraction-static` | 0.70 | 0.70 |
| `--chunked-prefill-size` passed | 65536 | 65536 |
| ...**resolved by the engine** | **65536** | **8192** (÷ dp_size) |
| routable prefill targets | 1 | 8 |
| `--kv-prefill-overlap-weight` | 20.0 | **5.0** |
| `--kv-decode-overlap-weight` | 2.0 | **1.0** |
| prefill leg log | `logs/prefill.log.gz` | `logs/prefill_dpa8.log.gz` |
| env script | `scripts/env_prefill.sh` | `scripts/env_prefill_dpa8.sh` |
| router script | `scripts/router.sh` | `scripts/router_tuned.sh` |

**Three things move between the arms, not one.** DPA, the resolved chunk size
(coupled to DPA by `server_args.py:4902`), and the router weights. Arm 2 is
therefore **not a DPA ablation** — see `analysis/dpa8_arm.md`.

GMU is 0.70 in both, but by different routes: arm 1 chose it to pair with an
undivided 65536 chunk; arm 2 was forced down to it after 0.80 crashed with
DP-attention activation OOM (`notes.md` Trap 6). It happens to match; it was not
held fixed by design.

## Load generator

| | |
|---|---|
| benchmark | **the customer's**, ROCm/MAD PR #173 `scripts/AgentX_CaseA/` |
| `replay_caseA.sh` | **unmodified**, md5 `7cde1afc627c7e4868eac0fd13741baa` (= the PR blob) |
| corpus | `caseA_conformance_corpus.tar.gz`, 200 sessions / 1,778 requests, seed 42 |
| conformance | `verify_caseA.py` → **13/13 axes PASS** (re-run here, output in `results/`) |
| replay engine | aiperf `0.8.0` from `SemiAnalysisAI/aiperf@cquil11/aiperf-agentx-v1.0` |
| scenario | `inferencex-agentx-mvp`, `--custom-dataset-type weka_trace` |
| driven from | the **prefill node** (that is where the aiperf image was built) |
| target | `http://10.245.145.242:8190` — the **router**, never a leg's own port |

## External absolute paths (outside this kit)

| what | where |
|---|---|
| model weights + tokenizer | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (408 GB). `generation_config.json` temp 1.0 / top_p 0.95 |
| scratch, logs, staged scripts | `/shared_nfs/yihou_agentx_caseA/` |
| image build context | `/shared_nfs/yihou_agentx_caseA/build/src.tar` (7.6 MB) |
| kvd L3 spill | `/tmp/kvd-long` **inside the container** — an overlay on `/mnt/m2m_nobackup` on spur, so the 512 G budget is safe |
| the read-only reference for the spur DPA=0 env | `/shared_nfs/yihou_final_pr/env_armB_prefill.sh` (another experiment's workspace — **read-only, never written**) |

## Secrets required (names and sources only — no values here)

| secret | source |
|---|---|
| spur job allocation | `sbatch -p amd-spur -q amd-burst-qos`; then `spur exec <job>`. Direct `ssh` to compute nodes is blocked and the error (`Permission denied (publickey)`) is misleading. |
| container registry | `export DOCKER_CONFIG=/tmp/dockercfg` before **every** docker call (docker 29 buildx plugin discovery). No login needed — base image pulled anonymously. |
| etcd | **unauthenticated** on the prefill node's private data-plane IP. |
| router / engine | no API key (`api_key=None`, `admin_api_key=None`). |
| HuggingFace | **not needed** — weights are staged on `/shared_nfs`; the aiperf build pulls only public PyPI + GitHub. |

**No secret value appears anywhere in this kit** — scripts, env snapshots and
the driver log were checked before packing.

## Node ledger — what was NOT ours

Jobs `33488 / 33490 / 33491 / 33505` and later `35682 / 35683`, all named
`par8ab`, belong to a **different, parallel experiment**. They were never
touched. Every screening allocation made during node selection was released
after use.
