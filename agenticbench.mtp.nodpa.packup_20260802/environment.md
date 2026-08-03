# Environment — noDPA run (2026-08-02)

> **This is a COLD deployment on TWO NEW NODES**, and that is the single most
> important difference from the lat1 kit. The 2026-08-01 pair (jobs 24300/24301,
> `crsuse2-m2m-253`/`-236`) hit its 24 h walltime and was **released at 08:36 UTC
> on 2026-08-02** — 3 hours before this run. Nodes, containers, images, etcd, kvd
> and the GPU radix tree were all rebuilt from scratch.
>
> So unlike lat1 (which reused Case A's live legs 21 h in), this run cannot claim
> "same live deployment". It claims something weaker and sufficient: **same branch,
> same commit, same Dockerfile, same flags except the one under test** — verified
> flag-by-flag from `server_args=` in the boot logs, not from the launch command.

## Nodes (spur / crsuse2-m2m)

| role | job | host | ens3 (data plane) | image id |
|---|---|---|---|---|
| prefill + etcd + router + kvd | **28490** | `crsuse2-m2m-231` | `10.245.150.172` | `sha256:683cb6d8…` |
| decode + kvd(OFF) | **28485** | `crsuse2-m2m-276` | `10.245.152.249` | `sha256:57009158…` |

Per-node hardware (captured live by `scripts/collect_env.sh`; full output in
`env/env_crsuse2-m2m-{231,276}.txt`):

| | |
|---|---|
| GPU | 8 × AMD Instinct MI355X `gfx950` |
| **amdgpu driver** | **6.14.14** |
| CPU | AMD EPYC 9575F 64-Core |
| kernel | `6.8.0-107-generic` (Ubuntu SMP, 2026-03-13) |
| sglang / torch (in-image) | **0.5.15.post1** / **2.9.1+rocm7.2.0.git7e1940d4** |
| image tag (both nodes) | `infera/engine-sglang:merged-mtp` |
| base image (pinned) | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` @ `sha256:40e940a0c55b87105c773d8b484616616b3a91662bfa223c48ff721d9793dc8d` |
| image digest, node 231 | `sha256:683cb6d89dacf36caf7288d9aa39f681766fd75f3a2a624c388a39868974eb99` |
| image digest, node 276 | `sha256:57009158e52fbb8881c05b284981333f84219e156c948b8ee7828378fd07e846` |

**Differing image ids are expected, not a defect.** Each node built the same
Dockerfile independently, so Rust object files and layer timestamps differ.
Equivalence is established by the bytecode assertion gate in `start_ctr.sh`
(8 assertions, `BYTECODE_GATE OK` on both), never by digest comparison.

`gfx950` is **`xnack-`** — no page-migration fallback, so a host-pointer access
from GPU is a hard fault. This is why the ROCm hicache host-alloc patch is
mandatory (below).

## Fabric — spur, configured oppositely to vultr

    IBDEV=mlx5_0   MC_GID_INDEX=3   MOONCAKE_DISABLE_HIP_DMABUF=0  (dma-buf ON)
    MC_MS_AUTO_DISC=0   MC_MS_FILTERS=mlx5_0   NIC=ens3

Verified live on node 231: `link mlx5_0/1 state ACTIVE physical_state LINK_UP
netdev ens3`. `MC_FORCE_TCP` count = **0** on both legs, i.e. RDMA did not
silently fall back to TCP.

Each node exposes both `ionic_0..7` and `mlx5_0`. A leg script that auto-discovers
ionic — as the branch's own `glm52_leg.sh` does — binds the wrong fabric here.

## Deployment under test — and the ONE flag that differs from lat1

    two-node PD over mooncake RDMA        (mlx5_0, GID 3, dma-buf ON)
    DP-attention                          PREFILL **OFF**  <-- THE VARIABLE
                                          DECODE  ON (dp8)
    expert parallelism (--ep-size)        8, BOTH legs, BOTH arms
    kv-aware routing                      ON   (Rust router, prefill 20.0 / decode 2.0)
    kvd (infera HiCacheStorage)           PREFILL ON / DECODE OFF
    MTP (EAGLE)                           DECODE ON: steps 3, topk 1, draft 4
    --disable-custom-all-reduce           ON, both legs
    --context-length                      262144
    --chunked-prefill-size (GLOBAL/step)  **65,536 prefill, matched to lat1**
                                          (8,192 in the retained chunk-control arm)
    --mem-fraction-static                 prefill **0.70** / decode 0.85
    --hicache-size                        32          (absolute GB)
    --enable-cache-report                 ON
    KV pool                               2,821,248 tokens/rank prefill

Two values differ from lat1 and both are documented rather than glossed:

| | lat1 | this run | why |
|---|---|---|---|
| prefill `enable_dp_attention` | True | **False** | the variable under test |
| prefill `mem_fraction_static` | 0.80 | **0.70** | **forced** — 0.80 does not boot without DPA; see `notes/nodpa_design.md` |

### `chunked_prefill_size` is per-GLOBAL, and DPA divides it by `dp_size`

`server_args.py:4902` — a **division**, not a clamp:

```python
if self._resolved().enable_dp_attention:
    self.chunked_prefill_size = self.chunked_prefill_size // self.dp_size
    logger.warning("DP attention is enabled. The chunked prefill size is "
                   f"adjusted to {self.chunked_prefill_size} to avoid MoE kernel issues.")
```

So the `server_args=` value means different things on the two arms, and matching
it naively **mismatches the machine**:

| arm | requested | `server_args=` | semantics | **global tokens/step** |
|---|---:|---:|---|---:|
| lat1 prefill (dp8) | 65,536 | 8,192 | **per rank** | **65,536** |
| **noDPA-65K prefill** | **65,536** | **65,536** | global | **65,536** ✓ matched |
| noDPA-8K prefill (control) | 8,192 | 8,192 | global | 8,192 (⅛) |

Cross-checked against `#new-token` in the engine logs: the modal prefill batch is
8,192 on **both** DPA and noDPA-8K arms — but on the DPA arm that is 8,192 per
rank on 8 ranks concurrently, and on noDPA-8K it is the entire machine.

**Decode has no prefill chunk to speak of.** A PD decode leg does not run prefill,
so `chunked_prefill_size` there is inert on both arms and is not part of the
comparison. (It is still printed in its `server_args=`, which is what made an
earlier version of this file list it as if it were matched.)

The 8,192 arm was run first, before the `// dp_size` semantics were read from
source. It is retained as `results/chunk8192_ARM/` and turned out to be worth
keeping: it measures the chunk effect in isolation, and that effect is nil
(0.98–1.06×), which is what licenses attributing the rest to DPA.

## Repository state

| | |
|---|---|
| worktree | `/home/yihou/dev/git/infera.merge.liying.kv.mtp` |
| branch | `yihou.dev.glm52.merged.experiment` |
| commit | **`e56e9756977b9c6efd2a0a762af53faddc961d6b`** |

### Uncommitted, and load-bearing

Two working-tree changes are **required** and deliberately **not committed**
(operator's standing instruction, reaffirmed at this packup):

| path | what |
|---|---|
| `deploy/docker/Dockerfile.sglang` (modified) | applies the ROCm hicache host-alloc patch at build time |
| `deploy/docker/patches/sglang_rocm/` (new) | the patch itself — `hipHostRegister` for `ALLOC_MEMORY_FUNCS["cuda"]` |

Both are carried in `patches/`. A cold reproducer must apply
`patches/0001-dockerfile-rocm-hicache-hostalloc.patch` to a clean `e56e975`
checkout **before building**, or the prefill leg dies with `Memory access fault by
GPU node-N` under kvd + long prompts.

Verify it reached the running image (behavioural, not a file listing):

```bash
docker exec agbench_mtp python3 -c "
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS, alloc_with_pin_memory
assert ALLOC_MEMORY_FUNCS['cuda'] is alloc_with_pin_memory"
```

### Applied inside the container, not in the repo

`patches/apply_p1v3.py` — the `GLM52_P1V3` DSA indexer fix, applied to the
**decode** container only. Target md5 before patch:
`632f17acd38737459b43f830ee60ee89` (confirmed live on this run's fresh image).
Original preserved at `/tmp/dsa_indexer.py.orig` inside the container.

**Every number in this kit is `merged-mtp` + P1V3**, not stock.

## Load generator

| | |
|---|---|
| driver | `Optimus-AgenticBench @ 1cf01cb`, branch `fix/realistic-profile-session-driver` |
| repo | `/home/yihou/dev/git/Optimus-AgenticBench` (editable install) |
| venv | `/shared_nfs/yihou_agentbench/venv` |
| entry | `python3 -m agent.agent_throughput --mode realistic --dashboard-mode` |
| workload | `spec/nodpa_full.yaml` |
| driven from | the **login node** |

**`1cf01cb`, not `main`** — main under-loads silently.

## External absolute paths (outside this kit)

| what | where |
|---|---|
| model weights | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (408 GB) |
| scratch, logs, staged scripts | `/shared_nfs/yihou_agbench_mtp/` |
| bench outputs | `/shared_nfs/yihou_agbench_mtp/bench/nodpa_full/` |
| kvd L3 spill | `/tmp/kvd-long` **inside the container** |
| the comparison arm | `../agenticbench.mtp.lat1.packup_20260802/` |
| the loaded arm (N≈44) | `../agenticbench.mtp.caseA.packup_20260801/` |

## Secrets

None appear in this kit. What a reproduction needs — **names and sources only**:

| secret | why | where it comes from |
|---|---|---|
| docker registry login | pulling the pinned base image | `DOCKER_CONFIG=/tmp/dockercfg` on each node; created empty by `build_image.sh`, populated out of band |
| spur cluster access | `spur exec` into the held jobs | the operator's own cluster account; no key embedded |
| HF token | **not needed** — weights already on `/shared_nfs` | n/a |
| etcd / kvd auth | **none** — unauthenticated on the node-local socket / private data plane | n/a |

No API key was passed to the driver (`--api-key` unset). Verified: no credential
value appears in any script, log or result file in this kit, including inside the
gzipped ones.

## Timeline (all UTC, 2026-08-02)

| time | what | outcome |
|---|---|---|
| 08:36 | jobs 24300/24301 hit 24 h walltime | **released** — lat1's deployment gone |
| 11:32 | discovered the release; began cold rebuild | — |
| 11:33–11:47 | held 28490 (`-231`) and 28485 (`-276`) | 4 bad nodes excluded (`JobHoldMaxRequeue`) |
| 11:47–11:57 | image built on both nodes, in parallel | `BYTECODE_GATE OK` ×2 |
| 11:58 | legs booted; **prefill CHUNK=65536** | wrong — see notes |
| 12:05 | prefill rebooted at **CHUNK=8192** | matches lat1's effective value |
| 12:22 | probe attempt 1 | **died at startup** — seed > 2**32-1 |
| ~12:23 | prefill leg **crashed: HSA OOM at GMU 0.80** | activation memory, not KV |
| 12:29 | prefill rebooted at **GMU 0.70** | ready, 0 faults |
| 12:35–12:40 | probe (seed 2026080202) | 3 gates pass |
| 12:41:42–13:14:47 | noDPA full @ **global chunk 8,192** | PASS — 175/177, 0 faults → kept as the chunk control |
| ~13:40 | read `server_args.py:4902`: DPA **divides** chunk by `dp_size` | the 8,192 arm was at ⅛ lat1's global budget |
| 13:59 | prefill rebooted at **CHUNK=65536** (GMU 0.70 unchanged) | ready, 0 faults; no activation OOM at 8× the chunk |
| 14:11–14:16 | probe (seed 2026080212) | 3 gates pass |
| **14:18:43–14:51:49** | **noDPA full @ global chunk 65,536 — THE RESULT** | **PASS — 115/117, 0 engine faults** |

## Gaps — stated, not hidden

- **The GMU 0.80 OOM crash log is not recoverable.** The leg script writes with
  `>`, so the reboot truncated the file; only the GMU 0.70 run survives
  (`mem_fraction_static=0.7` is the only value present). The crash was observed
  first-hand and quoted verbatim in `notes/nodpa_design.md`, but the **artifact**
  is gone. Anyone re-running will reproduce it by booting this arm at 0.80.
- **No GMU-matched comparison exists**, and none is reachable — 0.80 does not boot
  without DPA. Argued immaterial from `token usage: 0.04`; not proven by ablation.
- **3 TTFT outliers (1.7 % of requests) have no established cause.** Engine faults,
  retractions and scheduler exceptions were all checked and are zero.
- **No per-request TPOT array** is persisted by the driver — only `summary.json`'s
  four points. Same limitation as lat1.
- **DPA under load is untested here.** N=1 removes exactly what DPA optimises.
