# Environment — lat1 run (2026-08-02)

> **This is the SAME LIVE DEPLOYMENT as the Case A run of 2026-08-01.** The two
> legs were brought up once at 2026-08-01 08:36 UTC and were never restarted;
> lat1 ran against them at 05:39–06:12 UTC on 2026-08-02, ~21 h into the same
> containers. Not one server flag, env var or patch differs. That is what makes
> the Case A ↔ lat1 comparison in `analysis/` a same-server comparison rather
> than a cross-deployment one — the only variable is the client-side workload.
>
> Everything below is therefore carried verbatim from
> `agenticbench.mtp.caseA.packup_20260801/environment.md`, and was re-verified
> live before the lat1 run: router `active_workers: 2`, both containers `Up 20
> hours`, decode still carrying `GLM52_P1V3` (source count = 3).

## Nodes (spur / crsuse2-m2m)

| role | job | host | ens3 (data plane) | image id |
|---|---|---|---|---|
| prefill + etcd + router + kvd | **24300** | `crsuse2-m2m-253` | `10.245.157.89` | `sha256:42a303e5…` |
| decode + kvd(OFF) | **24301** | `crsuse2-m2m-236` | `10.245.146.87` | `sha256:ff7b02eb…` |

Per-node hardware (captured by `scripts/collect_env.sh`; full output in
`env/env_crsuse2-m2m-{253,236}.txt`):

| | |
|---|---|
| GPU | 8 × AMD Instinct MI355X `gfx950`, 288 GB VRAM each (309,220,868,096 B) |
| **amdgpu driver** | **6.14.14** |
| CPU | 2 × AMD EPYC 9575F, 59 cores/socket, 2 threads/core (236 logical) |
| RAM | 2.7 TiB |
| kernel | `6.8.0-107-generic` (Ubuntu SMP, 2026-03-13) |
| ROCm (in-image) | 7.2.0 |
| sglang / torch | 0.5.15.post1 / 2.9.1+rocm7.2.0 |
| image (both nodes, same tag) | `infera/engine-sglang:merged-mtp` |
| image digest, node 253 | `sha256:42a303e5820cce9fa58ee10968dac8e4b87cb9e5eddc00f47ffa6bd524b7ec91` |
| image digest, node 236 | `sha256:ff7b02eb6f1c6c184c2fb9a46dd18a48a2c18f50bfc35e0e0f306d34baf899f9` |

**Each node exposes BOTH fabrics** — 8 × `ionic_0..7` *and* `mlx5_0`. This run
used **`mlx5_0` only** (`MC_MS_FILTERS=mlx5_0`, `MC_MS_AUTO_DISC=0`). A leg
script that auto-discovers ionic devices — as the branch's own `glm52_leg.sh`
does — will find 8 of them here and bind the wrong fabric. That is the single
most important difference from the vultr sibling, where ionic *is* the right
answer.

**Differing image ids are expected and are not a defect.** Each node built the
same Dockerfile independently, so Rust object files and layer timestamps differ.
Equivalence is established by the bytecode assertion gate in `start_ctr.sh`
(8 assertions, passed on both), never by digest comparison.

`gfx950` is **`xnack-`** — there is no page-migration fallback, so a host-pointer
access from GPU is a hard fault, not a slow path. This is why the ROCm hicache
host-alloc patch is mandatory (below).

## Fabric — spur, and it is configured oppositely to vultr

    IBDEV=mlx5_0   MC_GID_INDEX=3   MOONCAKE_DISABLE_HIP_DMABUF=0  (dma-buf ON)
    MC_MS_AUTO_DISC=0   MC_MS_FILTERS=mlx5_0   NIC=ens3

For contrast, the vultr sibling runs 8 × ionic, `MC_GID_INDEX=1`, dma-buf OFF,
peermem ON. A leg script written for one cluster will either fail to start or
silently fall back to TCP on the other. Verified here: `MC_FORCE_TCP` count = 0
on both legs.

## Deployment under test

    two-node PD over mooncake RDMA        (mlx5_0, GID 3, dma-buf ON)
    DP-attention 8/8                      both legs
    kv-aware routing                      ON   (Rust router, prefill 20.0 / decode 2.0)
    kvd (infera HiCacheStorage)           PREFILL ON / DECODE OFF
    MTP (EAGLE)                           DECODE ON: steps 3, topk 1, draft 4
    --disable-custom-all-reduce           ON, both legs
    --context-length                      262144
    --chunked-prefill-size                65536      (8192 per rank at dp8)
    --mem-fraction-static                 prefill 0.80 / decode 0.85   <-- see notes
    --hicache-size                        32          (absolute GB, never --hicache-ratio)
    --enable-cache-report                 ON          (or the cache column reads 0)
    KV pool                               2,939,264 tokens/rank prefill
                                          3,085,504 tokens/rank decode
    per-rank free after init              284.0 GB   both legs

## Repository state

| | |
|---|---|
| worktree | `/home/yihou/dev/git/infera.merge.liying.kv.mtp` |
| branch | `yihou.dev.glm52.merged.experiment` |
| commit | `e56e975` |

### Uncommitted, and load-bearing

Two working-tree changes are **required** and are deliberately **not committed**
(operator's instruction — decide after the experiment whether they enter the
branch):

| path | what |
|---|---|
| `deploy/docker/Dockerfile.sglang` (modified) | applies the ROCm hicache host-alloc patch at build time |
| `deploy/docker/patches/sglang_rocm/` (new) | the patch itself — `hipHostRegister` for `ALLOC_MEMORY_FUNCS["cuda"]` |

Without them, kvd + long prompts GPU-fault on gfx950 with
`Memory access fault by GPU node-N on address <host VA>`. The branch does not
carry this fix; it was never needed on vultr because that validation ran
`--context-length 32768` with small prompts.

Verify it reached the running image (behavioural, not a file listing):

```bash
docker exec agbench_mtp python3 -c "
from sglang.srt.mem_cache.memory_pool_host import ALLOC_MEMORY_FUNCS, alloc_with_pin_memory
assert ALLOC_MEMORY_FUNCS['cuda'] is alloc_with_pin_memory"
```

### Applied inside the container, not in the repo

`patches/apply_p1v3.py` — the `GLM52_P1V3` DSA indexer fix, applied to the
**decode** container only. Target md5 before patch:
`632f17acd38737459b43f830ee60ee89`. Original preserved at
`/tmp/dsa_indexer.py.orig` inside the container for a revert-based A/B.

**Every number in this kit is `merged-mtp` + P1V3**, not stock.

## Load generator

| | |
|---|---|
| driver | `Optimus-AgenticBench @ 1cf01cb`, branch `fix/realistic-profile-session-driver` |
| repo | `/home/yihou/dev/git/Optimus-AgenticBench` (editable install) |
| venv | `/shared_nfs/yihou_agentbench/venv` |
| entry | `python3 -m agent.agent_throughput --mode realistic --dashboard-mode` |
| workload | `spec/caseA_full.yaml` |
| driven from | the **login node** — the bench is HTTP + a tokenizer; running it inside the prefill container would place the load generator on the host it measures |

**`1cf01cb`, not `main`** — main under-loads silently.

## External absolute paths (outside this kit)

| what | where |
|---|---|
| model weights | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (408 GB) |
| scratch, logs, staged scripts | `/shared_nfs/yihou_agbench_mtp/` |
| bench outputs | `/shared_nfs/yihou_agbench_mtp/bench/caseA_armB/` |
| kvd L3 spill | `/tmp/kvd-long` **inside the container** — on spur this is an overlay on `/mnt/m2m_nobackup` (23 TB free), so the 512 G budget is safe here |
| the MTP-off reference run | `../../infera.glm5.2.experiment/agenticbench.glm52.spur.packup_20260731/` |
| the vultr sibling (ground truth for both fixes) | `../caseA.glm52.fullfeature.packup_20260801/` |

## Secrets

None appear in this kit. What a reproduction needs:

- **container registry** — `export DOCKER_CONFIG=/tmp/dockercfg` before *every*
  docker call (docker 29 buildx plugin discovery).
- **spur job allocation** — `spur exec <job>`; direct `ssh` to compute nodes is
  blocked by an `AllowUsers` whitelist and returns a misleading
  `Permission denied (publickey)`.
- **etcd** is unauthenticated on a private data-plane IP.

## Timeline

| UTC | what | outcome |
|---|---|---|
| 12:12 | Case A attempt 1, GMU 0.88 | froze at t=556 s; prefill DP0 HSA OOM |
| 13:32 | Case A attempt 2, GMU 0.88 + custom-AR fix | **0 completions** — prefill OOM again during the correctness probe |
| 15:20 | both legs rebooted, **GMU 0.80** + P1V3 on decode | ready in 317 s |
| 15:31 | correctness: short 4/4, needle 3/5 then 4/5 | pass (see README) |
| **15:37:08–16:43:56** | **Case A full, ramp 400 + sustain 3600** | **PASS — 2,811/2,881, 0 engine faults** |

## Gaps

- **No MTP-off arm on this image.** The clean single-variable ablation was not
  run. It is the highest-value follow-up and costs one 67-minute window.
- **No TPOT percentile ladder.** The driver persists no per-request TPOT array;
  only `summary.json`'s four points exist. Not recoverable from these artifacts.
- **kvd is proven correct, not exercised.** 376,791 sets against 0 gets during
  the run — every request nested in a prefix the in-GPU radix cache already had.
- **Turn-count and inter-turn-delay p99 are window-censored** at 3,600 s sustain.

---

## lat1-specific facts (appended 2026-08-02)

| | |
|---|---|
| run directory | `lat1_full/2026-08-02-05-39-24` |
| wall clock | 2026-08-02 05:39:24 – 06:12:25 UTC (1,980.9 s) |
| driver repo | `/home/yihou/dev/git/Optimus-AgenticBench` @ **`1cf01cb`** |
| driver python | `/shared_nfs/yihou_agentbench/venv/bin/python3` (numpy, aiohttp, transformers) |
| tokenizer / weights | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (shared NFS, absolute) |
| router endpoint | `http://10.245.157.89:8190` (kv-aware Rust router on the prefill node) |
| scratch / artifacts | `/shared_nfs/yihou_agbench_mtp/` (>500 MB artifacts never go to `/home`) |
| repo branch / commit | `yihou.dev.glm52.merged.experiment` @ **`e56e975`** |

### Uncommitted working-tree state the image was built from

Two changes are **deliberately uncommitted** (operator instruction: keep, decide
after the experiments). Both are already in the running image and are carried in
`patches/`:

* `deploy/docker/Dockerfile.sglang` — modified to apply the ROCm hicache
  host-alloc patch at build time.
* `deploy/docker/patches/sglang_rocm/patch_hicache_rocm_host_alloc.py` — new,
  untracked.

A cold reproducer must apply `patches/0001-dockerfile-rocm-hicache-hostalloc.patch`
to a clean `e56e975` checkout before building, or the prefill leg will die with
`Memory access fault by GPU node-N` under kvd + long prompts (gfx950 is `xnack-`,
so there is no page-migration fallback).

The decode leg additionally carries **`GLM52_P1V3`**, applied *inside the running
container* and **not** in the image — see `patches/README.md`. Re-verified before
this run: source count = 3.

### Secrets required (names and sources only — no values here or anywhere in this kit)

| secret | why | where it comes from |
|---|---|---|
| docker registry login | pulling/pushing `infera/engine-sglang` | `DOCKER_CONFIG=/tmp/dockercfg` on each node; set up out of band |
| spur cluster access | `spur exec` into the held jobs | the operator's own cluster account; no key is embedded |
| HF token | **not needed** — weights are already on `/shared_nfs` | n/a |
| etcd / kvd auth | **none** — both are unauthenticated on the node-local socket / private data plane | n/a |

No API key was passed to the driver (`--api-key` unset). Verified: no credential
value appears in any script, log or result file in this kit.
