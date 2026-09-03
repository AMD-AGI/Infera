# Environment

Raw snapshots from `scripts/collect_env.sh`, taken 2026-09-02 04:57 UTC, are in
`env_n0433.txt` and `env_n0133.txt`. This file is the distilled version.

## Two hosts, and why there are two

The bring-up ladder (rungs 0→4) ran on **n04-33**. Partway through, that node
filled up with other users' work — a foreign host `etcd` on port 2379, another
user's `torchtitan-job27029`, and the team's own big-model track on GPUs 4-7 —
so the **fixlen sweep was re-established from scratch on n01-33** and the same
result reproduced there. Every number in `results/fixlen_*.csv` is from
**n01-33**; the ladder evidence in `results/ladder.md` is from **n04-33**.

That the result crosses two independently-built images on two hosts is the
strongest single statement in this packup.

| | n04-33 (ladder) | n01-33 (fixlen numbers) |
|---|---|---|
| hostname | `smci355-ccs-aus-n04-33.prov.aus.ccs.cpe.ice.amd.com` | `smci355-ccs-aus-n01-33.prov.aus.ccs.cpe.ice.amd.com` |
| data-plane IP (`fenic`) | `10.235.192.139` | `10.235.192.136` |
| GPUs used | 0-3 of 8 | 0-3 of 8 |

## Hardware (identical on both)

- **GPU**: 8 × AMD Instinct MI355X, `gfx950`, card model `0x75a3`,
  304 GiB HBM each (`VRAM Total Memory (B) = 309220868096`).
- **GPU driver**: `6.14.14` (amdgpu / ROCm KFD).
- **CPU**: 2 × AMD EPYC 9575F 64-Core (256 threads total).
- **RAM**: 3.0 TiB.
- **OS**: Ubuntu 22.04.5 LTS, kernel `6.8.0-107-generic`.
- **RDMA fabric**: 8 × `ionic_0..7`, all `State: Active` / `LinkUp` /
  `Rate: 400` / `Link layer: Ethernet` (RoCE), netdevs `benic1p1..benic8p1`,
  `192.168.{1..8}.4/31`.
  **Not used by this experiment** — MIX is single-node, aggregated, no PD, no
  Mooncake, no RDMA. Recorded because the same host serves PD runs.
- **docker**: 29.4.0 on both.

## Software

### Images

| role | tag | digest |
|---|---|---|
| base (ours) | `lmsysorg/sglang:v0.5.18-rocm720-mi35x` | image id `sha256:2560b7d1e789692828287407c42a35023e31a32c31a3be4a8f04ebdc65003901`, repo digest `lmsysorg/sglang@sha256:6d68cd19206716cb3f1e31e2ad89cd0852d7ae614a792773c30a4277f8955c72` |
| base (vendor, rung 0 only) | `lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260822` | image id `sha256:79510d3ddfa9da17923e1af2089ba76d81acf865ae62b1bccd8426db0140a70a`, repo digest `lmsysorg/sglang-rocm@sha256:4735a1841eacb4be8041ff3d07c45e2923f8a05c87375fff427681322c355e00` |
| built engine | `infera/engine-sglang:glm53-c821c425` | **built locally per node, never pushed** — n04-33 `sha256:6ecbeecdc0c198ea1b0b85553ac4f7d498cd80faecd726d0e0d4e0656bbcd1c6` (2026-09-01 08:47 UTC), n01-33 `sha256:fde285569b4aeaffcaf00042c3174787d8da93ec4ab6d293c21e3f0023b737d0` (2026-09-02 04:08 UTC). Different ids, same Dockerfile and same pinned ref. |
| etcd | `quay.io/coreos/etcd:v3.5.14` | — |

### Inside the built engine image (read out of the live n01-33 container)

| | |
|---|---|
| sglang | `0.5.18`, editable from `/sgl-workspace/sglang/python/sglang` |
| **sglang git HEAD** | **`c821c425c31b0e6c8151324b60fbc2857c39eaef`** — *"Merge branch 'xinyuan/glm-5.3-flash-support' into feat/glm-5.3-flash-rocm"*, i.e. PR **#36607**'s frozen head |
| overlay content check | `glm5_next.py` **1942** lines, `quark.py` **1172** lines (the two numbers that distinguish `c821c425` from the wrong pin `9e692c92` — 1834 / 1103) |
| aiter | git `d9e5ef7ce08ee7045d583aed768cff41aa9210fe`, `/sgl-workspace/aiter` |
| torch | `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a` |
| infera | installed at `/opt/infera`, editable; no `__version__` attribute exported |

### The pin, and why it is not the obvious one

`SGLANG_GLM53_REF=c821c425…` is set in `repo-changes/deploy/docker/Dockerfile.sglang.glm53`
(the file's own header carries the long rationale). In short:

- Infera PR #143 pins `9e692c92` — that is #36607's **first** commit and is
  missing `77a46694` (AITER mHC on gfx95) and `654df43c` (mixed Quark MXFP4 +
  block-FP8 loading). Building there silently gets the pre-mHC slow path **and**
  cannot load MXFP4 at all.
- #36507's head today is also unusable for AMD: its 2026-08-31 rebase dropped
  the AMD work, and `c767511e` (2026-09-01) reverts #36607 wholesale.
- `c821c425` is frozen by the #36607 merge and is exactly what
  `git fetch origin pull/36607/head` yields.

### Infera repo

- branch `yihou.dev.glm53.expr`. The experiment ran on
  **`f48b79d04316907f29478f9f037d893bdf50cd4a`** (`SLA-based planner (#139)`)
  **plus uncommitted working-tree changes**.
- Those changes were committed at 2026-09-02 05:06 UTC, while this packup was
  being written. The image this experiment used is built from
  `deploy/docker/Dockerfile.sglang.glm53` as committed in
  **`ea989b3b39c30a25d659fb331a0d0dce2ab9c3e1`** (verified byte-identical to
  what ran). Branch head at packup time:
  `37f2a8fbf50cd68cbd383aa06f3af385794770f2`.
- Full commit-by-commit breakdown: `repo-changes/README.md`.

## External dependencies (absolute paths, not in the repo)

- **Weights**: `/apps/data/models/GLM-5.3-Flash-MXFP4` — **212 GB**, 120
  safetensors shards, `model_type: glm5_next`,
  `architectures: [Glm5NextForConditionalGeneration]`, `transformers_version
  5.16.0`, `quantization_config.quant_method: quark`, 2552 `exclude` entries,
  288 routed experts + 1 shared, `moe_intermediate_size 2048`.
  Upstream: `OneNexus/GLM-5.3-Flash-MXFP4` on HuggingFace.
- **`/apps/data/models` is its own NFS mount** (a symlink to
  `/perf_apps/data/models`) and does **not** propagate into a container when you
  bind its parent `/apps`. It needs its own `-v`. See `notes.md` — this one
  costs an hour if you meet it cold.
- **Workspace**: `/apps/yihou/glm53.series.workspace_20260901/` (the scratch this
  packup was distilled from; nothing here depends on it at run time).

## Secrets required (names and sources only — no values here)

- **docker.io pull access** for `lmsysorg/sglang*`. Public images; if docker.io
  is unreachable from the build host, the Dockerfile header documents the
  Harbor pull-through mirror as a `--build-arg`, which needs cluster Harbor
  credentials from the team's normal docker login.
- **SSH to the MI355X hosts** — arranged through the user's existing
  `~/.ssh/config`; no key material is in this packup.
- **No HF token needed** — the weights are already on the shared mount.
- No etcd / S3 / API credentials: etcd is unauthenticated and local to the node.

## Gaps recorded honestly

- `aiter` and `infera` expose no `__version__`; they are pinned here by git SHA
  and install path respectively.
- The built image is **local per node** and was never pushed to a registry, so
  there is no repo digest for it. Reproducing means rebuilding from the
  Dockerfile at the same `SGLANG_GLM53_REF`.
- The `collect_env.sh` GPU/VRAM snapshot on n04-33 shows all 8 GPUs busy — that
  is other users' and the team's big-model work at snapshot time, not this run.
