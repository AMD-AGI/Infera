# Environment

Snapshotted 2026-07-30, at the time of the runs in this packup.

## Hardware

| | chi2879 | chi2867 |
|---|---|---|
| Role (steps 1-3, 5) | **prefill** TP8 | **decode** TP8 |
| Role (step 4, routing test) | **prefill** TP8 | **2× decode TP4** (GPU0-3 :30000, GPU4-7 :32000) |
| Data-plane IP | 10.2.122.10 | 10.2.122.44 |
| GPU | 8× AMD Instinct MI355X (gfx950, `0x75a3`) | 8× same (`rocm-smi` reports "AMD Radeon Graphics" — same `0x75a3` device id) |
| CPU threads | 128 | 128 |
| RAM | 3023 GB | 3023 GB |
| amdgpu driver | 6.16.13 | 6.16.13 |
| Kernel | 6.8.0-124-generic | 6.8.0-107-generic |

The Qwen3-1.7B MVP rounds (see notes.md) ran **single-node on chi2879 only**:
prefill TP4 on GPU0-3, decode TP4 on GPU4-7.

### RDMA fabric

- Type: **ionic RoCE v2** (Pensando), 8 rails per node: `ionic_0` … `ionic_7`.
- All 8 `PORT_ACTIVE` on both nodes (checked before every run).
- ionic kernel module: `version 26.03.3.001`, `srcversion 2B6E52BDAE17240EA1DB9BE`.
- NIC firmware: `1.117.5-a-77`.
- Routable GID at **index 1** (RoCE v2), hence `MC_GID_INDEX=1`:
  - chi2879 `ionic_0` GID[1] `fd93:16d3:59b6:40d:690:81ff:fe36:7ef0`
  - chi2867 `ionic_0` GID[1] `fd93:16d3:59b6:41a:690:81ff:fe39:ef88`
- Data-plane RTT chi2879 → chi2867: 0.069 ms avg (`ping -c2`).
- Host `libionic.so.1` must be **injected into the container** — without it RDMA
  silently degrades. `scripts/glm52_up.sh` does this; verify with
  `ibv_devinfo | grep -c PORT_ACTIVE` → `8` **inside** the container.

## Software

| | value |
|---|---|
| Docker image | `infera/engine-sglang:pd-unified` |
| Image ID (sha256) | `f8ec2d627392435b7cf4c97e47b93a3b36588bec43864a1758b7c0dc9405bd18` |
| sglang | 0.5.15.post1 |
| torch | 2.9.1+rocm7.2.0.git7e1940d4 |
| ROCm | 7.2.0 (`/opt/rocm-7.2.0`) |
| infera repo branch | `yihou.dev.glm5.2.mxfp4.experiment` |
| infera repo commit | `362192e7aceabe20849d8c0431fc81df1ab23759` |
| infera in image | `/opt/infera/` (patched at runtime, see below) |
| etcd | `quay.io/coreos/etcd:v3.5.14` |

The image is a **local build, not on a registry** — it is the Infera PR #19
rebuild, which is what makes mooncake cross-node RDMA work. If a node lacks it:
`docker save infera/engine-sglang:pd-unified | ssh <dst> docker load`. Both
chi2879 and chi2867 already had it as of 2026-07-30.

**Runtime patch:** `scripts/glm52_up.sh` `docker cp`s `scripts/net_fixed.py` over
`/opt/infera/infera/common/net.py` in each container. That is patch 0001 — the
image predates the fix. See `patches/0001-note.md`.

sglang version sensitivity worth knowing: on 0.5.15.post1 infera logs
`SGLang version has no recognized prefetch_threshold field` — the field was
renamed upstream, so that particular override silently no-ops.

## Dependencies outside the repo

| What | Where | Notes |
|---|---|---|
| GLM-5.2-MXFP4 weights | `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` | shared VAST NFS (`10.2.123.177:/aac-8634674/...` → `/mnt/vast`). 408 GB in 282 safetensors. `GlmMoeDsaForCausalLM`, 78 layers, 256 experts, hidden 6144. |
| Qwen3-1.7B (MVP only) | `/mnt/vast/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | 4 GB. Used to make the wiring loop cheap. |
| Kit staging dir | `/mnt/vast/c_huggingface/glm52_kvexp` | must be on the shared FS, visible to **both** nodes. |
| Host `libionic.so.1` | `/usr/lib/x86_64-linux-gnu/libionic.so.1` on each host | injected into the container at prep time. |
| kvd L3 backing dir | `/tmp/kvd-long` (steps 1-4) → `/kvd-long` bind-mounted from host `/mnt/nvme-raid/kvd-long` (step 5) | The `/tmp` form is the container overlay, so kvd logs `ssd region long on overlay`. See below for why the bind-mount still isn't enough. |

### Storage layout (matters for the L3 question — step 5)

| Path | Backing | State |
|---|---|---|
| `/mnt/nvme-raid` | `/dev/md0`, ext4, 231 GB free | md0 = **raid1 of sda2+sdb2, i.e. SATA SSDs — not NVMe** |
| `/dev/nvme0n1` … `nvme7n1` | 8× 7 TB, ext4 | **all unmounted**; `nvme0n1` already holds another team's `kvd-long`/`kvd-short`, 120 GB — left untouched |
| `/mnt/vast` | NFS `10.2.123.177:/aac-8634674/...` | 501 TB shared, where the model and kit live |

Two traps established in step 5:

1. Bind-mounting the **directory** does not expose the **device**. A stock
   container has no `/dev/md0` node, so `lsblk` can't classify it. Accurate L3
   classification needs `--device=/dev/md0` (or `--privileged`) *as well as*
   the `-v` mount.
2. Even with the device visible, `buffered` is the **correct** verdict on this
   box — md0's members are SATA, and the classifier prefers buffered there for
   the cold-read readahead win. Measuring O_DIRECT needs a real NVMe mount.

## Required secrets

Names and sources only — no values here, and none in any packed script.

| What | How it's arranged |
|---|---|
| Cluster SSH | jump host `root@149.28.124.225`, then `ssh chi2879` / `ssh chi2867`. Key-based; no password in any script. |
| Docker registry | **not needed** — the image is a local build already present on both nodes. |
| etcd | no auth (plain HTTP on the data-plane IP, port 2379). |

## Gaps

- `nproc` / RAM / driver captured post-hoc on 2026-07-30, not pinned at each
  run's exact timestamp. They did not change between runs.
- The Qwen3 MVP container (`kvexp` on chi2879) was **removed** after the run, so
  its raw logs are gone. Excerpts survive in
  `results/kvaware_kvd_activation_evidence.txt`, captured in-session.
