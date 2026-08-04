# Environment

Per-node snapshots in [`env/env_chi2835.txt`](env/env_chi2835.txt) and
[`env/env_chi2879.txt`](env/env_chi2879.txt), captured **2026-08-04 09:15 UTC**, with
both legs still live — so the recorded state is the one that served the run.

## Digest

| | |
|---|---|
| cluster | **vultr** MI355X |
| access | jump host `root@149.28.124.225` (= slurm login node chi2866), then `ssh chi2835` / `ssh chi2879` |
| **prefill node** | **chi2835**, data-plane `10.2.122.78`, kernel `6.8.0-107-generic` |
| **decode node** | **chi2879**, data-plane `10.2.122.10`, kernel `6.8.0-124-generic` |
| GPUs | 8 × AMD Instinct MI355X `gfx950` (`0x75a3`), 288 GB/card, per node |
| CPU / RAM | AMD EPYC 9575F 64-core (256 threads) / 3.0 TiB, both nodes |
| GPU driver | **6.16.13**, both nodes |
| ROCm | **7.2.0** |
| image | **`infera/engine-sglang:merged-e`** |
| image ID chi2835 | `sha256:27667ee43291bed2bddb9caf44a63217fdb994d6f423f6ed3bf7e807340fae7a` |
| image ID chi2879 | `sha256:bfcb6462fa306743e0bf43b32ac0263ce9094e13591f6f748263e5348bf97e41` |
| base image | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` |
| sglang / torch | `0.5.15.post1` / `2.9.1+rocm7.2.0.git7e1940d4` |
| repo | `AMD-AGI/Infera`, branch **`yihou.dev.glm52.example`** @ **`e2d462a`** |
| kit under test | `examples/sglang_1p1d_glm5.2/` at that commit, **plus the 5 fixes** in `patches/kit_fixes.diff` |
| slurm holders | `yeandy-debug` on **both** nodes — **not ours, never `scancel`** |

> The two image IDs differ **by design** — each node built independently from the same
> branch head. They are the binding artifact, not the tag.

> **The image is NOT the one the kit names.** `cluster/*.sh` ships
> `inferaimage/infera-sglang:0.2.0`, which the kit's own README flags as a placeholder;
> it does not exist on this cluster. Substituting `merged-e` is a site choice made in the
> wrapper, exactly where the kit intends site choices to go.

## RDMA fabric — an asymmetry, recorded

Both nodes report **mode A viable ★ best** (`peermem: present`, `module:ib_peer_mem`),
so this deployment uses `cluster.peermem.sh` semantics. Full reports in
[`results/preflight_chi2835.txt`](results/preflight_chi2835.txt) and
[`results/preflight_chi2879.txt`](results/preflight_chi2879.txt).

| node | ACTIVE rails | preflight aggregate | note |
|---|---|---|---|
| chi2835 (prefill) | **8/8** `ionic_0..7` | 3,200 Gb/s | full |
| chi2879 (decode) | **7/8** — `ionic_5` marked `[DOWN]` | 2,800 Gb/s | physically down |

Both nodes resolved the routable RoCEv2 GID at **index 1**. Every NIC is 400 Gb/s
ionic, `peermem: yes`, `ODP: -` (none), `dmabuf: yes`, kernel `CONFIG_PCI_P2PDMA=y`.

**The kit exposes ONE global `RDMA_IB_DEVICES`, but the two nodes differ.** The wrapper
therefore sets their **intersection** — `ionic_0,1,2,3,4,6,7` — so every device named is
`PORT_ACTIVE` on both. Consequences, stated rather than glossed:

- The prefill leg runs **7 of its 8** usable rails. One rail of prefill-side bandwidth
  is left on the table purely to satisfy a single-valued config field.
- The alternative — listing all 8 — would name a down rail to the decode leg, and every
  transfer targeting it fails.

This is **not controlled for**, and its effect on KV transfer time is **unmeasured**.
The reference run (`par8`) had the same asymmetry and handled it by enumerating
`PORT_ACTIVE` **per node** inside its launcher; the kit cannot express that. Recorded as
a kit limitation in [`notes.md`](notes.md) §3, not fixed — fixing it means changing the
config schema, which is beyond "make this run".

## Host RDMA provider injection

The image's ENTRYPOINT is `/usr/local/bin/infera-inject-host-ionic`. It copies the host's
libionic over the container's so the in-container `libibverbs` speaks the host
`ionic_rdma` kmod's ABI, and it is **load-bearing on this fabric**:

| node | host provider build |
|---|---|
| chi2835 | `libionic.so.1.1.54.0-184` |
| chi2879 | `libionic.so.1.1.54.0-187` |
| container's own (both) | `libionic.so.1.0.54.0-149.g3304be71` |

Without the injection, `ibv_devinfo` inside the container prints
`Driver ionic does not support the kernel ABI of 1 (supports 4 to 4)` for every card and
`No IB devices found` — **measured first-hand** on the first bring-up attempt. The wrapper
therefore sets `HOST_RDMA_LIB` (the **symlink**, so each node resolves its own build),
`HOST_RDMA_MOUNT=/host-libionic/libionic.so` (the path this image's entrypoint reads) and
`ENTRYPOINT_KEEP=1`.

## Resolved deployment shape

Read off the live process command lines, not off what was requested:

| | prefill (chi2835) | decode (chi2879) |
|---|---|---|
| tp / ep | 8 / **8** | 8 / **8** |
| dp / DP-attention | **1 / off** | **8 / on** |
| `mem-fraction-static` | **0.70** | **0.85** |
| `chunked-prefill-size` (resolved) | 65536 ÷ 1 = **65536** | 65536 ÷ 8 = **8192** |
| context length | 262144 | 262144 |
| MTP | off | **EAGLE(3, topk 1, 4 draft)** |
| custom all-reduce | disabled | disabled |
| kvd | **on**, `--hicache-size 32` | off (by design) |
| kv events | on | on |

`--ep-size 8` on both legs confirms expert parallelism did **not** collapse when
DP-attention was turned off on prefill — the trap the kit's Note 1 warns about.

## External dependencies (absolute paths, not in any repo)

| what | where |
|---|---|
| model + tokenizer | `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` (816 GB, shared VAST NFS). `generation_config.json` temp 1.0 / top_p 0.95 is **load-bearing** — temp 0 + MTP is indistinguishable from KV corruption |
| host libionic | `/usr/lib/x86_64-linux-gnu/libionic.so.1` (symlink) → bind-mounted to `/host-libionic/libionic.so` |
| kit staging (must be the same path on both nodes) | `/mnt/vast/c_huggingface/glm52_example_verify/kit` |
| repo preflight module (the workaround) | `/mnt/vast/c_huggingface/glm52_example_verify/repo_preflight/infera/tools/preflight/mooncake_mode.py` |
| kvd L3 tier | `/tmp/kvd-long` **inside the container** — node root disk, **not** `/mnt/vast`. `--long-bytes 64G` |
| customer bench corpus | `caseA_conformance_corpus.tar.gz` from ROCm/MAD PR #173, 200 sessions, 13/13 axes verified |
| aiperf image | `aiperf-agentx:v1.0`, built from `github.com/SemiAnalysisAI/aiperf@cquil11/aiperf-agentx-v1.0` |

## Secrets required (names and sources only — no values here)

| secret | source |
|---|---|
| cluster SSH | key-based access to `root@149.28.124.225`, then node-to-node as root. Arrange your own; nothing in this kit contains a key. |
| docker registry | **not needed** — both images already present on both nodes. A cold node would need the team registry login. |
| etcd | **unauthenticated**, on the prefill node's private data-plane IP. |
| router / engine | no API key (`api_key=None`, `admin_api_key=None`). |

**No secret value appears anywhere in this kit** — scripts, env snapshots, the router log
and the engine logs were checked before packing.
