# Spur ground truth (crsuse2-m2m) — what this skill is grounded in

The recommendation logic is not theoretical; it encodes validated runs on the AMD
spur cluster. Full deliveries live in the repo under
`crsuse/multinode_tp8_mlx5_dmabuf_dsv4_dockerfile_20260727/` (read its `GOTCHAS.md`
and `ENVIRONMENT.md` for the primary evidence).

## Hardware

| Item | Value |
|------|-------|
| Cluster | AMD `crsuse2-m2m` (Spur scheduler, single partition `amd-spur`) |
| GPU | 8× MI355X (gfx950, CDNA4), ~288 GiB VRAM/card |
| Node kernel | 6.8.0-x (Ubuntu 24.04); amdgpu 6.14.x; host ROCm 7.0.1 |
| Image ROCm | 7.2.0 (base `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`) |

## RDMA fabric — the crux

Nine RDMA devices: **8× AMD Pensando ionic** + **1× Mellanox mlx5**.

| | mlx5_0 | ionic_0..7 |
|--|--------|------------|
| vendor / driver | Mellanox ConnectX VF / `mlx5_core` | AMD Pensando DSC VF / `ionic` |
| netdev | **ens3** (has IP) | enP* (no IP) |
| link | 200 Gb/s | 400 Gb/s |
| **ODP** | ✅ `ODP_SUPPORT` + `ODP_SUPPORT_IMPLICIT` | ❌ none |
| routable GID | idx **3** (IPv4-mapped RoCEv2) | idx **1** (global IPv6 RoCEv2) |
| fork-safe | not required | needs `RDMAV_FORK_SAFE=1` |

**No peer-mem module is loaded on spur.** So Mode A is unavailable there; the node
lands in Mode B, and — because the only ODP NIC (200G mlx5) is slower and far fewer
than the eight 400G ionic rails — the **perf-regression** flag fires. The user
explicitly accepted "bandwidth regression OK, KV doubling NOT OK", which is exactly
why B-on-mlx5 is the spur choice.

## dma-buf behavior by NIC (the core insight)

| | mlx5 (ODP) | ionic (no ODP) |
|--|-----------|----------------|
| `ibv_reg_dmabuf_mr` | dynamic attach → **no pin, no double** | **pins the whole region** → double / KFD exhaust |
| 2-node TP8 PD | ✅ works (KV 237/288, not doubled) | ❌ SIGSEGV at KV-register step (independently reproduced) |
| single-node TP4 PD | ✅ stable | ⚠️ dies after the 1st request (session not alive) |

## Validated-run table

| Delivery | Nodes | TP | NIC | Registration | Result |
|----------|-------|----|-----|--------------|--------|
| `…_dmabuf_dsv4_dockerfile_20260727/` | 2 | 8 | mlx5 | `ibv_reg_dmabuf_mr` (Mode B) | ✅ cross-node PD, Dockerfile-reproducible build + single-node TP4 3-transport comparison |
| `…_dmabuf_dsv4_manualcommit_20260724/` | 2 | 8 | mlx5 | `ibv_reg_dmabuf_mr` (Mode B) | ✅ cross-node PD (manual `docker commit` image) |
| `…singlenode_tp4_noNIC_loopbacktcp_dsv4_20260723/` | 1 | 4+4 | none | mooncake TCP loopback (no RDMA) | ✅ intra-node 1P1D; ⏸ cross-node blocked on mooncake RDMA ENOMEM |

## How to run the probe on spur

Per the `spur-interactive-debug` skill: hold one 8-GPU node, then exec into a
running container that has the (dma-buf) image. Inside it:

```bash
python -m infera.tools.preflight.mooncake_mode          # or copy mooncake_mode.py in
```

`ibv_devinfo`, `rocm-smi`, `nm`, and the mooncake `.so` are all present in that
container, so ODP, GPU topology, and the `USE_HIP_DMABUF` check are all real — none
of which resolve on a bare login host.

## Unified image note

The `bab37a8` "unified Mooncake image" compiles **both** paths into one
`engine.so`; the mode is chosen entirely at runtime by env
(`MOONCAKE_DISABLE_HIP_DMABUF` 0/1) — so a single image serves Modes A, B, and C
and this skill just picks the env. The older `Dockerfile.sglang.dmabuf` is the
dma-buf-only variant. Either way, the runtime knobs this skill emits are the same.
