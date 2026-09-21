# RDMA / GPU topology map — 137 (prefill) + 136 (decode)

Re-measured first-hand 2026-09-18 from sysfs / rocm-smi on both nodes, after the
previous phase's deployment was torn down. **Both nodes are topologically
identical** (same PCI layout, same GPU↔ionic pairing, same rail-id per index).

Rail id = 2nd hextet of the port-1 GID index-1 (`/sys/class/infiniband/<dev>/ports/1/gids/1`).
NUMA / PCI domain from `/sys/bus/pci/devices/<bdf>/numa_node`.

## Per-GPU map (identical structure on 137 and 136)

| GPU | ionic | GPU BDF | ionic BDF | PCI dom | NUMA | rail id | netdev | IB state |
|-----|-------|---------|-----------|---------|------|---------|--------|----------|
| 0 | ionic_0 | 0002:00:01.0 | 0002:00:09.0 | 0002 | 0 | 08 | enP2p0s9  | ACTIVE |
| 1 | ionic_1 | 0002:00:02.0 | 0002:00:0a.0 | 0002 | 0 | 07 | enP2p0s10 | ACTIVE |
| 2 | ionic_2 | 0002:00:03.0 | 0002:00:0b.0 | 0002 | 0 | 05 | enP2p0s11 | ACTIVE |
| 3 | ionic_3 | 0002:00:04.0 | 0002:00:0c.0 | 0002 | 0 | 06 | enP2p0s12 | ACTIVE |
| 4 | ionic_4 | 0003:00:01.0 | 0003:00:09.0 | 0003 | 1 | 04 | enP3p0s9  | ACTIVE |
| 5 | ionic_5 | 0003:00:02.0 | 0003:00:0a.0 | 0003 | 1 | 03 | enP3p0s10 | ACTIVE |
| 6 | ionic_6 | 0003:00:03.0 | 0003:00:0b.0 | 0003 | 1 | 01 | enP3p0s11 | ACTIVE |
| 7 | ionic_7 | 0003:00:04.0 | 0003:00:0c.0 | 0003 | 1 | 02 | enP3p0s12 | ACTIVE |

Findings:
- **GPU_n ↔ ionic_n is NUMA-local for every n** (both sit in the same PCI domain /
  NUMA node). This re-confirms the `rdma.survey.8node.packup_20260917` pairing by
  first-hand re-measurement.
- **NUMA0 = GPUs 0-3 + ionic_0-3 (dom 0002); NUMA1 = GPUs 4-7 + ionic_4-7 (dom 0003).**
- **No defective rail on either node.** All 8 ionic devices on 137 AND 136 have a
  live netdev, IB state ACTIVE, and a non-zero GID. (Contrast: 135's ionic_7 had no
  netdev and an all-zero GID.)
- **Rail id per index is identical across nodes** — 137's ionic_n and 136's ionic_n
  carry the same rail id. So `GPU_n ↔ ionic_n` on BOTH prefill and decode gives an
  automatic same-rail KV path (prefill rank i and decode rank i land on the same
  rail), which is the property Mooncake `MC_ENABLE_DEST_DEVICE_AFFINITY` needs.

Full GID index-1 values (for the record):

| ionic | 137 GID[1] | 136 GID[1] |
|-------|------------|------------|
| ionic_0 | fc01:0800:880d:2d5f:0690:81ff:fe44:4d69 | fc01:0800:a00d:2d6f:0690:81ff:fe36:7441 |
| ionic_1 | fc01:0700:870d:2d5f:0690:81ff:fe44:9f41 | fc01:0700:9f0d:2d6f:0690:81ff:fe37:3489 |
| ionic_2 | fc01:0500:850d:2d5f:0690:81ff:fe43:c489 | fc01:0500:9d0d:2d6f:0690:81ff:fe37:3c81 |
| ionic_3 | fc01:0600:860d:2d5f:0690:81ff:fe44:d781 | fc01:0600:9e0d:2d6f:0690:81ff:fe37:4191 |
| ionic_4 | fc01:0400:840d:2d5f:0690:81ff:fe42:a771 | fc01:0400:9c0d:2d6f:0690:81ff:fe39:4711 |
| ionic_5 | fc01:0300:830d:2d5f:0690:81ff:fe44:3719 | fc01:0300:9b0d:2d6f:0690:81ff:fe39:5b39 |
| ionic_6 | fc01:0100:810d:2d5f:0690:81ff:fe42:8731 | fc01:0100:990d:2d6f:0690:81ff:fe39:33c1 |
| ionic_7 | fc01:0200:820d:2d5f:0690:81ff:fe44:37c1 | fc01:0200:9a0d:2d6f:0690:81ff:fe39:3ab1 |

## Ready-to-paste engine settings

The JSON keys are the **LOCAL visible device index** (0..N-1). When
`*_GPU_DEVICES` lists physical GPUs in ascending order, visible index k maps to the
k-th listed physical GPU, so the ionic name must match that physical GPU's rail.

### P8D8 — all 8 GPUs (both nodes)

```
PREFILL_GPU_DEVICES=0,1,2,3,4,5,6,7
DECODE_GPU_DEVICES=0,1,2,3,4,5,6,7
RDMA_DEVICE={"0":"ionic_0","1":"ionic_1","2":"ionic_2","3":"ionic_3","4":"ionic_4","5":"ionic_5","6":"ionic_6","7":"ionic_7"}
MC_TE_FILTERS=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
```

### P4D4 — recommended GPUs 0,1,2,3 (both nodes)

```
PREFILL_GPU_DEVICES=0,1,2,3
DECODE_GPU_DEVICES=0,1,2,3
RDMA_DEVICE={"0":"ionic_0","1":"ionic_1","2":"ionic_2","3":"ionic_3"}
MC_TE_FILTERS=ionic_0,ionic_1,ionic_2,ionic_3
```

**Why GPUs 0,1,2,3 for P4D4:**
- **Single NUMA node / single PCI domain (0002).** All four GPUs and their four NICs
  (ionic_0-3) sit on NUMA0, so there is zero cross-socket PCIe traffic on the KV
  path and the four rails (08/07/05/06) form one clean, contiguous, healthy set.
- **All four rails healthy on both nodes** — no dead rail to route around here (the
  135 `ionic_7` problem does not exist on 137/136).
- **No external tenant on these GPUs.** Unlike 135 (root k8s pod pinned to GPU1),
  137 and 136 have all 8 GPUs free, so there is no forced-index constraint. The
  previous `samerail` packup used GPUs 2,3,4,5 only because 135 forced avoidance of
  GPU1 and GPU7 — that quad straddles the NUMA0/NUMA1 boundary and is a compromise
  we do not need to inherit here.
- GPUs 4,5,6,7 (NUMA1, ionic_4-7, rails 04/03/01/02) are an equally valid mirror
  quad if NUMA0 is ever contended; both are single-NUMA and fully healthy.
