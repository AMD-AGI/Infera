# Hardware prep — crsuse2-m2m-276 (same-node 1P1D P4D4)

Role: `node-prep`. Freed the GPUs and produced a first-hand hardware record for a
**same-node** P4D4 bring-up on a single node (`crsuse2-m2m-276`, 8x MI355X).
All measured first-hand 2026-09-22 via `spur exec 165913` (job 165913 holds the
node; runs as user `yihou`). `ssh` to the node is refused by design.

Plan under test: **prefill = GPUs 0,1,2,3 with ionic_0-3; decode = GPUs 4,5,6,7
with ionic_4-7**, so the two legs never share a NIC. **Verdict: the plan is
valid and NUMA-clean — see §4.**

## 1. Teardown — what was stopped

The 8 GPUs were entirely held by another user's container. Verified ownership
first-hand before acting.

| container | image | uptime / created | GPU memory held | KFD PIDs | action |
|---|---|---|---|---|---|
| `hy4-nomtp` | `vllm-rocm-hy4-stacked:test` | Up 10h / 2026-09-21 20:22:28 UTC | ~285 GB on **all 8** cards | 494054-494061 (`VLLM::Worker_TP`) | **`docker stop`** |
| `yzhou_model` | `rocm/atom-dev:latest` | Up 3h | **none** | only PID 853429 (`sleep`) | **left running** |

`docker top hy4-nomtp` showed the vLLM engine (492929 `vllm`, 493689 `python`,
493690 `EngineCor`) plus 8 `VLLM::Worker_TP` workers = exactly the KFD PIDs
holding VRAM. `yzhou_model` runs only `sleep` and holds no GPU memory, so it was
left untouched. No other container was touched; no file was deleted.

**Owner report:** stopped `hy4-nomtp` (image `vllm-rocm-hy4-stacked:test`, up
~10 h, created 2026-09-21 20:22:28 UTC), which held ~285 GB on every card via
PIDs 494054-494061. `docker stop` returned 0; container not removed (only
stopped).

## 2. GPUs are clean

KFD processes cleared immediately; VRAM lagged the stop by a short interval
(first re-poll still showed 41-185 GB mid-release), then settled to baseline:

| after stop | GPU 0-7 VRAM used | KFD PIDs |
|---|---|---|
| settled | **297,766,912 B (~284 MiB) each** | **none** |

~284 MiB is the constant driver/firmware reserve, not a leak. **All 8 GPUs are
genuinely idle.** `yzhou_model` still up, holding zero GPU memory.

## 3. GPU + RDMA survey (first-hand, cross-checked)

GPUs: 8x **MI355X (gfx950, DID 0x75a3)**, VBIOS `113-M355-01-1K1-000C`, RAS
enabled. All 9 IB devices `PORT_ACTIVE`, `link_layer: Ethernet` (RoCE v2).
ionic rails run at **400 Gb/sec (4X NDR)**; `mlx5_0` (frontend `ens3`) at
200 Gb/sec.

### Per-GPU ↔ ionic ↔ NUMA table

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

Rail id = 2nd hextet of port-1 GID index-1. GPU NUMA verified two ways
(`rocm-smi --showtoponuma` **and** `/sys/bus/pci/devices/<bdf>/numa_node`) — both
agree: GPUs 0-3 → NUMA0, 4-7 → NUMA1. ionic NUMA read from
`/sys/class/infiniband/<d>/device/numa_node` — matches its GPU for every index.

**This node is topologically identical to 137/136** (same PCI layout, same
GPU↔ionic pairing, same rail id per index: _0→08, _1→07, _2→05, _3→06, _4→04,
_5→03, _6→01, _7→02).

### Full GID index-1 / node_guid (for the record)

| ionic | node_guid | GID[1] |
|-------|-----------|--------|
| ionic_0 | 0690:81ff:fe45:7b19 | fc01:0800:480d:2d70:0690:81ff:fe45:7b19 |
| ionic_1 | 0690:81ff:fe44:3df1 | fc01:0700:470d:2d70:0690:81ff:fe44:3df1 |
| ionic_2 | 0690:81ff:fe44:4ca9 | fc01:0500:450d:2d70:0690:81ff:fe44:4ca9 |
| ionic_3 | 0690:81ff:fe45:7f39 | fc01:0600:460d:2d70:0690:81ff:fe45:7f39 |
| ionic_4 | 0690:81ff:fe45:d6b1 | fc01:0400:440d:2d70:0690:81ff:fe45:d6b1 |
| ionic_5 | 0690:81ff:fe44:6dc1 | fc01:0300:430d:2d70:0690:81ff:fe44:6dc1 |
| ionic_6 | 0690:81ff:fe44:0d79 | fc01:0100:410d:2d70:0690:81ff:fe44:0d79 |
| ionic_7 | 0690:81ff:fe45:89b9 | fc01:0200:420d:2d70:0690:81ff:fe45:89b9 |

**No defective rail.** Every ionic device has a live netdev, IB state ACTIVE,
non-zero GID. (Contrast: 135's ionic_7 had no netdev + all-zero GID — absent
here.)

## 4. GPU↔NIC NUMA verdict — the 0-3/4-7 split holds

- **Prefill (GPUs 0,1,2,3 + ionic_0-3): all NUMA0 / PCI domain 0002.** Single
  socket, zero cross-socket PCIe on the KV path, four healthy rails (08/07/05/06).
- **Decode (GPUs 4,5,6,7 + ionic_4-7): all NUMA1 / PCI domain 0003.** Single
  socket, four healthy rails (04/03/01/02).
- The two legs occupy disjoint NUMA nodes and disjoint NIC sets — **they never
  share a GPU, a NIC, or a socket.** The plan is confirmed; nothing invalidates it.

## 5. Ready-to-paste engine maps

The RDMA JSON keys are the **LOCAL visible device index** inside each container.
The decode container runs under `HIP_VISIBLE_DEVICES=4,5,6,7`, so its local
indices 0..3 map to physical GPUs 4..7 → keys "0".."3" must point at ionic_4..7.

### Prefill leg — physical GPUs 0,1,2,3
```
PREFILL_GPU_DEVICES=0,1,2,3
RDMA_DEVICE={"0":"ionic_0","1":"ionic_1","2":"ionic_2","3":"ionic_3"}
MC_TE_FILTERS=ionic_0,ionic_1,ionic_2,ionic_3
```

### Decode leg — physical GPUs 4,5,6,7 (local indices 0-3)
```
DECODE_GPU_DEVICES=4,5,6,7
RDMA_DEVICE={"0":"ionic_4","1":"ionic_5","2":"ionic_6","3":"ionic_7"}
MC_TE_FILTERS=ionic_4,ionic_5,ionic_6,ionic_7
```

Same-node same-rail note: because rail id per index is symmetric, prefill rank i
(ionic_0+i) and decode rank i (ionic_4+i) sit on **different** rails here — that
is inherent to a same-node split across two NUMA/NIC groups, not a defect. Both
legs stay NUMA-local to their own NICs, which is the property that matters for
the KV transfer path on a single node.

## 6. Host facts (for HiCache / deployment)

| item | value |
|---|---|
| frontend IP | `ens3` = **10.245.152.249/20** (matches expected) |
| libionic | `/lib/x86_64-linux-gnu/libionic.so` → `libionic.so.1.0.54.0-149.g3304be71` present |
| host RAM | **2751 GB total**, 184 GB free, **2718 GB available** (2552 GB reclaimable buff/cache) — ample for HiCache host pool |
| `/mnt/m2m_nobackup` | 28 TB, **26 TB avail** (9% used) |
| `/shared_nfs` | 410 TB, **1.9 TB avail** (100% used); read-only model store |
| model dir | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` — exists, readable, **408 GB**, 282 safetensors shards + config/tokenizer |
| `mlx5_0` | `ens3`, NUMA0, 200 Gb/sec, not part of the KV rails |

## 7. Anomalies

- `yzhou_model` (image `rocm/atom-dev:latest`, another tenant) remains up,
  holding zero GPU memory — harmless, left in place per the deletion rule.
- `/shared_nfs` is 100% full on the aggregate volume (1.9 TB free). The GLM
  model already resides there and is read-only; do NOT stage new artifacts on
  `/shared_nfs` — use `/mnt/m2m_nobackup` (26 TB free) for workspace/output.
- k8s/crusoe networking (kube-ipvs0, flannel, cni0, nodelocaldns) is present on
  the host; ignore for GPU work.
