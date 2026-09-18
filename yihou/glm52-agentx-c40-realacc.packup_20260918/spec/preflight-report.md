# Preflight report — GLM-5.2 1P1D AgentX (yihou-agentx-hicache)

Operator: yihou. Nodes: prefill **crsuse2-m2m-135** (10.245.148.209), decode
**crsuse2-m2m-138** (10.245.157.237). Run shape: P4+DPA / D4+DPA on GPU devices
**2,3,4,5** of each node. Image: `infera-sglang:v0519-yihou-0917-nextnfix-hicache`.

## Verdict: GO (inventory-based; probe runs stood down)

Occupancy, the same-rail RDMA fabric for the run's GPUs (2,3,4,5), and image
presence are all clean on both nodes.

**Probe runs were stood down (team-lead call).** R05 at ~09:57 today ran the
real engine on these exact nodes, the same GPUs 2-5, the same pinned
`RDMA_DEVICE` JSON, and passed with 0 Mooncake failures and 0 router-affinity
503s. Nothing on the fabric has changed since; this round's only delta is a thin
image layer over three Python/CUDA *source* files, which the RDMA/Mooncake path
does not touch. A probe re-validation buys nothing over that end-to-end result.

Accordingly, the `preflight.sh` runs in §3-§4 are recorded as **harness
findings, not fabric findings** — the faults they surfaced were in the probe's
own invocation, not in the system under test. The one first-hand fabric datum
kept is §5: a pinned probe on today's image transferred cleanly on all four run
GPUs in both directions, consistent with R05. GO rests on R05 + inventory, not
on the probe.

---

## 1. Occupancy

Both nodes: GPUs **2,3,4,5 are free** (util 0%, ~298 MB baseline VRAM of 309 GB).

**crsuse2-m2m-135**
- GPU[0,2,3,4,5,6,7]: idle, ~297,766,912 B VRAM, util 0%.
- GPU[1]: **31% VRAM (96.94 GB)** — root-owned k8s vLLM pod `pod5d84e491`
  (Qwen3-32B), ~22d uptime. **Untouched by design; the run avoids GPU[1].**
  `check_nodes.sh` flags this and that is expected.
- Containers (all Exited except monitoring): `crusoe-vector` (Up, monitoring,
  not a GPU/PD container), `glm52-server` Exited(0) 9d, `glm52-main-prstack-c8`
  Exited(1) 10d, `topk-dev` Exited(137) 11d, `glm52-qr_int4` Exited(0) 2w.
  None running that hold GPUs 2-5.

**crsuse2-m2m-138**
- GPU[0..7]: all idle, ~297,754,624 B VRAM, util 0%.
- Container: **`glm52pd-decode-0-limou`** Exited(0) 6h ago, image
  `infera/engine-sglang:sgl-sep01-aiter-sep06`. **Reported, NOT removed** — its
  name lacks `yihou` (per authorisation). Holds no VRAM (Exited).

## 2. Same-rail RDMA sanity (authoritative sysfs, both nodes)

Run pins each visible GPU to one HCA via
`RDMA_DEVICE='{"0":"ionic_2","1":"ionic_3","2":"ionic_4","3":"ionic_5"}'` and
`MC_TE_FILTERS=ionic_2,ionic_3,ionic_4,ionic_5` (from `config.yihou.base.sh`).
The JSON keys are **LOCAL visible-device indices** under
`HIP_VISIBLE_DEVICES=2,3,4,5`, so each key resolves to a physical GPU as:

| JSON key | local idx | physical GPU | HCA |
|----------|-----------|--------------|-----|
| "0" | 0 | GPU2 | ionic_2 |
| "1" | 1 | GPU3 | ionic_3 |
| "2" | 2 | GPU4 | ionic_4 |
| "3" | 3 | GPU5 | ionic_5 |

Check the mapping against this table rather than trusting it. Both `RDMA_DEVICE`
and `MC_TE_FILTERS` must be exported for the run: with only the library mount,
Mooncake falls back to auto-discovery and the two ends independently pick
different HCAs from the NUMA-local pool. These rails are physically isolated
(`ionic_i` reaches only `ionic_i`), so a mismatch is *unreachable* and shows up
as GPUs 4-7 failing both directions while 0-3 pass — a symmetric, clean-looking
failure that reads like dead hardware and is not (documented in
`yihou/glm52-1p1d-samerail-c32-c40.packup_20260918/notes.md` §1).

All four target HCAs are healthy on both nodes:

| HCA | 135 state | 135 sysfs GID[1] | 135 netdev | 138 state | 138 sysfs GID[1] | 138 netdev |
|-----|-----------|------------------|------------|-----------|------------------|------------|
| ionic_2 | PORT_ACTIVE | non-zero | enP2p0s11 | PORT_ACTIVE | non-zero | enP2p0s11 |
| ionic_3 | PORT_ACTIVE | non-zero | enP2p0s12 | PORT_ACTIVE | non-zero | enP2p0s12 |
| ionic_4 | PORT_ACTIVE | non-zero | enP3p0s9  | PORT_ACTIVE | non-zero | enP3p0s9  |
| ionic_5 | PORT_ACTIVE | non-zero | enP3p0s10 | PORT_ACTIVE | non-zero | enP3p0s10 |
| ionic_7 | PORT_ACTIVE | **ALL-ZERO** | **NONE** | PORT_ACTIVE | non-zero | enP3p0s12 |

- **135 ionic_7 confirmed the sole defective HCA**: authoritative sysfs
  `/sys/class/infiniband/ionic_7/ports/1/gids/1` is all-zero and it has no
  associated netdev. NOTE: `ibv_devinfo -d ionic_7 -v` misleadingly prints a
  *non-zero* GID[1] — that value is stale; **sysfs is ground truth**. The run
  avoids ionic_7 by design (uses ionic_2..5 only).
- 138 ionic_7 is healthy (not defective there); irrelevant to the run.

## 3. Repo preflight (`preflight.sh`) — HARNESS FINDINGS ONLY

These are findings about the **probe tooling**, not the fabric. Two reusable
gotchas about `preflight.sh` came out of the two runs before stand-down:

1. **`preflight.sh` sources no config file.** Reading it end to end: it takes
   only `IMAGE=` plus node names. There is no `CONFIG=`/`TOPOLOGY=` parameter, so
   `config.sh:104`'s `HOST_RDMA_LIB` default can never reach it.
2. **It mounts the host GPUDirect provider only when the caller exports
   `HOST_RDMA_LIB`** (preflight.sh:62-63: `[[ -z "${HOST_RDMA_LIB:-}" ]] ||`).
   Without that export the container uses its stale in-image `libionic` and every
   GPU-memory RDMA transfer fails; with it, the log shows
   `infera-inject-host-ionic: replaced ... with host build`.
3. It enumerates **all 8 physical GPUs** (no `HIP_VISIBLE_DEVICES` restriction),
   and it does **not** apply the run's per-GPU `RDMA_DEVICE` JSON — it either
   auto-discovers or takes a single `INFERA_PREFLIGHT_RDMA_DEVICE`. So GPUs
   0,1,6,7 having no run mapping and GPUs 4-7 failing under auto-discovery are
   **expected harness artifacts, not NO-GO signals**; the run uses only 2,3,4,5.

Command of the second run (with the mount):
```
IMAGE=infera-sglang:v0519-yihou-0917-nextnfix-hicache \
OUT_DIR=<workspace>/preflight-libionic \
HOST_RDMA_LIB=/lib/x86_64-linux-gnu/libionic.so \
./preflight.sh crsuse2-m2m-135 crsuse2-m2m-138
```
(First run WITHOUT `HOST_RDMA_LIB` → `preflight/`, exit 1: every GPU transfer
failed both directions because the host ionic GPUDirect provider was not
injected. The real run mounts it via config.sh:104 / engine.sh:106-109, so the
mount is required; second run adds it → `preflight-libionic/`.)

Results with the libionic mount (`preflight-libionic/`):
- **network: OK** on both nodes.
- CPU-mem RDMA: **rdma-default 7.28 / 7.25 GB/s** both directions, verified.
- GPU-mem RDMA (auto HCA-selection): **gpu0-3 PASS ~45 GB/s; gpu4-7 FAIL
  (transfer_failed)**, both directions. Uniform lower/upper-half split.

## 4. Why gpu4-7 fail in auto mode, and why it is not a blocker

The split is a **known, reproducible auto-HCA-selection artifact** on these two
nodes, established first-hand by the prior task's differential reference
(`results/yihou-1p1d-c64/preflight/`):

- Prior **auto** run (`run1`) shows the *identical* split: gpu0-3 ~45 GB/s,
  gpu4-7 transfer_failed. Same nodes, reproduced today on a newer image.
- Prior **pinned** runs (`INFERA_PREFLIGHT_RDMA_DEVICE=<dev>` + `MC_GID_INDEX=1`,
  the same knobs the real run uses via its RDMA_DEVICE map) make **all of
  gpu2,3,4,5 pass in both directions**:
  - pin ionic_4: gpu2 29.7, gpu3 30.0, gpu4 41.1, gpu5 41.1 GB/s (both dirs).
  - pin ionic_5: gpu2 28.7, gpu3 29.9, gpu4 41.1, gpu5 40.9 GB/s (both dirs).

So the failing half is purely an artifact of leaving HCA selection to mooncake's
default; the real run never does that — it pins per-GPU to healthy ionic_2..5.

## 5. First-hand pin probe on TODAY's image (`preflight-pin/pin-ionic_4/`)

Command (`pin_probe.yihou.sh`, adapted from the prior task's tool):
```
IMAGE=infera-sglang:v0519-yihou-0917-nextnfix-hicache \
HOST_RDMA_LIB=/lib/x86_64-linux-gnu/libionic.so \
./pin_probe.yihou.sh ionic_4 pin-ionic_4
# sets INFERA_PREFLIGHT_RDMA_DEVICE=ionic_4, MC_GID_INDEX=1
```

All four run GPUs pass **both directions** on today's image, pinned:

| GPU | forward (→135) | reverse (→138) |
|-----|----------------|----------------|
| gpu2 | 29.87 GB/s ✓ | 28.91 GB/s ✓ |
| gpu3 | 27.91 GB/s ✓ | 29.97 GB/s ✓ |
| gpu4 | 40.40 GB/s ✓ | 41.27 GB/s ✓ |
| gpu5 | 40.32 GB/s ✓ | 41.22 GB/s ✓ |

Container log confirms the host provider was injected
(`infera-inject-host-ionic: replaced ... with host build`).

**Scope note (honest):** this probe pins a single HCA (ionic_4) for all GPUs;
the run uses a per-GPU map (gpu2→ionic_2, gpu3→ionic_3, gpu4→ionic_4,
gpu5→ionic_5). Coverage of the run's exact map rests on three legs: (a) HCA
health of ionic_2..5 verified via sysfs GID/netdev (§2), fabric-level and
image-independent; (b) each of gpu2,3,4,5 verified to drive a healthy
bidirectional transfer today (above); (c) the prior task's per-HCA pin runs
(`pin-ionic_2/3/4/5`) each passed. ionic_2/ionic_3/ionic_5 were not re-pinned
individually on today's image — the mooncake/RDMA path is untouched by the
nextn/hicache thin layers, so this is a low-risk gap, not a blocker.

## Anomalies stated plainly

- Repo `preflight.sh` auto mode reports gpu4-7 fail. This is a **harness
  artifact** (auto HCA-discovery + missing per-GPU pin), not a fabric fault —
  see §3. It does not enter the GO/NO-GO judgement, which rests on R05 (real
  engine, ~09:57 today, same nodes/GPUs/pin, 0 Mooncake failures) plus the
  inventory here. The §5 pinned probe corroborates but is not load-bearing.
- 135 GPU[1] busy (k8s Qwen3-32B) — expected, avoided, never masked.
- 138 `glm52pd-decode-0-limou` Exited — left in place (name lacks yihou).
- No `infera-preflight-*` / `yihou-pinprobe-*` container left running on either
  node (both CLEAR after stand-down).
