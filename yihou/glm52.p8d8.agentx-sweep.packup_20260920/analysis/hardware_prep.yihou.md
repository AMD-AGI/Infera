# Hardware prep — 137 (prefill) + 136 (decode)

Role: `hwprep`. Made 137/136 ready for a fresh 1P1D deployment and produced the
topology inputs for both an 8-GPU and a 4-GPU shape. All measured first-hand
2026-09-18. Scope was **137 and 136 only**; 135/138 (another session's live
experiment) were not touched.

## 1. Teardown — what was removed

Removed only containers whose name contains `yihou` (`docker rm -f`). Left every
non-ours container running/exited untouched (`peaceful_haibt`, `crusoe-vector`,
`crusoe-amd-log-collector`, `crusoe-amd-exporter`, and the non-yihou `glm52-*`
leftovers).

**crsuse2-m2m-137 — 6 removed:**
- `glm52c72-router-yihou`  (was Up 3h, image `infera-sglang:v0519-yihou-r2`)
- `glm52c72-prefill-yihou` (was Up 3h, image `infera-sglang:v0519-yihou`)
- `glm52c72-etcd-yihou`    (was Up 6h)
- `glm52a-etcd-yihou`      (was Exited)
- `glm52a-router-yihou`    (was Up 9h, image `rocm/infera:sglang-v0.2.10`)
- `glm52a-prefill-yihou`   (was Up 29h, image `rocm/infera:sglang-v0.2.10`)

**crsuse2-m2m-136 — 2 removed:**
- `glm52c72-decode-yihou`  (was Up 3h, image `infera-sglang:v0519-yihou`)
- `glm52a-decode-yihou`    (was Up 9h, image `rocm/infera:sglang-v0.2.10`)

Left untouched (not ours): on 137 `peaceful_haibt` (base image
`lmsysorg/sglang-rocm:v0.5.19-...`, Up but **no GPU devices mounted** —
`docker inspect` shows empty `HostConfig.Devices`), `crusoe-vector`, and the
Exited non-yihou `glm52-*` / `agitated_chandrasekhar`; on 136 `crusoe-vector`,
`crusoe-amd-log-collector`, `crusoe-amd-exporter`, and the Exited non-yihou
`glm52-*` / `infera-download-*`.

## 2. Idle-VRAM evidence (all 8 GPUs, both nodes)

Before teardown all GPUs were ~262 GB (137) / ~265 GB (136) of 309.22 GB used,
held by `sglang::schedul` processes. After removing our containers, VRAM released
(not instantaneous — re-checked after a short wait):

| node | GPU 0-7 used after teardown | compute pids |
|------|-----------------------------|--------------|
| 137  | 297,754,624 B (~284 MiB) each | none |
| 136  | 297,754,624 B (~284 MiB) each | none |

~284 MiB is the constant driver/firmware reserve, not a leak. `rocm-smi
--showpids` shows **no** sglang/python/vllm compute processes on either node. **All
16 GPUs are genuinely idle.**

## 3. RDMA rail survey (re-measured, not trusted)

Full per-GPU table, GID values, and the ready-to-paste engine maps are in
`../scripts/rdma_map.yihou.md`. Summary:

| node | ionic devices | all ACTIVE + live netdev + non-zero GID? | defective rail |
|------|---------------|------------------------------------------|----------------|
| 137  | ionic_0..7    | yes                                      | none |
| 136  | ionic_0..7    | yes                                      | none |

- `GPU_n ↔ ionic_n` is NUMA-local for all n on both nodes (GPU and NIC share the
  same PCI domain: 0002/NUMA0 for 0-3, 0003/NUMA1 for 4-7). Re-confirms the
  `rdma.survey.8node.packup_20260917` pairing.
- Rail id (GID 2nd hextet) per index is **identical across 137 and 136**:
  ionic_0→08, _1→07, _2→05, _3→06, _4→04, _5→03, _6→01, _7→02. So `GPU_n↔ionic_n`
  on both legs yields an automatic same-rail KV path.
- **No dead rail** on 137/136 — the 135 `ionic_7` defect (no netdev, all-zero GID)
  is absent here.

## 4. Shape recommendations

- **P8D8:** all 8 GPUs / ionic_0..7 on both nodes. Every rail healthy.
- **P4D4: recommend GPUs 0,1,2,3** (NUMA0, ionic_0-3, rails 08/07/05/06) — single
  NUMA node, single PCI domain, four healthy contiguous rails, no external tenant.
  GPUs 4,5,6,7 are an equally valid single-NUMA mirror. This deliberately does NOT
  copy the `samerail` packup's 2,3,4,5 quad, which straddled NUMA and was forced by
  135-specific constraints that do not apply on 137/136.

## 5. Anomalies

None material. `peaceful_haibt` (base image, another tenant, no GPUs) remains up on
137 — harmless, left in place per the deletion rule. VRAM release lagged teardown by
the expected few seconds; confirmed clean on re-check.
