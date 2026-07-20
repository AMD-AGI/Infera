# Infera Preflight

Cluster diagnostic tool: before running inference / PD-disaggregation, check each
node's environment up front to surface slow nodes, inconsistent configs, broken
RDMA, and mis-provisioned storage. Inspired by Primus's `preflight`.

Covers: node base info, single-node perf & topology, storage, and multi-node
(SLURM) — cross-node RoCE bandwidth + Mooncake/Mori KV-transfer measurements (see
the table) — aggregated into one HTML report.

## Checks

| Category | Check | What it looks at | Status |
| --- | --- | --- | --- |
| Base info | Host | CPU / memory / disk / NUMA / memlock limit | ✅ |
| | GPU | GPU count / model / gfx / VRAM / temp / driver | ✅ |
| | Network | plain NICs, RDMA (ionic) devices & link state, ionic↔netdev map, IP | ✅ |
| | Firmware | firmware version (MEC), GPU-direct (ais-check), kernel P2PDMA | ✅ |
| | Cross-node consistency | firmware (MEC) / driver / gfx / GPU-count comparison, warn on mismatch | ✅ |
| Single-node perf & topology | GPU compute | per-GPU bf16 GEMM throughput (TFLOPs) | ✅ |
| | HBM bandwidth | per-GPU ~1GiB random-copy read+write bandwidth (GB/s) | ✅ |
| | Intra-node interconnect | per-GPU-pair P2P copy bandwidth (GB/s), 8×8 matrix | ✅ |
| | GPU topology | inter-GPU xGMI/PCIe matrix, GPU↔NUMA mapping balance | ✅ |
| Multi-node / PD | RDMA fabric | per RDMA NIC's RoCE v2 GID / subnet / MTU (informational; reachability from the live test) | ✅ |
| | Inter-node interconnect | cross-node RoCE bandwidth (`ib_write_bw`, N×N NIC matrix, GB/s) | ✅ |
| | Mooncake KV transfer | Mooncake KV-move bandwidth: rdma / rdma-default / tcp, with the env each needs | ✅ |
| | Mori KV transfer | Mori IOEngine KV-move bandwidth (RDMA only) | ✅ |
| Storage | NVMe / KV throughput | local NVMe inventory; NVMe↔DRAM multi-threaded throughput (substrate); NVMe↔HBM single-stream staged throughput (CPU-bounce, needs GPU); KV not on local NVMe → FAIL | ✅ |

Levels are `info / warn / fail`, rolled up to the worst per node. Most checks set
**no absolute threshold** and instead compare **across nodes / within a node**
(outliers or inconsistency warn); GPU compute and intra-node interconnect add
spec-based absolute floors on **MI355X** (bf16-8k ≥ 1300 TFLOPS, xGMI per-link ≥
50 GB/s). RDMA fabric is informational only (reachability is left to the live
test) to avoid false positives from static subnet comparison.

## Usage

```bash
# Single node: collect + render report
python -m infera.tools.preflight --dump-path output/preflight

# Run only selected probes
python -m infera.tools.preflight --gpu --network
```

Outputs: one `<dump-path>/<host>.json` per node + `<dump-path>/infera_preflight_report.html`.

> Running directly on a host only does the image-independent checks; GPU perf
> (compute/HBM/P2P) and ais-check must run **inside the engine container** (see
> "Multi-node" below).
>
> Storage throughput tests the largest local NVMe mount by default;
> `--storage-path <dir>` picks a directory, `INFERA_PREFLIGHT_STORAGE_GB`
> (default 4, 0 skips) sets the volume. The NVMe↔HBM part needs torch+GPU and
> only runs inside the engine container, otherwise it's skipped.

### Multi-node (SLURM)

SLURM is the main path, one command (`run_preflight_slurm.sh`): `srun` runs one
task per node collecting in parallel into a shared dir, and rank 0 renders the
combined report — no manual steps.

```bash
# NODES, PARTITION and IMAGE are required (no defaults); other knobs are at the
# top of the script. A private image is auto-logged-in and pulled with DOCKER_TOKEN.
NODES=node1,node2 PARTITION=<partition> IMAGE=<image> ./run_preflight_slurm.sh
```

What the script does: `docker pull` on every node first (synchronized start, so
the multi-node barriers don't time out) → one container per node runs all checks
→ rank 0 renders the report. Key points:

- torch / ais-check / Mooncake / Mori only exist in the image, so they **run
  inside the engine container**; keep the image ENTRYPOINT and mount the host's
  `libionic.so`, else in-container `ib_write_bw` / Mooncake / Mori can't see the
  ionic RDMA devices.
- rank/world/node-name come from `SLURM_PROCID / SLURM_NNODES / SLURMD_NODENAME`
  forwarded into the container; `PREFLIGHT_IMAGE` is shown in the report.
- Mooncake / Mori set the env they need themselves (Mooncake sets `MC_GID_INDEX`
  / `MC_FORCE_TCP` per variant, Mori sets `MORI_RDMA_DEVICES`; shown in the
  report) — this only affects preflight, not production config.
- Without SLURM it falls back to a single node; manual fallback: `--collect-only`
  on each node, then one `--render-only`.
- Use a **fresh `--dump-path`** each run (the script clears it): rank 0 decides
  everyone has reported by counting `*.json` files in the dir.

## Report layout

- **Nodes**: node / image / collection time.
- **0. Cluster overview**: one summary table per category (rows = nodes, cols =
  sections, click to jump to detail).
- **Per-category detail (1 Base info / 2 Single-node perf & topology / 3
  Multi-node / PD / 4 Storage)**: each section, node by node; P2P, GPU topology,
  inter-node bandwidth, etc. shown as matrices.

Exit code: 2 if any `fail`, else 0 (CI-friendly).
