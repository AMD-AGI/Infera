# Preflight

A cluster diagnostic to run **before** inference or PD bring-up, so slow nodes,
inconsistent configs, broken RDMA and mis-provisioned storage surface as a report
rather than as a deployment that hangs or runs at a fraction of the speed it should.

Ships as a console script:

```bash
infera-preflight --dump-path output/preflight
```

or equivalently `python -m infera.tools.preflight`.

## Why it exists

Most of what it checks fails **silently**. RDMA that the container cannot see does
not raise — Mooncake falls back to a transport that cannot cross nodes, and the
symptom is a request that never returns while every pod stays Ready. A KV path on
network storage does not raise either; weight load simply takes an order of
magnitude longer, exceeds the ready timeout, and presents as a crash loop. Neither
tells you what is actually wrong.

## Checks

| Category | Covers |
|---|---|
| Base info | CPU / memory / disk / NUMA / memlock; GPU count, model, gfx, VRAM, driver; plain NICs and **RDMA (ionic) devices with link state**, ionic↔netdev map; firmware (MEC), GPU-direct (`ais-check`), kernel P2PDMA |
| Cross-node consistency | firmware / driver / gfx / GPU-count compared **between** nodes, warning on mismatch |
| Single-node perf | per-GPU bf16 GEMM throughput; HBM bandwidth; per-GPU-pair P2P bandwidth as an 8×8 matrix; xGMI/PCIe topology and GPU↔NUMA balance |
| Multi-node / PD | per-NIC RoCE v2 GID / subnet / MTU; **cross-node RoCE bandwidth** (`ib_write_bw`, N×N NIC matrix); **Mooncake KV-transfer bandwidth measured separately over `rdma`, `rdma-default` and `tcp`**; Mori IOEngine KV-move bandwidth |
| Storage | local NVMe inventory; NVMe↔DRAM and NVMe↔HBM throughput; **KV not on local NVMe → FAIL** |

Findings are `info` / `warn` / `fail`, rolled up to the worst per node. Most checks
set **no absolute threshold** — they compare across nodes and within a node, so an
outlier or an inconsistency is what warns. GPU compute and intra-node interconnect
add spec-based floors on MI355X. RDMA fabric is informational only, deliberately:
comparing subnets statically produces false positives, so reachability is left to
the live bandwidth test.

```{admonition} The Mooncake rows are the ones to read for PD
:class: tip
They report KV-move bandwidth over RDMA and over TCP **separately**. A fabric that
will silently serve at TCP speed therefore appears as a number, instead of as a
deployment that is inexplicably slow — which is the failure this tool exists to
convert into a fact.
```

## Usage

```bash
# single node: collect and render
infera-preflight --dump-path output/preflight

# selected probes only
infera-preflight --netperf --mooncake
```

Outputs one `<dump-path>/<host>.json` per node plus a combined
`<dump-path>/infera_preflight_report.html`.

```{admonition} Run it inside the engine container for the full set
:class: warning
On a bare host only the image-independent checks run. GPU perf (compute / HBM /
P2P), `ais-check`, Mooncake and Mori exist only in the engine image, so those are
skipped outside it. Keep the image ENTRYPOINT and mount the host's `libionic.so`,
or in-container `ib_write_bw` / Mooncake / Mori cannot see the ionic RDMA devices —
which looks identical to not having any.
```

Storage throughput tests the largest local NVMe mount by default. `--storage-path
<dir>` picks a directory; `INFERA_PREFLIGHT_STORAGE_GB` sets the volume (default 4,
`0` skips). The NVMe↔HBM part needs torch and a GPU and is skipped otherwise.

### Multi-node

SLURM is the main path — one command runs one task per node in parallel into a
shared directory, and rank 0 renders the combined report:

```bash
NODES=node1,node2 PARTITION=<partition> IMAGE=<image> \
  infera/tools/preflight/run_preflight_slurm.sh
```

`NODES`, `PARTITION` and `IMAGE` are required and have no defaults. Rank, world size
and node name come from `SLURM_PROCID` / `SLURM_NNODES` / `SLURMD_NODENAME`.
Mooncake and Mori set the env each variant needs themselves — `MC_GID_INDEX`,
`MC_FORCE_TCP`, `MORI_RDMA_DEVICES` — and the report shows which; that affects
preflight only, not production config.

**Use a fresh `--dump-path` each run.** Rank 0 decides everyone has reported by
counting `*.json` in the directory, so a stale file makes it render early. The
script clears the path for you.

Without SLURM it falls back to a single node. The manual equivalent is
`--collect-only` on each node, then one `--render-only` over the shared directory.

## Report layout

**Nodes** — node, image, collection time. **0. Cluster overview** — one summary
table per category, rows are nodes, click through to detail. Then per-category
detail: base info, single-node perf and topology, multi-node / PD, storage — with
P2P, GPU topology and inter-node bandwidth rendered as matrices.

## The other preflight: config validation at launch

Separate from this tool, `infera/common/disagg_preflight.py` runs automatically
when a disaggregated worker starts, **before** the engine subprocess, and fails
fast rather than hanging. It catches the two silent-failure modes of cross-node PD:

- a worker advertising a non-routable host (`0.0.0.0`, `127.0.0.1`) to etcd, so the
  peer and the router cannot reach its bootstrap or KV endpoint;
- configurations prone to silent TCP fallback, which is 5–20× slower than RDMA.

It is pure config validation with no hardware probing, so it runs anywhere
including CI — and cannot tell you the NIC itself is healthy. That is what the tool
above is for.

## Source

[`infera/tools/preflight/`](https://github.com/AMD-AGI/Infera/tree/main/infera/tools/preflight)
