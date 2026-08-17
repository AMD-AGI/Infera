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

### Two nodes on a cluster without SLURM

SLURM is convenient, not required. Rank, world size and node name come from plain
environment variables — `SLURM_PROCID`, `SLURM_NNODES`, `PREFLIGHT_HOST` — and all
coordination happens through files in the shared `--dump-path`. Nothing consults a
scheduler. The variables keep their SLURM names because that is where they came
from; setting them by hand on a cluster that has never seen SLURM works the same.

**The dump path must resolve to the same storage from both nodes.** That is not a
convenience: that directory is *how the nodes find each other*. Each publishes its
NIC information there, waits for the others, then walks the pair matrix behind a
file barrier. Two unrelated local directories give you two single-node runs that
measure nothing cross-node.

#### On Kubernetes

Run one Pod per node, pinned with `nodeSelector`, both mounting the same shared
storage at the same path inside the container. The two Pods differ in four places
only: the node, the rank, the host name, and the host path that backs `/dump`.

```yaml
apiVersion: v1
kind: Pod
metadata: {name: preflight-rank0, namespace: infera}
spec:
  nodeSelector: {kubernetes.io/hostname: <NODE_A>}
  restartPolicy: Never
  hostNetwork: true                    # RDMA rails are host interfaces
  dnsPolicy: ClusterFirstWithHostNet
  initContainers:                      # infera comes from the overlay payload
  - name: infera-overlay
    image: <overlay image>
    command: ["sh","-c","cp -a /payload/. /overlay/"]
    volumeMounts: [{name: overlay, mountPath: /overlay}]
  containers:
  - name: main
    image: <the engine image you deploy with>
    command: ["/overlay/bin/infera-exec","python3","-m","infera.tools.preflight",
              "--dump-path","/dump","--mooncake"]
    env:
    - {name: SLURM_PROCID,   value: "0"}      # "1" on the other Pod
    - {name: SLURM_NNODES,   value: "2"}
    - {name: PREFLIGHT_HOST, value: "<NODE_A>"}
    securityContext: {privileged: true, capabilities: {add: ["IPC_LOCK","SYS_PTRACE"]}}
    volumeMounts:
    - {name: overlay, mountPath: /overlay, readOnly: true}
    - {name: dump,    mountPath: /dump}       # same mountPath on both Pods
    - {name: ib,      mountPath: /dev/infiniband}
    - {name: host-libionic, mountPath: /host-libionic/libionic.so, readOnly: true}
  volumes:
  - {name: overlay, emptyDir: {}}
  - {name: dump, hostPath: {path: <SHARED_DIR_ON_THIS_NODE>, type: DirectoryOrCreate}}
  - {name: ib,   hostPath: {path: /dev/infiniband, type: Directory}}
  - name: host-libionic
    hostPath: {path: /usr/lib/x86_64-linux-gnu/libionic.so.1, type: File}
```

Apply both, then read the report from the shared directory:

```bash
kubectl apply -f preflight-rank0.yaml -f preflight-rank1.yaml
kubectl -n infera wait --for=jsonpath='{.status.phase}'=Succeeded   pod/preflight-rank0 pod/preflight-rank1 --timeout=20m
# <shared-dir>/infera_preflight_report.html
```

Why each piece is there:

| | |
|---|---|
| the overlay initContainer | `infera` is not in the engine image — it comes from the overlay payload, and `infera-exec` is what puts it on `PYTHONPATH` |
| `privileged` + `/dev/infiniband` | without them the container sees **zero** RDMA devices, which is indistinguishable from the node having none |
| the host `libionic.so` mount | the vendor image's libionic must match the host's ionic kernel ABI, or libibverbs rejects every device |
| `hostNetwork: true` | the RDMA rails are host interfaces; the pod network cannot reach them |
| `<SHARED_DIR_ON_THIS_NODE>` differing per Pod | the *container* path must match (`/dump`); the host paths only have to reach the same storage. On a fleet where one node mounts the array locally and the other over NFS, they will not look alike |

```{admonition} netperf is skipped in the engine image
:class: warning
The cross-node `ib_write_bw` matrix needs the `perftest` package, which these
engine images do not ship. You get

    WARN: [netperf] netperf skipped (no ib_write_bw or RDMA device)

That message names two causes and it is almost always the first one — verified on
this fleet, where the same Pod reported **8** RDMA devices via
`ibv_get_device_list` while `command -v ib_write_bw` came back empty. Do not read
it as "no RDMA".

`--mooncake` does **not** need `perftest` and does run, which is the more useful
measurement anyway: it reports KV-move bandwidth over `rdma`, `rdma-default` and
`tcp` separately, so a fabric that will silently serve at TCP speed shows up as a
number.
```

#### On plain hosts over SSH

Same idea without the Pod wrapper — two shells, started together:

```bash
# --- node A ---
export DUMP=/shared/preflight-run1            # same storage, reachable from both
SLURM_PROCID=0 SLURM_NNODES=2 PREFLIGHT_HOST=$(hostname) \
  infera-preflight --dump-path "$DUMP" --netperf --mooncake

# --- node B, at the same time ---
SLURM_PROCID=1 SLURM_NNODES=2 PREFLIGHT_HOST=$(hostname) \
  infera-preflight --dump-path "$DUMP" --netperf --mooncake
```

Ranks must be `0` and `1`. Rank 0 waits for the others and renders
`infera_preflight_report.html`; the rest exit once they have published.

```{admonition} Start them together
:class: warning
The nodes wait for each other, but not indefinitely: 120 s for the initial exchange
of NIC information and 300 s for rank 0 to see every node's JSON. Start the second
node a few minutes late and you get a report with the cross-node sections missing
and `WARN: only 1/2 node reports before timeout` on stderr — which reads as a
fabric problem and is not one. The per-pair barrier is far more generous (30 min),
since a single bandwidth measurement can legitimately take a while.
```

#### No shared filesystem at all

Collect separately and render once. This measures **per-node state only** — the
cross-node probes have no way to coordinate, so there is no RoCE matrix and no
Mooncake comparison:

```bash
infera-preflight --collect-only --dump-path /local/preflight     # on each node
# copy every <host>.json into one directory, then anywhere:
infera-preflight --render-only --dump-path /collected
```

```{admonition} Use a fresh dump path for every run
:class: tip
Rank 0 decides everyone has reported by counting `*.json` in the directory. A
leftover file from an earlier run makes it render immediately, with one node's data
silently coming from that earlier run. `run_preflight_slurm.sh` clears the path for
you; by hand, you have to.
```


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
