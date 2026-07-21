# PD disaggregation

```{admonition} One-pager
:class: tip
**What:** run **prefill** (read the prompt) and **decode** (generate tokens) on
separate GPUs/nodes, and move the KV cache between them. **Why:** the two phases
have different hardware appetites; splitting them lets you scale each
independently. **Cost:** a KV transfer over RDMA and a bit of bring-up care.
```

```{graphviz}
digraph pd_flow {
  rankdir=LR; bgcolor="transparent";
  node [shape=box style="rounded,filled" fillcolor="#eef2f7" color="#5577cc" fontname="Helvetica,Arial,sans-serif" fontsize=11 margin="0.22,0.13"];
  edge [fontname="Helvetica,Arial,sans-serif" fontsize=10 color="#5577cc"];

  C   [label="Client" fillcolor="#f4f4f4" color="#999999"];
  RTR [label="Router\nauto-detects a\nprefill+decode pair" fillcolor="#fff3cd" color="#caa300"];
  P   [label="Prefill worker\nencode whole prompt → build KV\ncompute-bound"];
  D   [label="Decode worker\ngenerate one token at a time\nbandwidth-bound"];

  C -> RTR;
  RTR -> P;
  P -> D [label="KV cache over RDMA\n(Mooncake / MoRI)" penwidth=1.9];
  D -> C [label="stream tokens" style=dashed color="#8a8a8a"];
}
```

## Prefill vs decode

| Phase | Work | Bottleneck |
|---|---|---|
| **Prefill** | encode the whole prompt at once | compute-bound (big matmuls) |
| **Decode** | generate one token at a time | memory-bandwidth-bound, loves high concurrency |

Run them on the same GPU and they contend. Disaggregate ("PD") and you can put,
say, **1 prefill worker feeding 2 decode workers** (1P2D), and size each side to
the load. The router auto-detects a valid prefill+decode pair and dispatches to
both, coordinating the KV hand-off.

## When to disaggregate

PD isn't free — every request pays a KV transfer over the fabric. For a
**balanced** workload (moderate input:output ratio) a single **aggregated** worker
with [KV-aware routing](kv_aware_routing.md) is usually simpler and just as fast.
Reach for PD when one phase dominates or you need to scale the two independently:

- **Long inputs, short outputs** — prefill dominates, and a big prompt on a shared
  GPU stalls everyone else's decode (head-of-line blocking). Dedicated prefill
  workers keep decode flowing.
- **Decode-heavy / high concurrency** — give decode its own pool (and its own,
  often larger, TP) sized for bandwidth and batch size.
- **Independent scaling** — add prefill for longer inputs, or decode for more
  concurrent users, without resizing the other side.
- **Agentic / cache-reuse pipelines** — pairs well with KV-aware routing and a
  tiered cache under it.

```{admonition} Start simple
:class: tip
Start **aggregated**, measure, and disaggregate once prefill/decode interference
or queue buildup actually shows up. And PD only pays off over a real RDMA fabric —
over TCP it's *slower* (see the [self-check](#self-check-is-rdma-actually-being-used)).
```

## Topologies

You'll see shorthand like **1P1D** (1 prefill + 1 decode), **2P1D**, **1P2D**.
More prefill helps long inputs; more decode helps high concurrency.

## Launching a PD pair

The router stays connector-agnostic — each worker advertises its disagg protocol
in etcd and the router shapes the request body accordingly.

```{admonition} Add the no-broker dev flags
:class: important
Every command below also needs `--discovery-backend etcd --request-transport
http --kv-event-transport zmq` (omitted in some blocks for brevity) — the same
no-broker path the [Quickstart](../getting_started/quickstart.md)
uses — plus a server started with those flags. Drop them only if you're on the
production NATS + Kubernetes plane.
```

Each engine tab shows the **Mooncake** and **MoRIIO** variants — pick the
connector, run the prefill and decode commands. The connector must match on both
legs. (Dev flags `--discovery-backend etcd --request-transport http
--kv-event-transport zmq` are omitted below per the note above.)

::::{tab-set}

:::{tab-item} SGLang (native bootstrap)
SGLang selects the transport with `--disaggregation-transfer-backend`:

**Mooncake**
```bash
# prefill
HIP_VISIBLE_DEVICES=1 python -m infera.engine.sglang \
  --model-path <model> --port 30001 --host 0.0.0.0 --etcd-endpoint <etcd>:2379 \
  --disaggregation-mode prefill --disaggregation-bootstrap-port 8998 \
  --disaggregation-transfer-backend mooncake
# decode
HIP_VISIBLE_DEVICES=2 python -m infera.engine.sglang \
  --model-path <model> --port 30002 --host 0.0.0.0 --etcd-endpoint <etcd>:2379 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake
```

**MoRIIO (MoRI)**
```bash
# prefill
HIP_VISIBLE_DEVICES=1 python -m infera.engine.sglang \
  --model-path <model> --port 30001 --host 0.0.0.0 --etcd-endpoint <etcd>:2379 \
  --disaggregation-mode prefill --disaggregation-bootstrap-port 8998 \
  --disaggregation-transfer-backend mori
# decode
HIP_VISIBLE_DEVICES=2 python -m infera.engine.sglang \
  --model-path <model> --port 30002 --host 0.0.0.0 --etcd-endpoint <etcd>:2379 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mori
```
:::

:::{tab-item} vLLM
vLLM selects the transport via the `kv_connector` in `--kv-transfer-config`:

**Mooncake**
```bash
# prefill / producer
HIP_VISIBLE_DEVICES=1 VLLM_HOST_IP=<routable> python -m infera.engine.vllm \
  --model <model> --port 30001 --host 0.0.0.0 \
  --advertise-host <routable> --etcd-endpoint <etcd>:2379 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
# decode / consumer
HIP_VISIBLE_DEVICES=2 VLLM_HOST_IP=<routable> python -m infera.engine.vllm \
  --model <model> --port 30002 --host 0.0.0.0 \
  --advertise-host <routable> --etcd-endpoint <etcd>:2379 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

**MoRIIO (MoRI)** — read/pull mode; set `VLLM_MORIIO_CONNECTOR_READ_MODE=1` on **both** legs:
```bash
# prefill / producer
HIP_VISIBLE_DEVICES=1 VLLM_HOST_IP=<routable> VLLM_MORIIO_CONNECTOR_READ_MODE=1 \
python -m infera.engine.vllm \
  --model <model> --port 30001 --host 0.0.0.0 \
  --advertise-host <routable> --etcd-endpoint <etcd>:2379 \
  --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_producer"}'
# decode / consumer
HIP_VISIBLE_DEVICES=2 VLLM_HOST_IP=<routable> VLLM_MORIIO_CONNECTOR_READ_MODE=1 \
python -m infera.engine.vllm \
  --model <model> --port 30002 --host 0.0.0.0 \
  --advertise-host <routable> --etcd-endpoint <etcd>:2379 \
  --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_consumer"}'
```
:::

:::{tab-item} ATOM
ATOM PD uses the **Mooncake** connector (`atom-mooncake`) only — MoRIIO is not an
ATOM transport. The connector goes in `--kv-transfer-config` (note the lowercase
`mooncake` and the extra handshake fields ATOM needs):

```bash
# prefill / producer — note /v1/completions only (see below)
HIP_VISIBLE_DEVICES=1 MC_DISABLE_HIP_TRANSPORT=1 RDMAV_FORK_SAFE=1 \
python -m infera.engine.atom \
  --model <model> --server-port 30001 --host 0.0.0.0 -tp 1 \
  --advertise-host <routable> --etcd-endpoint <etcd>:2379 \
  --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","handshake_port":6301,"http_port":30001,"proxy_ip":"<routable>","ib_device":"<rdma-nic>"}'
# decode / consumer — kv_role=kv_consumer, its own port/handshake_port
```
:::

::::

Then send the same chat `curl` — the router uses PD automatically.

```{admonition} ATOM is /v1/completions-only for PD
:class: note
ATOM threads the KV-transfer params only on the text-completion path, so
disaggregated ATOM requests go to `/v1/completions`, not `/v1/chat/completions`.
```

## Cross-node prerequisites (RDMA)

Intra-host PD "just works." Cross-host PD moves the KV over RDMA and needs care:

- **Same RDMA rail** — prefill and decode must use matching `ib_device` (e.g.
  both `ionic_0`). Mismatched rails fail with `received packet mismatch`.
- **GID index** — for cross-node RoCEv2 set the transport's GID-index env to `1`:
  `MC_GID_INDEX=1` for **Mooncake**, `MORI_IB_GID_INDEX=1` for **MoRIIO** (and
  `NCCL_IB_GID_INDEX=1` for the collectives path). Infera sets these defaults on
  ROCm for you; the default index 0 is link-local and times out on `QP → RTR`
  between hosts.
- **Force RDMA, not the intra-node shortcut** — Mooncake: `MC_DISABLE_HIP_TRANSPORT=1`
  (else its XGMI path advertises an empty segment the peer rejects).
- **Routable addresses + shared etcd** — `--advertise-host <node-ip>`, all
  workers/servers on one reachable `--etcd-endpoint`.

## Self-check: is RDMA actually being used?

The dominant PD support issue is a **silent TCP fallback** — the run works but
TTFT is 5–20× worse than the RDMA numbers. Infera self-checks for this so you
don't have to spot it by eye:

- **Launch preflight (automatic).** Before the engine starts, a disaggregated
  worker refuses to come up if it would advertise a non-routable host
  (`0.0.0.0` / `127.0.0.1`) or if its transport isn't an RDMA backend — both
  fail fast with an actionable message instead of a slow run.
- **Transport probe (automatic, runtime).** Infera inspects the negotiated
  transport (via `/sys/class/infiniband`) and **refuses to run PD over TCP**: if
  the probed transport isn't RDMA or xGMI it aborts rather than serve at
  degraded TTFT. Override only for benchmarks with `--disaggregation-allow-tcp`.

Manual checks when bringing up a new fabric:

```bash
# install the tools if not present
apt-get install -y libibverbs-utils perftest

ibv_devinfo            # RDMA NIC present + port ACTIVE?
show_gids              # find the RoCEv2 (v2) GID index -> set MC_GID_INDEX
ib_write_bw <peer>     # RDMA bandwidth host-to-host (run server on one, client on the other)
```

If `ib_write_bw` can't connect across hosts, fix the fabric (rail, GID, routable
IPs) before debugging the engines — PD can't outrun a broken RDMA path.

## Choosing an engine

| | SGLang | vLLM |
|---|---|---|
| KV transports | native bootstrap; `mooncake` / `mori` | `MooncakeConnector` / `MoRIIOConnector` |
| best at | high-concurrency decode (DP-attention) | broad model coverage |

## Transports: Mooncake vs MoRIIO

Both are **RDMA KV transports over the same ionic RoCEv2 fabric** — they move the
prefill worker's KV to the decode worker; the [self-check](#self-check-is-rdma-actually-being-used)
applies to both. They differ in who initiates the transfer and how D finds P:

| | **Mooncake** | **MoRIIO** (MoRI-IO) |
|---|---|---|
| Model | P2P: prefill RDMA-writes KV into decode's GPU | AMD MoRI IO layer, **read/pull**: decode RDMA-reads KV from prefill |
| Peer discovery | decode finds prefill via a **bootstrap server** (`bootstrap_addr`) | ZMQ notify between the pair |
| Pairs with | vLLM, SGLang, or ATOM (same engine on both legs) | vLLM (`MoRIIOConnector`) and SGLang (`mori` backend) — not ATOM |
| vLLM connector | `MooncakeConnector` | `MoRIIOConnector` |
| SGLang flag | `--disaggregation-transfer-backend mooncake` | `--disaggregation-transfer-backend mori` |

## Environment variables

Set on **each engine** (prefill and decode). These are the cross-node RDMA knobs;
flags like `--advertise-host` / `--kv-transfer-config` cover the rest.

| Env | Typical | What it does |
|---|---|---|
| `MC_GID_INDEX` | `1` | Mooncake RoCEv2 GID index — index 0 is link-local and times out cross-node. |
| `NCCL_IB_GID_INDEX` | `1` | RoCEv2 GID index for the collectives (RCCL) path. |
| `MC_DISABLE_HIP_TRANSPORT` | `1` | Force RDMA, not the intra-node HIP/XGMI shortcut (cross-node). |
| `MORI_IB_GID_INDEX` | `1` | RoCEv2 GID index for the MoRIIO transport. |
| `VLLM_MORIIO_CONNECTOR_READ_MODE` | `1` | MoRIIO read/pull mode (decode reads KV from prefill). Set on both legs. |
| `VLLM_HOST_IP` | *(routable IP)* | Address a vLLM worker advertises for KV transfer; pair with `--advertise-host`. |
| `RDMAV_FORK_SAFE` | `1` | libibverbs fork-safety for RDMA workers. |

Full list on the [environment variables](../reference/environment.md) page.

## Related

- [Routing & transport](routing_and_transport.md) — how the router picks
  topology (concurrent push vs serial pull).
- [KV-Cache Offload](kv_cache_offload.md) — stack a tiered KV cache under the PD
  transport so cache-hit requests skip prefill entirely.

