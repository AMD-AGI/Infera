# Engines

A **worker** is one model engine on one (or a few) GPUs. Infera supports three,
all launched the same way (`python -m infera.engine.<name>`) and all
self-registering into etcd. The router treats them uniformly.

## Which engine?

| Engine | Launch module | Strengths | Pick it when |
|---|---|---|---|
| **SGLang** | `infera.engine.sglang` | DP-attention, strong high-concurrency decode, native disagg bootstrap | you want maximum decode throughput at high concurrency |
| **vLLM** | `infera.engine.vllm` | broad model coverage, Mooncake/MoRI KV connectors, mature | you need a model SGLang doesn't run, or Mooncake/MoRI PD |
| **ATOM** | `infera.engine.atom` | AMD's first-party engine; fp8 + mxfp4 on Instinct | you're standardizing on AMD's own stack, or it's the validated engine for your checkpoint |

Run a **single** engine type per model: a PD pair must be the same engine —
prefill and decode both vLLM, both SGLang, or both ATOM. Infera does not mix
engine types (e.g. SGLang prefill with a vLLM or ATOM decode) within one serving
stack.

```{admonition} ATOM integrates more narrowly today
:class: note
ATOM self-registers via **etcd only** — it has no `--discovery-backend` /
`--request-transport` / `--kv-event-transport` knobs (those are vLLM/SGLang-only),
so it runs on the dev/etcd plane rather than the NATS + Kubernetes production
plane. Aggregated / MIXED serving is the well-trodden path; PD uses the mooncake
connector (`atom-mooncake`).
```

## Launching a worker

The minimum is a model, a port, and the etcd endpoint. The examples below also
assume the no-broker dev flags `--discovery-backend etcd --request-transport
http --kv-event-transport zmq` (see the
[Quickstart](../getting_started/quickstart.md)); on
the production NATS + Kubernetes plane those are the defaults.

::::{tab-set}

:::{tab-item} SGLang
```bash
HIP_VISIBLE_DEVICES=0 python -m infera.engine.sglang \
  --model-path <model> --port 30000 --host 0.0.0.0 \
  --etcd-endpoint <etcd-host>:2379
```
Multi-GPU: add `--tp-size N` and expose N GPUs in `HIP_VISIBLE_DEVICES`.
:::

:::{tab-item} vLLM
```bash
HIP_VISIBLE_DEVICES=0 python -m infera.engine.vllm \
  --model <model> --port 30000 --host 0.0.0.0 \
  --etcd-endpoint <etcd-host>:2379
```
Multi-GPU: add `--tensor-parallel-size N`.
:::

:::{tab-item} ATOM
```bash
HIP_VISIBLE_DEVICES=0 python -m infera.engine.atom \
  --model <model> --server-port 30000 --host 0.0.0.0 \
  -tp 1 --etcd-endpoint <etcd-host>:2379
```
:::

::::

```{admonition} Cross-node? Advertise a routable address
:class: important
On a single host the defaults are fine. Across hosts, every worker must
advertise an address the server (and, for PD, the peer worker) can actually
reach: pass `--advertise-host <routable-ip>` and point all workers + servers at
a **shared** etcd (`--etcd-endpoint <reachable-host>:2379`).
```

## Enabling features at the worker

Most Infera features are turned on by **flags on the worker** plus a matching
**policy on the server**:

| Feature | Worker flags | Server side |
|---|---|---|
| [KV-aware routing](../features/kv_aware_routing.md) | `--enable-kv-events` + `--page-size 16` (SGLang) / `--block-size 16` (vLLM) | `--router-policy kv-aware` |
| [PD disaggregation](../features/pd_disaggregation.md) | `--disaggregation-mode prefill\|decode` (SGLang) or `--kv-transfer-config` (vLLM/ATOM) | (auto) |
| [KV-Cache Offload](../features/kv_cache_offload.md) | engine ↔ kvd wiring (see that page) | — |

## Parallelism: TP, DP-attention, external-LB DP

**Tensor parallel (TP)** — one worker spans N GPUs: `--tp-size N` (SGLang) /
`--tensor-parallel-size N` (vLLM). Expose N GPUs in `HIP_VISIBLE_DEVICES`.

### DP-attention (SGLang)

Attention runs **data-parallel** across N GPUs behind **one** endpoint — the
flagship high-concurrency-decode configuration. Turn it on at the worker with
`--dp-size N --enable-dp-attention`:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m infera.engine.sglang \
  --model-path <model> --port 30000 --host 0.0.0.0 \
  --etcd-endpoint <etcd>:2379 \
  --tp-size 8 --dp-size 8 --enable-dp-attention
```

For an **MoE** model, pair it with expert-parallel + the DP LM head (and the MoRI
all-to-all): `--ep-size 8 --moe-dense-tp-size 1 --enable-dp-lm-head
--moe-a2a-backend mori --deepep-mode normal`.

infera registers the worker as **`dp_size=N` (rank-multiplexed)** — one endpoint
fronting N attention-DP ranks — and the router steers each request to a rank via
the `X-Data-Parallel-Rank` header. In PD this is typically the **decode** engine
(prefill stays plain-TP); see [PD disaggregation](../features/pd_disaggregation.md).

### External-LB DP (vLLM)

vLLM DP isn't rank-multiplexed — run **one launcher per rank**, each its own
worker on its own port:

```bash
--data-parallel-size N --data-parallel-rank K --data-parallel-external-lb   # K = 0 … N-1
```

Each rank self-registers into etcd; the router spreads across them by prefix. See
the [project README](https://github.com/AMD-AGI/Optimus#data-parallel-replicas).

## Container images

For production you run engines from prebuilt images that overlay the Infera
connector + RDMA shims onto a vendor base (the SGLang base is pinned via the
`SGLANG_BASE_IMAGE` build-arg). See
[Deployment → Engine images](../serving/deployment.md#engine-images).
