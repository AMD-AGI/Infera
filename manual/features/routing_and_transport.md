# Routing & transport

```{admonition} One-pager
:class: tip
Four knobs govern the fleet: **how workers are discovered** (`--discovery-backend`),
**how requests reach a worker** (`--request-transport`), **how KV events flow**
(`--kv-event-transport`), and **how the router picks a worker** (`--router-mode` +
`--router-policy`). Set the discovery/transport knobs the **same** on the server
and every worker. For the server as a component, see
[Server & router](../components/server.md).
```

```{graphviz}
digraph routing_transport {
  rankdir=LR; bgcolor="transparent";
  node [shape=box style="rounded,filled" fillcolor="#eef2f7" color="#5577cc" fontname="Helvetica,Arial,sans-serif" fontsize=11 margin="0.2,0.12"];
  edge [fontname="Helvetica,Arial,sans-serif" fontsize=10];

  WK     [label="Workers"];
  S      [label="Server / Router" fillcolor="#fff3cd" color="#caa300"];
  EV     [label="Worker KV-cache events" fillcolor="#f4f4f4" color="#999999"];
  CL     [label="Client request" fillcolor="#f4f4f4" color="#999999"];
  PICK   [label="mode → policy → topology"];
  CHOSEN [label="Chosen worker(s)" fillcolor="#e2efdd" color="#6a9a4a"];

  WK -> S [label="register: --discovery-backend\n(etcd / kubernetes)" penwidth=1.6 color="#5577cc"];
  EV -> S [label="--kv-event-transport\n(nats / zmq)" style=dashed color="#8a8a8a"];
  CL -> S;
  S -> PICK [label="--router-mode / --router-policy" penwidth=1.6 color="#5577cc"];
  PICK -> CHOSEN [label="--request-transport\n(http / nats)" penwidth=1.6 color="#5577cc"];
}
```

## The knobs

| Knob | Flag (server + worker unless noted) | Options | Default (this build) |
|---|---|---|---|
| Discovery | `--discovery-backend` | `kubernetes` \| `etcd` | `kubernetes` |
| Request transport | `--request-transport` | `nats` \| `http` | `nats` |
| KV-event transport | `--kv-event-transport` | `nats` \| `zmq` | `nats` |
| Router mode (server) | `--router-mode` | `auto` \| `direct` | `auto` |
| Router policy (server) | `--router-policy` | `kv-aware` \| `round-robin` | `kv-aware` |

```{important}
The defaults (**NATS + Kubernetes**) assume a reachable NATS broker and a k8s API
— the production/operator setup, where `NATS_SERVER` is injected for you. The
[Quickstart](../getting_started/quickstart.md) uses the simpler **no-broker path**
(`etcd` + `http` + `zmq`); see [Running without a broker](#running-without-a-broker).
```

## How the router picks — mode → policy → topology

**1. Mode (`--router-mode`)** — *who* decides:

- **`auto`** (default) — the server selects the worker **in-process** via the
  policy below and the PD-preferring `AutoRouter`.
- **`direct`** — trust an upstream **GAIE Inference Gateway** Endpoint Picker
  (EPP): dispatch to the worker named by the `x-worker-instance-id` request header
  (and `x-prefill-instance-id` for the PD prefill leg), skipping in-process
  selection. Falls back to the policy when the header is absent. This is the
  per-worker frontend-sidecar topology the [operator](../components/operator.md)
  wires up with `spec.gaie`.

**2. Policy (`--router-policy`)** — *how* `auto` mode scores workers:

- **`round-robin`** — stateless even spread. Best when requests are unique/short
  (no prefix to reuse).
- **`kv-aware`** — route to the worker that already caches the prompt's prefix.
  See [KV-aware routing](kv_aware_routing.md) for the cost function and the
  `--kv-overlap-weight` dial.

**3. Topology** — mixed vs disaggregated is **automatic**: when PD workers exist,
the `AutoRouter` prefers a prefill+decode pair and the `DisaggRouter` shapes the
per-connector body (concurrent push or serial pull); otherwise it routes to a
single mixed worker. You don't select PD per request. See
[PD disaggregation](pd_disaggregation.md).

## Discovery: how the worker list is built

- **`etcd`** — workers self-register with a lease + heartbeat; the server watches
  `/infera/workers/`. No Kubernetes needed — the external/standalone path (and
  what the Quickstart uses).
- **`kubernetes`** — workers annotate their own Pod; the server watches Pods by
  `--k8s-label-selector`. Zero etcd — the operator default.

## Request transport

- **`http`** — the router forwards directly to the worker's engine HTTP. Simple,
  no broker — good for dev/CI/single box.
- **`nats`** — the router publishes each request to the worker's per-instance NATS
  subject and streams the reply back. Adds admission control and clean
  cancellation, at the cost of running a broker.

### NATS request controls

When `--request-transport nats` (flag > env var > default):

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--nats-req-idle-timeout` | `INFERA_NATS_REQ_IDLE_TIMEOUT` | `900`s | max wait for the *next* reply chunk (reset per chunk). Expiry → 504 + cancel. `0` = forever. |
| `--nats-req-max-duration` | `INFERA_NATS_REQ_MAX_DURATION` | `0` (off) | hard wall-clock cap on the whole request. Expiry → 504 + cancel. |
| `--nats-req-max-pending` | `INFERA_NATS_REQ_MAX_PENDING` | `0` (off) | per-worker admission limit; backlog at the cap → **429**. |

On timeout or client disconnect the router publishes to
`infera.cancel.<worker>` so the worker aborts the in-flight generation instead
of burning GPU.

### Request failover

`--request-max-retries` (default `1`) retries on an **alternate** mixed worker
when a request fails *before* the first response — unreachable worker, NATS error,
idle-timeout-before-first-token, or a 429 admission reject. It never retries
mid-stream (once tokens flow, a failure surfaces to the client). Raise it for more
resilience; set `0` to fail fast.

## KV-event transport

Powers [KV-aware routing](kv_aware_routing.md). `--kv-event-transport`:

- **`nats`** — one broker subscription (`infera.kv.events.>`) for the whole
  fleet; workers run a relay to forward their engine events.
- **`zmq`** — the server opens a per-worker SUB socket directly. No broker.

## Running without a broker

For **no NATS** (local dev, CI, a single box), override transport + discovery to
the direct path on **both** the server and every worker:

```bash
# server
python -m infera.server --host 0.0.0.0 --port 8000 \
  --router-tokenizer-path <model> \
  --discovery-backend etcd --etcd-endpoint 127.0.0.1:2379 \
  --request-transport http --kv-event-transport zmq

# each worker
python -m infera.engine.sglang --model-path <model> --port 30000 --host 0.0.0.0 \
  --discovery-backend etcd --etcd-endpoint 127.0.0.1:2379 \
  --request-transport http --kv-event-transport zmq
```

This is exactly the shape the [Quickstart](../getting_started/quickstart.md) and
the PD bring-up recipes use.

```{admonition} Match the flags fleet-wide
:class: important
Discovery and transport must agree across the server and all workers. A worker on
`http`/`etcd` won't be reached by a server expecting `nats`/`kubernetes`.
```

## Environment variables

Each knob is also a flag (flag > env > default); set on the **server and every
worker**.

| Env | Default | What it does |
|---|---|---|
| `NATS_SERVER` / `NATS_URL` | *(operator-injected)* | NATS broker address when `--request-transport nats`. |
| `INFERA_NATS_REQ_IDLE_TIMEOUT` | `900` (s) | Max wait for the next reply chunk; expiry → 504 + cancel. |
| `INFERA_NATS_REQ_MAX_DURATION` | `0` (off) | Hard wall-clock cap on a whole request. |
| `INFERA_NATS_REQ_MAX_PENDING` | `0` (off) | Per-worker admission limit; backlog at the cap → 429. |
| `INFERA_REQUEST_MAX_RETRIES` | `1` | Retry on an alternate worker before the first token. `0` = fail fast. |
| `INFERA_ROUTER_MODE` | `auto` | `direct` to trust a GAIE EPP's per-request worker pick. |
| `INFERA_K8S_LABEL_SELECTOR` | *(operator default)* | Pod selector for `--discovery-backend kubernetes`. |

Full list on the [environment variables](../reference/environment.md) page. The
complete flag table is in the [CLI reference](../reference/cli.md).
