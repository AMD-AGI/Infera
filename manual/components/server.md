# Server & router

`infera.server` is a FastAPI process that speaks the **OpenAI API** and holds
the **router**. It carries no model weights — just a live view of the worker fleet
(from etcd) and the logic that picks a worker per request. Run any number of
replicas behind a load balancer; they share one fleet view.

## Endpoints

### Inference (OpenAI-compatible)

| Endpoint | Use |
|---|---|
| `POST /v1/chat/completions` | chat-style requests (messages array) |
| `POST /v1/completions` | text-completion requests (raw prompt) |
| `GET /v1/models` | list the served model(s) — used by OpenAI clients and readiness checks |

These match the OpenAI schema, so any OpenAI client library works — point its
`base_url` at `http://<server>:8000/v1`.

### Anthropic Messages API

| Endpoint | Use |
|---|---|
| `POST /v1/messages` | Anthropic-compatible chat (system blocks, tool use, streaming) |

The server also speaks the **Anthropic Messages API** via a translation layer —
point an Anthropic client's `ANTHROPIC_BASE_URL` at `http://<server>:8000` and it
runs against the same workers. Text and tool-use are supported; multimodal and
extended-thinking content are not translated.

```bash
curl localhost:8000/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","max_tokens":50,
       "messages":[{"role":"user","content":"1+1=?"}]}'
```

```{admonition} PD + ATOM uses /v1/completions
:class: note
ATOM only threads the KV-transfer params on the text-completion path, so
disaggregated ATOM requests must go to `/v1/completions`. The chat path is
unaffected. See [PD disaggregation](../features/pd_disaggregation.md).
```

### Inspection & admin

| Endpoint | Use |
|---|---|
| `GET /v1/workers` | read-only dump of the worker registry |
| `GET /v1/admin/cache-view/<worker_id>` | the router's mirrored KV-cache state for one worker (KV-aware mode) |
| `POST /v1/cache/prewarm` | one-way hint to async-pull KV blocks into the warm tier ahead of a request (needs `--kvd-socket-path`) |
| `POST /v1/admin/profile/start` · `/stop` | start/stop the torch profiler on the workers (requires `--enable-profiling`; 403 otherwise) |

### Operations (health & metrics)

| Endpoint | Use |
|---|---|
| `GET /health` | liveness/readiness — returns the active-worker count (wire to k8s probes) |
| `GET /metrics` | Prometheus exposition — worker-pool + cache-view gauges (scrape with Grafana) |

## The router

Every request is routed by the server. The router is selected by two axes:

**Mode (`--router-mode`, default `auto`)**

- **`auto`** — the server selects the worker **in-process** using `--router-policy`
  (below) and the PD-preferring `AutoRouter`.
- **`direct`** — trust an upstream **GAIE Inference Gateway** Endpoint Picker (EPP):
  the server dispatches to the worker named by the `x-worker-instance-id` request
  header and skips in-process selection (falling back to the policy when the header
  is absent). This is the per-worker frontend-sidecar topology; see
  [Deployment → operator (GAIE)](operator.md).

**Policy (`--router-policy`, default `kv-aware`)** — used when mode is `auto`:

- **`kv-aware`** — route by prefix-cache locality using each worker's KV events.
  See [KV-aware routing](../features/kv_aware_routing.md).
- **`round-robin`** — stateless spread.

Independently, the `AutoRouter` picks the **topology**: it prefers a
prefill+decode pair when [PD](../features/pd_disaggregation.md) workers exist, and
falls back to a single **mixed** worker otherwise — you don't choose per request.
The full policy + transport reference is in
[Routing & transport](../features/routing_and_transport.md).

```text
client → server → router.dispatch(request, live_workers) → worker(s) → tokens → client
```

## Running multiple replicas

Every replica watches the same etcd, so they share one fleet view. Scale the
front door independently of the GPUs:

```bash
python -m infera.server --port 8000 --etcd-endpoint <etcd-host>:2379 \
  --router-tokenizer-path Qwen/Qwen3-0.6B \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq
python -m infera.server --port 8001 --etcd-endpoint <etcd-host>:2379 \
  --router-tokenizer-path Qwen/Qwen3-0.6B \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq
# ...put a load balancer in front of :8000 and :8001
```

(The three transport/discovery flags are the no-broker dev path; on the
production NATS + Kubernetes plane they're the defaults and can be dropped — see
[Routing & transport](../features/routing_and_transport.md).)

## Key server flags

| Flag | Default | What it does |
|---|---|---|
| `--port` | 8000 | listen port |
| `--etcd-endpoint` | — | `host:port` of the shared etcd |
| `--router-mode` | `auto` | `auto` (in-process policy) \| `direct` (GAIE EPP header) |
| `--router-policy` | `kv-aware` | `kv-aware` \| `round-robin` |
| `--router-tokenizer-path` | (required) | small tokenizer for prefix hashing (e.g. `Qwen/Qwen3-0.6B`) |
| `--kv-overlap-weight` | 1.0 | KV-aware: trade cache locality vs load balance |

The full flag list is in the [CLI reference](../reference/cli.md).

## Related

- [Engines](engines.md) — the workers the router dispatches to.
- [KV-aware routing](../features/kv_aware_routing.md) · [Routing & transport](../features/routing_and_transport.md) — the routing guides.
