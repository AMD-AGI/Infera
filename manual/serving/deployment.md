# Deployment

Two ways to run a fleet, smallest to largest:

| Target | How | Use when |
|---|---|---|
| **Manual** | `python -m infera.*` by hand | dev, benches, a small no-orchestrator fleet |
| **Kubernetes** | the operator + `InferaDeployment` CRD | multi-host, rolling upgrades, autoscale |

Both paths use the same images under `deploy/docker/`.

## Manual (bare-metal)

No images, no orchestrator — the `python -m infera.*` path (the same one the
[Quickstart](../getting_started/quickstart.md) uses), framed for a small standing
fleet. Start etcd, one or more servers, then the workers; every process points at
the **same** `--etcd-endpoint` and shares the no-broker dev plane
(`--discovery-backend etcd --request-transport http --kv-event-transport zmq`).

```bash
# 1. etcd — the shared registry
docker run -d --name infera-etcd --net host quay.io/coreos/etcd:v3.5.14 \
  etcd --advertise-client-urls http://<host>:2379 \
       --listen-client-urls http://0.0.0.0:2379

# 2. server(s) — the router/frontend on :8000; put a load balancer in front for HA
python -m infera.server --host 0.0.0.0 --port 8000 \
  --etcd-endpoint <host>:2379 --router-tokenizer-path <model> \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq

# 3. workers — one per GPU, each self-registers (repeat on any host/GPU)
HIP_VISIBLE_DEVICES=0 python -m infera.engine.vllm \
  --model <model> --port 30000 --host 0.0.0.0 --advertise-host <host> \
  --etcd-endpoint <host>:2379 \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq
```

Scale by launching more workers (on any host) against the same etcd; stop one and
its lease expires so it drops out of the fleet. On the production **NATS +
Kubernetes** plane those three dev flags are the defaults — drop them and see
[Routing & transport](../features/routing_and_transport.md).

## Kubernetes

The production path — install the platform with Helm, submit an
`InferaDeployment` CRD (aggregated, PD, or multi-node via the operator),
monitor, and send a request — has its own guide:
**[Kubernetes deployment](kubernetes.md)**. Ready-to-fill CR templates live in
`examples/k8s-deployments/`, and the CRD field reference is on the
[Operator](../components/operator.md) page.

## Engine images

Engine images overlay the Infera connector + `sitecustomize` hook +
Mooncake/ionic RDMA shims on top of a vendor base. Pick by runtime:

| Dockerfile (under `deploy/docker/`) | Base | Use for |
|---|---|---|
| `Dockerfile.vllm` | `vllm/vllm-openai-rocm:nightly-cbe9c40f…` | vLLM on MI355X (incl. hipFile) |
| `Dockerfile.sglang` | `lmsysorg/sglang:v0.5.18-rocm720-mi35x` | SGLang on MI355X |
| `Dockerfile.sglang.gfx942` | `lmsysorg/sglang:v0.5.16-rocm720-mi30x` | SGLang on MI325X |
| `Dockerfile.atom` | `rocm/atom:rocm7.2.4_…atom0.1.4` | ATOM on ROCm |

Build from the repo root, e.g.:

```bash
docker build -f deploy/docker/Dockerfile.sglang \
  -t rocm/infera:sglang-dev .
```

```{admonition} Pin the SGLang base image
:class: important
The SGLang base is tied to a specific ROCm + build-date tag and is selected via
the `SGLANG_BASE_IMAGE` build-arg — override it to match the tag you've validated
rather than relying on the default. vLLM and ATOM each use a single base image.
```

```{admonition} hipFile on upstream rocm/sgl-dev
:class: note
The official `rocm/sgl-dev` tags strip the hipFile stack (no `libhipfile.so`).
The rocm720 Dockerfile rebuilds it from source so the async-read patch has
something to patch. Run `ais-check` inside the container on an MI355X
host to confirm the kernel exposes P2PDMA — otherwise hipFile falls back to a
CPU bounce (still works, just slower).
```

### Optional session affinity in the Rust router

Set `INFERA_SESSION_AFFINITY` on the Rust router:

| Value | Behavior |
|---|---|
| `off` (default) | Ignore session affinity and use the configured routing policy. |
| `prefill` | Keep each session on its selected Prefill worker/rank; choose Decode normally. |
| `both` | Keep independent Prefill and Decode targets; also pin aggregated workers. |

Clients opt in per request with `X-Dynamo-Session-ID`. The ID is scoped by model
and role; it is not added to model input. Requests without the header keep
normal routing. Enabled modes reject duplicate, empty or over-1024-byte IDs.
Prefill and Decode rank numbers need not match.

`INFERA_SESSION_AFFINITY_TTL_SECS` is the idle lifetime (default 3600, allowed
1–86400 seconds). Active request leases prevent idle expiry. A changed or
unavailable target is reselected; failed requests and failed upstream streams
invalidate the affected binding. An old failure callback cannot delete a newer
binding. Bound requests still perform cache lookup and normal load accounting.

Bindings live in one Router process, with a limit of 65,536 model/session/role
entries. At capacity, new entries fall back to normal routing; active bindings
are not evicted. Restarting the router loses bindings. Multiple router replicas
need their own ingress affinity if cross-request consistency is required.

This option changes placement, not cache allocation. Decode affinity is most
useful when the engine can retain and reuse Decode prefixes, but does not itself
enable radix cache or guarantee better throughput. The Python router is
unchanged. Session active-lease, hit and selection counters are exposed through
`infera_router_session_*` metrics, with role labels and no session-ID labels.
