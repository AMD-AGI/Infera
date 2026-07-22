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
| `Dockerfile.sglang` | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` | SGLang on ROCm |
| `Dockerfile.atom` | `rocm/atom:rocm7.2.4_…atom0.1.4` | ATOM on ROCm |

Build from the repo root, e.g.:

```bash
docker build -f deploy/docker/Dockerfile.sglang \
  -t infera/engine-sglang:rocm720-mi35x .
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
