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

### Decode radix cache and HiCache with speculative decoding

This experimental option requires both Infera's forwarding changes and the
SGLang engine patch. Stock v0.5.18 rejects the MTP/radix combination even if the
environment variable is set; updating only the Infera package is insufficient.
Rebuild `deploy/docker/Dockerfile.sglang`, then set
`SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1` in the Decode worker environment.

With EAGLE/NEXTN top-k 1 on a supported Mooncake Decode leg:

- KV events enable automatic Decode radix flag forwarding.
- Adding `--enable-hierarchical-cache` and the desired HiCache options enables
  the radix prerequisite before SGLang's first argument validation, so an extra
  explicit radix flag is unnecessary. This also works with KV events disabled.
- Unsupported models still fail the radix compatibility check; an explicit
  `--disable-radix-cache` is incompatible with an automatic HiCache request.

Without the runtime opt-in, speculative Decode keeps its old behavior. Neither
HiCache nor session affinity is enabled by default. HiSparse, DCP, backend and
cache-builder restrictions remain. Hybrid SWA/SSM and EAGLE3 are outside scope.
The gfx942 image is not patched by this change. Custom compatible bases can use
`APPLY_SGLANG_DECODE_RADIX_SPEC_PATCH=0` to skip the build patch.

#### Launch examples

Build from this PR's checkout:

```bash
docker build -f deploy/docker/Dockerfile.sglang \
  --build-arg APPLY_SGLANG_DECODE_RADIX_SPEC_PATCH=1 \
  -t infera-sglang:mtp-decode-cache .
```

Run the following **inside that rebuilt mi35x container**, with model files and
ROCm/RDMA devices available. These are Decode launch commands; a matching
Prefill worker, etcd and a Router using HTTP requests/ZMQ KV events must already
be running. Replace the example model path and addresses with your deployment's
values. The Prefill worker must advertise the same served model name.

Common Decode arguments (Bash):

```bash
MODEL_PATH=/models/GLM-5.2-MXFP4
DECODE_IP=10.0.0.12
ETCD_ENDPOINT=10.0.0.10:2379

export SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1

DECODE_ARGS=(
  --model-path "$MODEL_PATH"
  --served-model-name glm5.2-mxfp4
  --trust-remote-code
  --host 0.0.0.0 --port 30000
  --advertise-host "$DECODE_IP"
  --discovery-backend etcd --etcd-endpoint "$ETCD_ENDPOINT"
  --request-transport http
  --enable-kv-events --kv-events on --kv-event-transport zmq
  --disaggregation-mode decode
  --disaggregation-transfer-backend mooncake
  --tp-size 8 --dp-size 8 --enable-dp-attention
  --speculative-algorithm EAGLE
  --speculative-eagle-topk 1
  --speculative-num-steps 5 --speculative-num-draft-tokens 6
  --mem-fraction-static 0.85
  --enable-metrics
  --json-model-override-args '{"index_share_for_mtp_iteration":false}'
)
```

Choose one of the two modes below.

**Decode radix + MTP, without HiCache:**

```bash
python -m infera.engine.sglang "${DECODE_ARGS[@]}"
```

KV events cause Infera to append the Decode radix flag. The patched SGLang hook
accepts the explicit MTP opt-in for EAGLE/NEXTN top-k 1.

**Decode radix + HiCache + MTP:**

```bash
python -m infera.engine.sglang "${DECODE_ARGS[@]}" \
  --enable-hierarchical-cache \
  --hicache-ratio 1.5 \
  --hicache-write-policy write_through \
  --hicache-io-backend kernel \
  --hicache-mem-layout page_first
```

HiCache's required radix flag is set **before** `ServerArgs.from_cli_args`
performs cache compatibility checks, and is also forwarded to the engine.
Users do not need to add `--disaggregation-decode-enable-radix-cache` manually.
An explicit HiCache request implies this prerequisite even with KV events off;
this does not turn KV-event publication back on. Explicit radix flags remain
supported and are not duplicated.

The HiCache settings above reproduce the earlier configuration; size the host
pool for the node's memory. They are not a performance recommendation: the
historical C40 `write_through` test reduced total-token throughput by about 4.5%
and increased mean ITL by about 6.2% compared with Decode radix alone.

An old benchmark harness may still reject `DECODE_MTP=1` plus `DECODE_HICACHE=1`
before invoking Infera. These direct engine commands do not pass through that
harness guard; updating such scripts is separate from engine support.


The commands above document the intended launch configuration; they have not
been GPU-validated against the rebuilt default v0.5.18 image. Historical runtime
validation used the patched v0.5.19 build. Argument/patch tests do not replace
GPU prefix-reuse and real-acceptance accuracy checks.
