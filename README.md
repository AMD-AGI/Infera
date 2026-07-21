# ROCm Infera: More token goodput from frontier models.

[![CI](https://github.com/AMD-AGI/Infera/actions/workflows/ci.yml/badge.svg)](https://github.com/AMD-AGI/Infera/actions/workflows/ci.yml)
[![Release](https://github.com/AMD-AGI/Infera/actions/workflows/release.yml/badge.svg)](https://github.com/AMD-AGI/Infera/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[What's Infera?](#whats-infera) | [Key features](#key-features) | [Quick Start](#quick-start) | [Engine images](#engine-images) | [Documentation](#documentation) | [License](#license)

A distributed, production serving mesh — disaggregated prefill/decode, KV-aware routing, and
cache offload, tuned to your production SLA.

Infera is open and ROCm-native, built for AMD Instinct™ GPUs. It presents a single
**OpenAI-compatible** endpoint — and the **Anthropic Messages** API through a translation
layer — in front of one or many model workers, and runs **vLLM, SGLang, or ATOM** underneath.

## What's Infera?

Frontier models are served across nodes; inference engines are optimized within one. vLLM,
SGLang, and ATOM batch, schedule, and manage memory well on a single node — but coordinating a
fleet of them is left to you: hundreds of gigabytes of KV cache to track, agentic clients that
replay the same history on every step, and requests to place across many GPUs without throwing
away work the fleet has already done.

Infera does **not** replace those engines. It is the layer in front of them that turns per-node
engines into one fleet behind a single API, so that more of the GPU time you pay for becomes
tokens you keep rather than prefill you recompute. Built and validated on AMD Instinct MI355X.

## Key features

The three levers that turn GPU time into tokens:

- **KV-aware routing** — score every live worker by how much of the prompt's prefix it already holds and route to the best match, keeping the fleet balanced. The prefix is served from cache instead of being recomputed.
- **Prefill/decode disaggregation** — prefill is compute-bound, decode is bandwidth-bound, and running both on the same GPUs underuses each. Run them on separate GPUs/nodes, size each pool for its own job, and stream the KV between them over Mooncake or MoRI (AI NIC RDMA).
- **Cache offload — tiered KV cache (`kvd`)** — when HBM fills, keep KV warm instead of dropping it and paying to recompute: GPU HBM → host RAM → local NVMe (L3) *or* a distributed store (L4). Durable across restarts, shared across engines on a host, and — on the distributed tier — pooled across nodes, so one node can pick up a prefix another already computed.

Around them:

- **Multi-engine** — run vLLM, SGLang, or ATOM behind one common serving interface.
- **OpenAI- and Anthropic-compatible API** — `/v1/chat/completions`, `/v1/completions`, and `/v1/messages` (Anthropic Messages, translated in-process).
- **Self-registering fleet** — workers register into etcd and heartbeat, so the router works from a live view and never routes to a worker that is gone; run any number of stateless server replicas.
- **Kubernetes-native** — an operator reconciles an `InferaDeployment` CRD (aggregated / PD / multi-node), with an optional Gateway API (GAIE) endpoint picker.

## Quick Start

### Option A — Docker (fastest)

The engine images bundle Infera and the engine, so no host-side installation is required. Start
etcd, then run the server and a worker in an engine container.

```bash
# Step 1 — etcd (shared registry)
docker run -d --name infera-etcd --network host quay.io/coreos/etcd:v3.5.14 \
  etcd --advertise-client-urls http://127.0.0.1:2379 --listen-client-urls http://0.0.0.0:2379

# Step 2 — engine container (drop infiniband/IPC_LOCK/libionic for a single-host, non-RDMA run)
docker run --rm -it --network host --ipc host --shm-size 32g             \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband           \
  --group-add video --group-add render --cap-add IPC_LOCK                \
  -v /usr/lib/x86_64-linux-gnu/libionic.so:/host-libionic/libionic.so:ro \
  docker.io/rocm/infera-sglang:v0.1.0 bash

# Step 3 — inside the container: start the server and a worker
python -m infera.server --port 8000 --etcd-endpoint 127.0.0.1:2379 \
  --router-tokenizer-path Qwen/Qwen3-0.6B \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq &

python -m infera.engine.sglang --model-path Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 30000 --etcd-endpoint 127.0.0.1:2379 \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq &

# Step 4 — send a request (allow a few seconds for the worker to load the model)
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":50}'
```

To build the image yourself, see [Engine images](#engine-images) and replace
`rocm/infera-sglang:v0.1.0` above with your local `infera-sglang:dev` tag.

### Option B — From source (pip)

> The PyPI package (`amd-infera`) is not yet published; install from a clone.

In a ROCm environment (e.g. an engine base image), install Infera plus at least one engine, then
run the same etcd + server + worker stack as in Option A:

```bash
pip3 install ".[sglang]"        # or .[vllm] / .[atom]  (add .[gaie] for the EndpointPicker)
pip3 install -e ".[dev,sglang]" # editable + lint/test tooling, for contributors
```

**Optional — Rust router** (`--router-backend rust`), the multi-core data plane. It is not yet
shipped as a wheel, so build the binary from source; this requires a
[Rust toolchain](https://rustup.rs):

```bash
cd rust && cargo build --release && cd -   # produces rust/target/release/infera-router
```

The server auto-discovers the binary (or set `INFERA_ROUTER_BIN=/path/to/infera-router`); then
run the server with `--router-backend rust`. The default backend is `python`, which requires no
Rust toolchain.

### Option C — Kubernetes

The operator reconciles an `InferaDeployment` CRD (aggregated / prefill-decode / multi-node)
with an optional Gateway API (GAIE) endpoint picker. Install the cluster dependencies (LWS +
NATS + operator; add `INSTALL_GATEWAY=true` for the GAIE stack), then apply a deployment:

```bash
deploy/scripts/deploy-k8s.sh            # LWS + NATS + operator (see --help for options)

# Edit the <PLACEHOLDERS> in these templates for your cluster before applying.
kubectl apply -f examples/k8s-deployments/pvc-models.yaml
kubectl apply -f examples/k8s-deployments/single-node-aggregated.yaml
```

Ready-to-fill deployment templates (single-node, prefill/decode, multi-node TP, GAIE) and their
placeholders are in [`examples/k8s-deployments/`](examples/k8s-deployments/README.md).

## Engine images

Each engine has one canonical Dockerfile under `deploy/docker/` that overlays Infera on a
vendor base. Build from the repo root; override the base with `--build-arg <ENGINE>_BASE_IMAGE=...`.

| Dockerfile          | Base image                                      |
|---------------------|-------------------------------------------------|
| `Dockerfile.sglang` | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`   |
| `Dockerfile.vllm`   | `vllm/vllm-openai-rocm:nightly-cbe9c40f…`       |
| `Dockerfile.atom`   | `rocm/atom:rocm7.2.4_…_atom0.1.4_20260612`      |

```bash
docker build -f deploy/docker/Dockerfile.sglang -t infera-sglang:dev .
docker build -f deploy/docker/Dockerfile.vllm   -t infera-vllm:dev .
docker build -f deploy/docker/Dockerfile.atom   -t infera-atom:dev .
```

## Benchmarks

**Kimi agentic benchmark** — a long-context, multi-turn coding-agent workload
(Kimi-K2.6-MXFP4 on MI355X) that compares prefill/decode disaggregation against a
single-node TP8 baseline at a fixed per-user interactivity SLA. Workload definition,
topologies, launch scripts, the concurrency-sweep harness, results, and reproduction
steps are in [`examples/kimi_agentic_bench/`](examples/kimi_agentic_bench/README.md).

## Documentation

Full guides, deployment recipes, and reference live in the Sphinx manual:

```bash
sudo apt-get install -y graphviz                            # `dot`, for the diagrams
cd manual && pip install -r sphinx/requirements.txt && make html   # open _build/html/index.html
```

## License

MIT — see [LICENSE](LICENSE). © 2026 Advanced Micro Devices, Inc.
