# Installation

Infera is a Python package — installed as `amd-infera`, imported as `infera`.
A router-only host needs Infera but no local inference engine or GPU. Worker hosts
need **Infera** plus a compatible **engine** (vLLM, SGLang, or ATOM), usually
provided together in an engine image. The system requirements below describe worker hosts.

## System requirements

| Component | Requirement |
|---|---|
| GPU | AMD Instinct **MI355X** (gfx950) |
| ROCm | [7.2+](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html) |
| OS | Linux x86-64 (validated on Ubuntu 24.04) |
| Python | 3.10+ |
| Docker | required — used to run etcd in dev and to build engine images |
| Discovery | etcd (dev) or the Kubernetes API (production) — a one-line `docker run` gets etcd up for dev, see the [Quickstart](quickstart.md) |
| RDMA NIC | AMD AINIC — **only needed for cross-node PD** |
| Engine | at least one of vLLM / SGLang / ATOM (below) |

## Install Infera

```bash
pip install amd-infera
```

That gives you the three entry points used throughout this manual:

| Command | What it is |
|---|---|
| `python -m infera.server` | the OpenAI-compatible server + router |
| `python -m infera.engine.{vllm,sglang,atom}` | a model worker |
| `python -m infera.kvd` | the tiered KV-cache daemon (optional) |

Use the module commands above for services. The package also installs operational
commands including `infera-preflight`, `infera-kvd-probe`, `infera-kvd-l3-bench`, and
`inferasim`; run each with `--help` for usage. Simulation needs the `projection` extra.

The Linux x86_64 wheel (glibc ≥ 2.28) includes the Rust router. Select it with
`python -m infera.server --router-backend rust`; the default is `python`.
The universal wheel does not include the Rust binary. Source installs can build
it with `cd rust && cargo build --release --bin infera-router`, or use
`INFERA_ROUTER_BIN` to point at an existing binary.

## Install an engine

Pick whichever you'll serve with (you can install more than one):

- **vLLM (ROCm)** — `vllm/vllm-openai-rocm:nightly-cbe9c40f…` is the validated
  base image (digest-pinned nightly), or a matching ROCm `pip` build.
- **SGLang (ROCm)** — use `lmsysorg/sglang:v0.5.18-rocm720-mi35x` on
  MI355X and `lmsysorg/sglang:v0.5.16-rocm720-mi30x` on MI325X.
- **ATOM** — `rocm/atom:rocm7.2.4_...atom0.1.4`.

In practice most people don't `pip install` the engine by hand — they use a
prebuilt **engine image** that already layers the Infera connector, the
`sitecustomize` hook, and the Mooncake/ionic RDMA shims on top of the vendor
base. See [Deployment → Engine images](../serving/deployment.md#engine-images).

## Docker

For serving (rather than developing locally) the container path is usually
easier: the **engine images** already bundle Infera, the `sitecustomize` hook, and
the Mooncake / ionic RDMA shims on top of the vendor ROCm base — nothing to
`pip install` by hand.

Build the engine image for the runtime you'll serve with, from the repo root
(all three engine Dockerfiles live under `deploy/docker/`):

```bash
# vLLM
docker build -f deploy/docker/Dockerfile.vllm \
  -t rocm/infera:vllm-dev .

# SGLang
docker build -f deploy/docker/Dockerfile.sglang \
  -t rocm/infera:sglang-dev .

# ATOM
docker build -f deploy/docker/Dockerfile.atom \
  -t rocm/infera:atom-dev .
```

Then bring up a full stack (etcd + server + engine). The per-runtime
Dockerfiles, Kubernetes manifests, and the manual (bare-metal) recipe —
including KV-aware routing, the kvd daemon, and PD disaggregation — are
all covered in [Deployment](../serving/deployment.md).

## Verify

```bash
python -c "import infera; print('infera OK')"
python -m infera.server --help | head -5
```

Next: the [Quickstart](quickstart.md) brings up a real stack and serves a model.
