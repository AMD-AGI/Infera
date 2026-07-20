# Installation

Infera is a Python package — installed as `amd-infera`, imported as `infera`.
It orchestrates engines — so you install **Infera** plus at least **one engine**
(vLLM and/or SGLang and/or ATOM) in the same environment.

## System requirements

| Component | Requirement |
|---|---|
| GPU | AMD Instinct **MI355X** (gfx950) |
| ROCm | 7.2+ |
| OS | Linux x86-64 (validated on Ubuntu 24.04) |
| Python | 3.10+ |
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

```{admonition} There are no console scripts in this build
:class: note
Always invoke the canonical module form `python -m infera.<thing>`. If you see
`infera-server` or `infera-kvd` named anywhere, read it as
`python -m infera.server` / `python -m infera.kvd`.
```

## Install an engine

Pick whichever you'll serve with (you can install more than one):

- **vLLM (ROCm)** — `vllm/vllm-openai-rocm:nightly-cbe9c40f…` is the validated
  base image (digest-pinned nightly), or a matching ROCm `pip` build.
- **SGLang (ROCm)** — `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` is
  the validated base image.
- **ATOM** — `rocm/atom:rocm7.2.4_...atom0.1.4`.

In practice most people don't `pip install` the engine by hand — they use a
prebuilt **engine image** that already layers the Infera connector, the
`sitecustomize` hook, and the Mooncake/ionic RDMA shims on top of the vendor
base. See [Deployment → Engine images](../serving/deployment.md#engine-images).

## Docker

For serving (rather than hacking on the code) the container path is usually
easier: the **engine images** already bundle Infera, the `sitecustomize` hook, and
the Mooncake / ionic RDMA shims on top of the vendor ROCm base — nothing to
`pip install` by hand.

Build the engine image for the runtime you'll serve with, from the repo root
(all three engine Dockerfiles live under `deploy/docker/`):

```bash
# vLLM
docker build -f deploy/docker/Dockerfile.vllm \
  -t infera/engine-vllm:dev .

# SGLang
docker build -f deploy/docker/Dockerfile.sglang \
  -t infera/engine-sglang:dev .

# ATOM
docker build -f deploy/docker/Dockerfile.atom \
  -t infera/engine-atom:dev .
```

Then bring up a full stack (etcd + server + engine). The per-runtime
Dockerfiles, Kubernetes manifests, and the manual (bare-metal) recipe —
including KV-aware routing, the kvd cache daemon, and PD disaggregation — are
all covered in [Deployment](../serving/deployment.md).

## Verify

```bash
python -c "import infera; print('infera OK')"
python -m infera.server --help | head -5
```

Next: the [Quickstart](quickstart.md) brings up a real stack and serves a model.
