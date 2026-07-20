# Infera `deploy/` — index

Container images, the operator, Helm values, manifests, and install scripts for
running Infera on Kubernetes.

## Layout

| Path | What it is |
| --- | --- |
| `docker/` | Canonical per-engine Dockerfiles (`Dockerfile.sglang`, `Dockerfile.vllm`, `Dockerfile.atom`) plus `Dockerfile.kvd` / `Dockerfile.server`, out-of-tree build patches (`patches/`: engine + Mooncake-on-ROCm), and build helper scripts (`scripts/`). |
| `operator/` | The infera-operator (Go) + its Helm chart and CRDs. See `operator/README.md`. |
| `manifests/` | Templated manifests applied with `kubectl` (e.g. `gaie-inferadeployment.yaml`). |
| `scripts/` | Install and smoke-test scripts (see below). |

Ready-to-fill `InferaDeployment` deployment templates (single-node, prefill/decode,
multi-node TP, GAIE) live in [`../examples/k8s-deployments/`](../examples/k8s-deployments/README.md).

## Scripts

| Script | What it does |
| --- | --- |
| `scripts/deploy-k8s.sh` | One-shot Helm installer for the cluster deps (LWS + NATS + operator, optional kgateway/GAIE). `deploy/scripts/deploy-k8s.sh --help` for the env knobs. |
| `scripts/gaie-smoke.sh` | End-to-end smoke test for the GAIE (Gateway API Inference Extension) path. |
| `docker/scripts/build_hipfile.sh` | Builds hipFile (AMD AI Storage) + the `ais-check` probe. Baked into `Dockerfile.vllm` by default; re-runnable on a host as `--probe-only`. |

## hipFile (GPU-direct L3)

Built into `Dockerfile.vllm` by default (`ARG BUILD_HIPFILE=1`; pass
`--build-arg BUILD_HIPFILE=0` to skip). For the end-to-end kvd + hipFile serving
recipe and tuning, see the Infera manual (`manual/`).
