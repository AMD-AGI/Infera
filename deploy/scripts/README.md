# `deploy-k8s.sh` — Infera Kubernetes installer

One-shot, idempotent Helm installer for the Infera cluster dependencies. Each
step is a `helm upgrade --install` (or `kubectl apply` for CRDs), so re-running is
safe and you can install only what you need.

## What it installs

| Step | Component | Default | Notes |
|------|-----------|---------|-------|
| 1 | **LeaderWorkerSet (LWS)** controller | on | Required for multi-node TP workers (`numberOfNodes > 1`). |
| 2 | **NATS (JetStream)** | on | Shared KV-event + request broker. Skip if each `InferaDeployment` runs its own (`spec.nats.deploy: true`). |
| 3 | **infera-operator** | on | Reconciles `InferaDeployment` CRs; ships the CRD + RBAC. |
| 4 | **Inference Gateway (kgateway + GAIE CRDs)** | off | Gateway API + GAIE CRDs + kgateway control/data plane + optional Gateway. |

This installs only platform/infrastructure — **no inference workload**. To run a
model, apply an `InferaDeployment` afterwards (see `examples/k8s-deployments/`).

## Prerequisites (NOT installed by this script)

- **AMD GPU device plugin** — GPUs must be schedulable as `amd.com/gpu`.
- **Engine image** — the `infera-sglang|vllm` image referenced by your IDEP must be pullable by the nodes.
- **Model data** — reachable inside the pod (PVC / hostPath / HF download).
- **imagePullSecret** — for a private registry, per-namespace where you apply the IDEP.
- `helm` and `kubectl` in `PATH`.

## Quick start

```bash
# Base install (LWS + NATS + operator) into infera-system
deploy/scripts/deploy-k8s.sh

# Cluster without a default StorageClass: pin the JetStream PVC
NATS_STORAGE_CLASS=local-path deploy/scripts/deploy-k8s.sh

# Only the operator (skip LWS + NATS)
INSTALL_LWS=false INSTALL_NATS=false deploy/scripts/deploy-k8s.sh

# Add the inference gateway (kgateway + GAIE CRDs)
INSTALL_GATEWAY=true GATEWAY_NS=infera deploy/scripts/deploy-k8s.sh

deploy/scripts/deploy-k8s.sh --help     # inline usage
```

## Environment variables

### Step toggles

| Var | Default | Meaning |
|-----|---------|---------|
| `INSTALL_LWS` | `true` | Install the LeaderWorkerSet controller. |
| `INSTALL_NATS` | `true` | Install the shared NATS/JetStream broker. |
| `INSTALL_OPERATOR` | `true` | Install the infera-operator. |
| `INSTALL_GATEWAY` | `false` | Install the kgateway + GAIE CRD stack (step 4). |

### General

| Var | Default | Meaning |
|-----|---------|---------|
| `INFERA_NAMESPACE` | `infera-system` | Namespace for the operator + shared NATS. |
| `LWS_NAMESPACE` | `lws-system` | Namespace for the LWS controller. |
| `HELM_TIMEOUT` | `5m` | `--wait --timeout` per Helm step. |

### LeaderWorkerSet (step 1)

| Var | Default | Meaning |
|-----|---------|---------|
| `LWS_CHART` | `oci://registry.k8s.io/lws/charts/lws` | LWS Helm chart (OCI). |
| `LWS_VERSION` | `0.9.0` | Chart version (tested). |
| `LWS_RELEASE` | `lws` | Helm release name. |

### NATS / JetStream (step 2)

| Var | Default | Meaning |
|-----|---------|---------|
| `NATS_REPO_NAME` | `nats` | Helm repo alias. |
| `NATS_REPO_URL` | `https://nats-io.github.io/k8s/helm/charts/` | Helm repo URL. |
| `NATS_CHART` | `nats/nats` | Chart ref. |
| `NATS_VERSION` | *(empty)* | Chart version; empty = latest. |
| `NATS_RELEASE` | `infera-nats` | Helm release name (also the Service host). |
| `NATS_STORAGE_SIZE` | `10Gi` | JetStream PVC size (`--set`). |
| `NATS_STORAGE_CLASS` | *(empty)* | JetStream PVC StorageClass (`--set` when set). **Empty = cluster default SC; set to e.g. `local-path` on clusters without a default SC, else the PVC stays Pending.** |

JetStream config (enabled, file store, single replica) is passed inline via
`--set` — there is no local values file to edit.

Shared NATS address after install: `nats://<NATS_RELEASE>.<INFERA_NAMESPACE>.svc:4222`
(point IDEPs at it with `spec.nats.deploy: false` + `NATS_SERVER`).

### infera-operator (step 3)

| Var | Default | Meaning |
|-----|---------|---------|
| `OPERATOR_RELEASE` | `infera-operator` | Helm release name. |
| `OPERATOR_CHART` | `deploy/operator/helm/infera-operator` | Local chart path. |
| `OPERATOR_IMAGE_REPO` | *(empty)* | Override image repo (`--set`; empty = chart default). |
| `OPERATOR_IMAGE_TAG` | *(empty)* | Override image tag (`--set`; empty = chart default). |

### Inference Gateway (step 4, `INSTALL_GATEWAY=true`)

Step 4 has three distinct concerns, split into three variable groups:

- **`GATEWAY_API_*` / `GAIE_CRD_*`** — the vendor-neutral **standards** (the k8s
  Gateway API CRDs and the GAIE inference-extension CRDs). Any gateway
  implementation uses these; they are cluster-wide shared CRDs.
- **`KGATEWAY_*`** — the specific **implementation** installed here (the kgateway
  controller = data plane + control plane). Swap these out to use a different
  Gateway API implementation (Istio, Envoy Gateway, …).
- **`GATEWAY_*` (name/ns/create)** — the actual **`Gateway` object** you create
  (a concrete entry point of `gatewayClassName: kgateway`).

#### Standards — Gateway API + GAIE CRDs (shared, vendor-neutral)

| Var | Default | Meaning |
|-----|---------|---------|
| `GATEWAY_API_VERSION` | `v1.4.1` | k8s Gateway API CRDs version. **Shared CRD — upgrading may affect other gateway implementations already on the cluster (e.g. Istio / higress).** |
| `GATEWAY_API_MANIFEST` | *(derived from version)* | Gateway API CRD manifest URL / file (air-gapped: use `file://`). |
| `GAIE_CRD_VERSION` | `v1.5.0` | GAIE inference-extension CRDs (tested). |
| `GAIE_CRD_MANIFEST` | *(derived from version)* | GAIE CRD manifest URL / file. |

#### Implementation — kgateway controller

| Var | Default | Meaning |
|-----|---------|---------|
| `KGATEWAY_NAMESPACE` | `kgateway-system` | Namespace for the kgateway controller. |
| `KGATEWAY_VERSION` | `v2.1.1` | kgateway chart version. |
| `KGATEWAY_CRDS_CHART` | `oci://cr.kgateway.dev/kgateway-dev/charts/kgateway-crds` | kgateway CRD chart; override to a mirror. |
| `KGATEWAY_CHART` | `oci://cr.kgateway.dev/kgateway-dev/charts/kgateway` | kgateway controller chart; override to a mirror. |
| `KGATEWAY_IMAGE_REGISTRY` | *(empty)* | Pull controller + data-plane images from a mirror prefix. |

#### Gateway object — the entry point created for you

| Var | Default | Meaning |
|-----|---------|---------|
| `CREATE_GATEWAY` | `true` | Also create a `Gateway` resource (set false to bring your own). |
| `GATEWAY_NAME` | `inference-gateway` | Name of the `Gateway` object. |
| `GATEWAY_NS` | `infera` | Namespace of the `Gateway` object (usually where your inference service runs). |
| `GATEWAY_MANIFEST` | *(derived from version)* | Sample Gateway manifest URL / file. |

> Deploying a GAIE **workload** is out of scope for this script — apply a
> GAIE-enabled `InferaDeployment` yourself (see
> `examples/k8s-deployments/kgateway-gaie-aggregated.yaml`, or render
> `deploy/manifests/gaie-inferadeployment.yaml` with `envsubst`).

## Verify

```bash
kubectl get crd inferadeployments.infera.amd.com
kubectl -n infera-system get deploy,pod            # operator (+ NATS) Running
kubectl -n lws-system get pod                       # LWS controller (if installed)
# GAIE (if installed):
kubectl get crd inferencepools.inference.networking.k8s.io
kubectl -n kgateway-system get deploy,pod
```
