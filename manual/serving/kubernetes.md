# Kubernetes deployment

Infera's production path is Kubernetes-native: install the platform (operator +
NATS + LeaderWorkerSet controller) with Helm, submit an **`InferaDeployment`**
CRD, and let the operator reconcile it into Deployments / LeaderWorkerSets,
Services, and the KV-event plane. Ready-to-fill `InferaDeployment` templates
(single-node, prefill/decode, multi-node TP, GAIE) live under
[`examples/k8s-deployments/`](../../examples/k8s-deployments/README.md).

## Prerequisites

- A Kubernetes cluster (v1.24+) with **≥ 1 ROCm node**, and `kubectl` + `helm`.
- The **AMD GPU device plugin**, so GPUs are schedulable as `amd.com/gpu`:
  ```bash
  kubectl describe node | grep amd.com/gpu     # Capacity should be > 0
  ```
  This is the hard prerequisite. **etcd is not needed** — the operator's default
  discovery is the in-cluster API.
- Your **engine image** (`infera/engine-{sglang,vllm,atom}`) pullable by the
  nodes; a per-namespace `imagePullSecret` for a private registry.
- **Model data** reachable inside the pod (a `hostPath`/PVC via `extraPodSpec`,
  baked into the image, or an HF download with a token secret).

## 1. Install the platform

One script installs the cluster dependencies, each as an idempotent
`helm upgrade --install`: the **LeaderWorkerSet** controller (multi-node workers),
**NATS / JetStream** (KV-event + request transport), and the **Infera operator**
(with its CRD + RBAC):

```bash
deploy/scripts/deploy-k8s.sh                 # LWS + NATS + operator (namespace: infera-system)
# toggles: INSTALL_NATS=false · INSTALL_GATEWAY=true (kgateway/GAIE) · OPERATOR_IMAGE_TAG=…
# deploy/scripts/deploy-k8s.sh --help  for the full list
```

Verify the operator and CRD are up:

```bash
kubectl get crd inferadeployments.infera.amd.com
kubectl -n infera-system get deploy,pod       # operator (+ NATS) Running
```

## 2. Deployment resources

| Resource | What it is |
|---|---|
| **`InferaDeployment`** (`idep`) | the canonical CR — one inference graph: a `server` plus `worker` pools. **You write this.** |
| Deployment / **LeaderWorkerSet** | the per-pool workloads the operator creates (LWS when `numberOfNodes > 1`). |
| Service + NATS | a ClusterIP for the server and the KV-event broker — created for you. |

The full field reference is on the [Operator](../components/operator.md) page.

## 3. Deploy your first model

Apply an `InferaDeployment`. Aggregated (mixed) serving first:

```yaml
# infera-qwen.yaml
apiVersion: infera.amd.com/v1alpha1
kind: InferaDeployment
metadata:
  name: qwen
spec:
  backendFramework: sglang
  image: infera/engine-sglang:dev
  nats: {deploy: true}
  services:
    server:
      componentType: server
      replicas: 1
      args: ["--router-tokenizer-path", "Qwen/Qwen3-0.6B"]
    worker:
      componentType: worker
      role: mixed
      replicas: 2
      resources: {gpu: 1, gpuType: amd.com/gpu}
      args: ["--model-path", "Qwen/Qwen3-0.6B", "--tp-size", "1"]
```

```bash
kubectl apply -f infera-qwen.yaml
```

For **PD** (prefill/decode pools) and **multi-node** (`numberOfNodes > 1` →
LeaderWorkerSet), see the full disaggregated CRD example on the
[Operator](../components/operator.md) page.

## 4. Monitor

```bash
kubectl get idep qwen -w                    # BACKEND / STATE → ready
kubectl get pods -l infera.amd.com/deployment=qwen   # server + workers Running
kubectl logs -f deploy/qwen-worker          # engine model-load progress
```

## 5. Send a request

```bash
kubectl port-forward svc/qwen-server 8000:80 &
curl localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B",
       "messages":[{"role":"user","content":"1+1=?"}],
       "max_tokens":50}'
```

## Scaling & upgrades

- **Scale** — `kubectl scale` (or edit `replicas` in the CR) on a worker pool; the
  operator reconciles the Deployment/LWS.
- **Upgrade** — change `image` / `args` in the CR and re-apply; native
  Deployment/LWS rollout drains in-flight requests.
- **Logs / registry** — `kubectl logs -l infera.amd.com/deployment=<name>`.

## Deployment templates

Instead of hand-writing the CR above, start from the ready-to-fill templates in
[`examples/k8s-deployments/`](../../examples/k8s-deployments/README.md) — one per
topology (single-node aggregated, prefill/decode disaggregated, multi-node TP,
and GAIE), each with a shared model-store PVC and documented `<PLACEHOLDERS>`.

## Gateway API (GAIE)

To front the fleet with a Kubernetes **Inference Gateway** — an Endpoint Picker
(EPP) + InferencePool + HTTPRoute + a per-worker frontend sidecar — set
`spec.gaie` on the CR and install the gateway stack:

```bash
INSTALL_GATEWAY=true deploy/scripts/deploy-k8s.sh
```

The server then runs in `--router-mode direct`, honoring the EPP's per-request
worker pick. See [Routing & transport](../features/routing_and_transport.md).

## Related

- [Operator](../components/operator.md) — the `InferaDeployment` CRD field reference.
- [Deployment](deployment.md) — the manual (`python -m`) deployment path.
