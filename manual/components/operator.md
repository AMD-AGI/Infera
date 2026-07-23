# Operator (Kubernetes)

The **Infera operator** is a Kubernetes controller that reconciles a single
`InferaDeployment` custom resource (short name `idep`) into all the workloads of
an inference graph — the server, the worker pools, and the KV-event plane. It's the
first-class way to run **PD and multi-node** on Kubernetes.

## What it reconciles

You declare a `server` plus one or more `worker` pools (`mixed` / `prefill` /
`decode`); the operator builds:

- a **Deployment + ClusterIP Service** for the `server`,
- a **Deployment** (single-node) or a **LeaderWorkerSet** (`numberOfNodes > 1`,
  for cross-node TP or a multi-node PD group) for each worker pool — GPUs are
  requested via `amd.com/gpu`,
- an operator-managed **NATS (JetStream)** StatefulSet for the KV-event plane.

It injects `--disaggregation-mode` (for prefill/decode workers), the discovery +
managed-NATS (`--kv-event-transport nats`) flags, and the ports; you supply the
rest (model, tp-size, transfer backend, IB device) per service in `args`.

## Example — disaggregated (PD)

```yaml
apiVersion: infera.amd.com/v1alpha1
kind: InferaDeployment
metadata:
  name: demo
spec:
  backendFramework: sglang
  image: rocm/infera:sglang-v0.1.1
  discoveryBackend: kubernetes             # in-cluster API; no external etcd
  nats: {deploy: true, storageSize: 4Gi}   # operator-managed JetStream (KV events)
  services:
    server:
      componentType: server
      replicas: 2
      args: ["--router-policy","kv-aware","--router-tokenizer-path","Qwen/Qwen3-0.6B"]
    prefill:
      componentType: worker
      role: prefill                        # → operator adds --disaggregation-mode prefill
      replicas: 1
      numberOfNodes: 1                     # >1 ⇒ LeaderWorkerSet (cross-node TP / PD group)
      resources: {gpu: 1, gpuType: amd.com/gpu, memory: 64Gi, sharedMemory: 32Gi}
      args: ["--model-path","Qwen/Qwen3-0.6B","--tp-size","1","--enable-kv-events",
             "--disaggregation-transfer-backend","mori","--disaggregation-ib-device","<rdma-nic>"]
    decode:
      componentType: worker
      role: decode                         # → operator adds --disaggregation-mode decode
      replicas: 1
      resources: {gpu: 1, gpuType: amd.com/gpu, memory: 64Gi, sharedMemory: 32Gi}
      args: ["--model-path","Qwen/Qwen3-0.6B","--tp-size","1","--enable-kv-events",
             "--disaggregation-transfer-backend","mori","--disaggregation-ib-device","<rdma-nic>"]
```

```bash
kubectl apply -f my-idep.yaml
kubectl get idep                 # BACKEND / STATE / AGE
```

- **Roles:** `mixed` (aggregated) · `prefill` · `decode`. The server pod has no GPU.
- **Multi-node:** `numberOfNodes > 1` produces a **LeaderWorkerSet** (cross-node TP
  or a multi-node PD group) — install the LWS CRD (`leaderworkerset.x-k8s.io/v1`).
- **Discovery:** `kubernetes` (default, no etcd — the operator provisions the
  ServiceAccount + Role) or `etcd` (external; set `spec.etcdEndpoint`).
- **Gateway API (GAIE):** `spec.gaie` adds a per-worker frontend sidecar + Endpoint
  Picker + InferencePool + HTTPRoute to front the fleet with an Inference Gateway
  (the server runs in `--router-mode direct` — see [Server & router](server.md)).

## Related

- [Deployment](../serving/deployment.md) — the manual and raw-manifest
  Kubernetes paths.
