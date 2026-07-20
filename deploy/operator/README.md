# Infera Operator

A standalone Kubernetes operator for Infera. It keeps Infera
self-contained — it does **not** depend on any external inference operator.

## What it does

One CRD, `InferaDeployment` (`idep`), describes the inference graph and the
controller reconciles it into Kubernetes workloads:

- **`server`** component → a `Deployment` + ClusterIP `Service` (the
  `infera.server` router/frontend on :8000).
- **`worker`** components (`mixed` / `prefill` / `decode`) → a `Deployment`
  (single-node) or a **`LeaderWorkerSet`** (multi-node, `numberOfNodes > 1`,
  for cross-node TP/PD). GPUs are requested via **`amd.com/gpu`** by default
  (override with `resources.gpuType`).
- An operator-managed **NATS (JetStream)** `StatefulSet` + headless `Service`
  for the KV-event plane (the JetStream store dir is a PVC, driven by
  `$(NATS_STORE_DIR)`). **etcd is referenced externally** via
  `spec.etcdEndpoint`.

The controller injects `--etcd-endpoint`, the managed `--nats-server` +
`--kv-event-transport nats`, the port, and (for prefill/decode workers)
`--disaggregation-mode`, then appends each service's free-form `args`
(model path, tokenizer, tp-size, transfer backend, ib devices, ...).

## Multi-node (LeaderWorkerSet)

`numberOfNodes > 1` produces a `LeaderWorkerSet` (created as an unstructured
object so the operator takes no compile-time dependency on the LWS module).
The **LWS controller/CRD must be installed** in the cluster
(`leaderworkerset.x-k8s.io/v1`).

## Layout

```
api/v1alpha1/        CRD types (+ generated deepcopy)
internal/controller/ reconcile logic (builders, NATS, status)
cmd/main.go          manager entrypoint
config/crd|rbac      generated manifests
config/samples       example InferaDeployment (disagg PD)
```

## Develop

```bash
make generate manifests   # regenerate deepcopy + CRD/RBAC after editing api/
make build vet            # compile + vet
make docker-build IMG=... # build the manager image
make install              # kubectl apply the CRD
kubectl apply -f config/samples/
```

## Scope

Intentionally minimal: a single CRD (no separate component / request /
scaling-adapter / checkpoint / model resources), no Grove PodClique, no
operator-driven rolling update (relies on Deployment/LWS native rollout), no
EPP/GMS/failover. These can be layered in later as needed.
