# Infera Kubernetes deployment examples

`InferaDeployment` templates for the infera-operator, one per topology. All
cluster-specific values are `<PLACEHOLDERS>` — fill them in (each file has a
header listing the ones it uses) before applying.

## Prerequisite

`pvc-models.yaml` first — it provides the model-store PVC every example mounts
at `/models`:

```bash
kubectl apply -f pvc-models.yaml
kubectl apply -f single-node-aggregated.yaml   # then any topology below
```

## Examples

| File | Topology | Notes |
|------|----------|-------|
| `pvc-models.yaml` | Model-store PV/PVC | Shared prerequisite; mounted at `/models`. |
| `single-node-aggregated.yaml` | 1 router + 1 mixed worker (tp=1) | Simplest; good smoke test. |
| `pd-disaggregated.yaml` | Router + prefill pool + decode pool | KV streamed prefill→decode over RoCE RDMA (mori). Needs `rdma/hca` + the RDMA placeholders. |
| `multinode-tp-aggregated.yaml` | One worker spanning 2 nodes (tp=2) | Cross-node tensor parallel via LeaderWorkerSet. Needs the LWS CRD/controller. |
| `kgateway-gaie-aggregated.yaml` | Worker behind an Inference Gateway | Operator injects EPP + InferencePool + frontend sidecar. Needs the InferencePool CRD; full data path needs a Gateway (e.g. kgateway). |

The router reaches workers over HTTP and workers publish KV events over zmq, so
no NATS broker is deployed (`nats.deploy: false`).

## Placeholders

Common (all files):

| Placeholder | Meaning |
|-------------|---------|
| `<NAMESPACE>` | Target namespace (must exist). |
| `<ENGINE_IMAGE>` | Infera engine image; must contain `infera.server` + `infera.engine.sglang`. |
| `<IMAGE_PULL_SECRET>` | `docker-registry` secret in `<NAMESPACE>` for that image. |
| `<MODEL_PVC>` | Model-store PVC name (from `pvc-models.yaml`). |
| `<MODEL_PATH>` | Model dir path **inside** the pod, i.e. `/models/<model-dir>`. |

`pvc-models.yaml` (storage backend):

| Placeholder | Meaning |
|-------------|---------|
| `<CSI_DRIVER>` | CSI driver name (e.g. `csi.weka.io`). |
| `<CSI_VOLUME_HANDLE>` | Backend volume handle/path containing the models. |
| `<CSI_SECRET_NAME>` / `<CSI_SECRET_NAMESPACE>` | CSI node/controller secret and its namespace. |

`pd-disaggregated.yaml` (RoCE RDMA):

| Placeholder | Meaning |
|-------------|---------|
| `<RDMA_IB_DEVICE>` | RoCE device for mori (one name, from `ibv_devices`). |
| `<RDMA_HCA_LIST>` | Comma-separated RoCE devices for `NCCL_IB_HCA`. |
| `<ROCE_V2_GID_INDEX>` | GID index that is RoCE v2 **and** IPv4 (`show_gids <dev>`); wrong index makes KV transfer time out. |

`kgateway-gaie-aggregated.yaml` (Inference Gateway):

| Placeholder | Meaning |
|-------------|---------|
| `<GATEWAY_CLASS>` | GatewayClass name (e.g. `kgateway`). |
| `<MODEL_HF_ID>` | HF id for the EPP/sidecar tokenizer (they have no model mount) and served model name; must match the worker tokenizer. |
