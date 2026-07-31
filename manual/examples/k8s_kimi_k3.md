# Example: Kimi-K3 on Kubernetes — vLLM, model-cache, single node

A **Kubernetes-native** recipe for **Kimi-K3** (the multimodal VLM): a
`model-cache` PVC populated by a download Job, then an **`InferaDeployment`** the
operator reconciles into a router + a TP8 vLLM worker on one 8-GPU node, using
only `kubectl`/`helm` and one CR.

The runnable manifests live in the repo under
[`examples/k8s-deployments/`](https://github.com/AMD-AGI/Infera/tree/main/examples/k8s-deployments)
(`model-cache/`, `kimi-k3-vllm.yaml`). For the general k8s flow and the CR field
reference, read [Kubernetes deployment](../serving/kubernetes.md) first.

```{admonition} What is validated
:class: important
The **recipe mechanics** are validated end to end on single-node **k3s**
(MI355X, `amd.com/gpu`): the `model-cache` PVC + download Job, the operator
reconciling an `InferaDeployment` into router + worker + Service, and a real
`/v1/chat/completions` completion — checked with both the **sglang** and **vLLM**
engines on `Qwen3-0.6B`, including serving straight from the model-cache PVC
(see [`RECIPE-single-node.md`](https://github.com/AMD-AGI/Infera/blob/main/examples/k8s-deployments/RECIPE-single-node.md)).

Kimi-K3 needs a vLLM build with `kimi_k3` model support. Layer infera onto the
upstream `kimi_k3` vLLM by overriding the base of the standard vLLM image build:

```bash
docker build -f deploy/docker/Dockerfile.vllm \
  --build-arg VLLM_BASE_IMAGE=vllm/vllm-openai-rocm:kimi-k3 \
  --build-arg BUILD_AITER=0 --build-arg BUILD_MOONCAKE=0 \
  --build-arg BUILD_HIPFILE=0 --build-arg INSTALL_LIBIONIC=0 \
  -t rocm/infera:vllm-kimi-k3 .
```

The result carries `infera.server` + `infera.engine.vllm` **and**
`KimiK3ForConditionalGeneration`. With this image the full path is validated on a
single 8-GPU node: operator → router → `infera.engine.vllm` TP8 worker serving
Kimi-K3 (MXFP4), and a **multimodal** image request through the router returns a
correct colour-grounded answer. The disabled layers are RDMA-PD / kvd-L3 only;
aiter stays the base's kimi-tuned build.
```

## Topology

One 8-GPU node, aggregated (no PD): a CPU-only **router** (`infera.server`) in
front of one **mixed vLLM worker** at **TP8**, both mounting the same model-cache
PVC read-only.

| Component | GPUs | What it runs |
|---|---|---|
| `kimi-k3-server` | 0 | `infera.server` — OpenAI endpoint + router, ClusterIP `:8000` |
| `kimi-k3-worker` | 8 | `infera.engine.vllm` — Kimi-K3, `--tensor-parallel-size 8`, `--mm-encoder-tp-mode data` |

## Prerequisites

- A Kubernetes cluster with one **8× ROCm-GPU node** and `kubectl` + `helm`.
  Single-node **k3s** is fine — put its data-dir on a **large disk** so the
  engine-image import doesn't fill root:
  ```bash
  curl -sfL https://get.k3s.io | \
    INSTALL_K3S_EXEC="--write-kubeconfig-mode=644 --data-dir /mnt/<big-disk>/k3s" sh -
  export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
  ```
- **AMD GPU device plugin** so GPUs schedule as `amd.com/gpu`:
  ```bash
  kubectl apply -f https://raw.githubusercontent.com/ROCm/k8s-device-plugin/master/k8s-ds-amdgpu-dp.yaml
  kubectl describe node | grep amd.com/gpu     # -> amd.com/gpu: 8
  ```
- The **infera-operator** (CRD + controller); no LWS/NATS needed for single-node
  aggregated:
  ```bash
  INSTALL_LWS=false INSTALL_NATS=false deploy/scripts/deploy-k8s.sh
  kubectl get crd inferadeployments.infera.amd.com
  ```
- The engine **image present in the node container runtime**. k3s uses
  containerd (not docker) — import a local image as a tar, not a stream:
  ```bash
  docker save <infera-vllm-kimi-k3-image> -o /mnt/<big-disk>/img.tar
  sudo k3s ctr images import /mnt/<big-disk>/img.tar
  ```

## 1. Model cache (PVC + download Job)

The [`model-cache/`](https://github.com/AMD-AGI/Infera/tree/main/examples/k8s-deployments/model-cache)
pair is a PVC plus a Job that downloads the checkpoint into it. Kimi-K3 is
**gated (~1.5 TB)**, so create the HF token secret and size the PVC to ~2000 Gi
first:

```bash
kubectl create namespace infera
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=<token> -n infera

# edit model-cache.yaml: storage: 2000Gi (+ a ReadWriteMany class if multi-node)
# edit model-download.yaml: uncomment envFrom(hf-token-secret); repo -> moonshotai/Kimi-K3
kubectl apply -f examples/k8s-deployments/model-cache/model-cache.yaml -n infera
kubectl apply -f examples/k8s-deployments/model-cache/model-download.yaml -n infera
kubectl wait --for=condition=Complete job/model-download -n infera --timeout=8h
```

```{admonition} Already have the checkpoint on the node?
:class: tip
Skip the download Job and back the model volume with a `hostPath` pointing at the
cached directory instead of the PVC (see `single-node-qwen-*.yaml` for the
`hostPath` form). The download Job is for a cold cluster.
```

## 2. Deploy

```bash
# set image: in kimi-k3-vllm.yaml to your kimi_k3-capable infera-vLLM image
kubectl apply -f examples/k8s-deployments/kimi-k3-vllm.yaml -n infera

kubectl -n infera get inferadeployment -w    # STATE -> ready (weights + graphs: several min)
kubectl -n infera get pods                   # kimi-k3-server 1/1, kimi-k3-worker 1/1
```

The operator creates the two Deployments, the `kimi-k3-server` ClusterIP
Service on `:8000`, and the k8s-discovery ServiceAccount.

## 3. Send a multimodal request

```bash
kubectl -n infera port-forward svc/kimi-k3-server 8000:8000 &
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "kimi-k3", "max_tokens": 64,
  "messages": [{"role":"user","content":[
    {"type":"text","text":"What is the dominant colour of this image?"},
    {"type":"image_url","image_url":{"url":"data:image/png;base64,<...>"}}
  ]}]}'
```

A colour-grounded answer confirms the vision → prefill → decode path. (The same
image test against the standalone docker serve is
[`examples/kimi_k3/mm_test.py`](https://github.com/AMD-AGI/Infera/blob/main/examples/kimi_k3/mm_test.py).)

## Notes & gotchas

- **k3s stores images under `--data-dir`.** A 30–80 GB engine image imported into
  the default (root-disk) location trips the kubelet DiskPressure eviction
  threshold and evicts the operator/worker. Put the data-dir on a large disk.
- **containerd ≠ docker.** A locally-built image must be imported into the node
  runtime (`k3s ctr images import <tar>`); `imagePullPolicy: IfNotPresent` then
  resolves it without a registry.
- **Model mount needs `extraPodSpec`.** The operator's simpler `args` mode
  auto-builds the entrypoint but only mounts `dshm` + `/boot`; to mount a
  PVC/hostPath model you must give the full `command` (as these manifests do).
- **sglang ↔ vLLM is not just the image.** The router is engine-agnostic; the
  worker's entrypoint module *and* flags differ (`--model-path`/`--tp-size` vs
  `--model`/`--tensor-parallel-size`). See `RECIPE-single-node.md`.
- **Kimi-K3 needs `--kv-cache-dtype auto`.** `infera.engine.vllm` defaults to an
  fp8_e4m3 KV cache, but Kimi-K3's MLA then picks the `mla_gluon` kernel
  (batch_size=1 only) and crashes warmup at `--max-num-seqs 128`. Pass
  `--kv-cache-dtype auto` (or env `INFERA_DEFAULT_KV_FP8=0`) to keep the native KV.
- **Slow load vs the readiness probe.** Kimi-K3 loads ~1.5 TB (~7 min); set
  `skipReadinessProbe: true` on the worker so the operator doesn't restart it
  mid-load.
