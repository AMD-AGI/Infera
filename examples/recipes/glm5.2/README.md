# GLM-5.2-MXFP4 on Kubernetes

Serve GLM-5.2-MXFP4 with infera on an AMD MI355X node (TP8, SGLang).

Built as **stock vendor image + overlay + sidecar**: the manifests run the
unmodified `lmsysorg/sglang` image and mount infera in from an overlay payload, so
an upstream SGLang bump is a one-line image-tag edit here.

## 1. Choose your combo

| Combo | Serving | KV cache | Manifest |
|---|---|---|---|
| **mixed** | one worker, prefill + decode | GPU only | [`mixed/deploy.yaml`](mixed/deploy.yaml) |
| **mixed + kvd** | one worker, prefill + decode | + L2 host RAM, L3 on a PVC | [`mixed-kvd/deploy.yaml`](mixed-kvd/deploy.yaml) |
| **pd** | prefill and decode on separate nodes | GPU only | [`pd/deploy.yaml`](pd/deploy.yaml) |
| **pd + kvd** | prefill and decode on separate nodes | + kvd per role | [`pd-kvd/deploy.yaml`](pd-kvd/deploy.yaml) |

Start with **mixed**. Add **kvd** when requests share long prefixes (a common system
prompt, multi-turn chat, RAG) — that is what the L2/L3 tiers pay for. Move to **pd**
only if you have two nodes on a mutually routable RoCE fabric.

`export COMBO=mixed` (or `mixed-kvd`, `pd`, `pd-kvd`) — the commands below use it.

## 2. Prerequisites

**Hardware.** 8× MI355X (or MI300X+) on one node for `mixed`; two such nodes on a
shared RoCE fabric for `pd`. ~700 GB of host RAM if you enable kvd — its L2 arena is
pinned host memory charged to the sidecar's limit.

**Cluster.** Kubernetes 1.28+ with:

```bash
# AMD GPU device plugin — nodes must advertise amd.com/gpu
kubectl get nodes -o custom-columns=NODE:.metadata.name,GPU:.status.allocatable.'amd\.com/gpu'

# the infera operator (provides the InferaDeployment CRD)
helm install infera-operator deploy/operator/helm/infera-operator -n infera-system --create-namespace
kubectl -n infera-system rollout status deploy/infera-operator
```

On k3s, install with `--data-dir` on a large disk. The default lives under `/var/lib`,
and pulling these images fills it — the node goes `DiskPressure` and evicts the
operator before anything serves.

**Weights.** The manifests expect the `model-cache` PVC to hold GLM-5.2-MXFP4 at
`/models/GLM-5.2-MXFP4`:

```bash
kubectl apply -f examples/k8s-deployments/model-cache/model-cache.yaml
kubectl apply -f examples/k8s-deployments/model-cache/model-download.yaml
kubectl -n infera wait --for=condition=complete job/model-download --timeout=6h
```

Keeping weights on the node instead? Replace the `model` volume with a `hostPath`.

## 3. Deploy

```bash
kubectl create namespace infera --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f examples/recipes/glm5.2/${COMBO}/deploy.yaml
```

For the `pd` combos, first substitute the two node names:

```bash
sed -e "s/<PREFILL_NODE>/node-a/" -e "s/<DECODE_NODE>/node-b/" \
    examples/recipes/glm5.2/${COMBO}/deploy.yaml | kubectl apply -f -
```

Watch it come up — weight load is several minutes, which is why the worker uses a
`startupProbe` rather than a readiness probe:

```bash
kubectl -n infera get pods -w
kubectl -n infera logs -f -c main \
  -l infera.amd.com/deployment=glm52-${COMBO},infera.amd.com/service=worker
```

## 4. Smoke test

```bash
kubectl -n infera port-forward svc/glm52-${COMBO}-server 8000:8000 &

curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm5.2-mxfp4",
       "messages":[{"role":"user","content":"What is the capital of France?"}],
       "max_tokens":60}' | jq -r '.choices[0].message.content'
```

Expect a coherent answer naming Paris. **Garbage or repeated tokens** means the ROCm
DSA indexer env vars did not take effect — check that `SGLANG_OPT_USE_TILELANG_INDEXER=1`,
`SGLANG_OPT_USE_TOPK_V2=0` and `SGLANG_OPT_USE_JIT_NORM=0` are set on the worker.

On the kvd combos, confirm KV is actually landing in the tiers — this is the check
that catches a silently-storing-nothing kvd:

```bash
POD=$(kubectl -n infera get pod -o name \
  -l infera.amd.com/deployment=glm52-${COMBO},infera.amd.com/service=worker | head -1)
kubectl -n infera exec $POD -c kvd -- \
  /overlay/bin/infera-exec python3 -m infera.kvd.statctl --socket /kvd/kvd.sock
```

`entries` and `long_bytes` must be non-zero after a few requests. If `entries: 0`,
kvd rejected the KV layout — see the `NOTE ON KV DTYPE` header in the manifest.

## 5. Tear down

```bash
kubectl -n infera delete -f examples/recipes/glm5.2/${COMBO}/deploy.yaml
```

Then check the GPUs actually released. A deleted Pod can leave processes holding
VRAM, and the next deploy OOMs on a box that looks idle:

```bash
rocm-smi --showpids
```

## Validation status

| What | Status |
|---|---|
| GLM-5.2 TP8 SGLang serving, real tokens | validated on MI355X — [`manual/examples/k8s_glm5.2_sglang.md`](../../../manual/examples/k8s_glm5.2_sglang.md) |
| kvd sidecar + PVC L3 under SGLang | validated (3674 entries / 421 MB), on Qwen3-0.6B |
| overlay payload on a stock SGLang base | validated — [`deploy/overlay/README.md`](../../../deploy/overlay/README.md) |
| these three combined, as written here | **not yet run end-to-end** |
| pd / pd-kvd for this model | structure only; needs a routable RoCE fabric |

The validated runs used the baked `rocm/infera:sglang-*` images. These manifests
compose that same software as an overlay — proven separately, not yet together.

## Source

[`examples/recipes/glm5.2/`](.) in [AMD-AGI/Infera](https://github.com/AMD-AGI/Infera)
· [all combos](../README.md) · [overlay build](../../../deploy/overlay)
