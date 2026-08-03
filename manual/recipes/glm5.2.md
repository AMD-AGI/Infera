# GLM-5.2-MXFP4

Serve GLM-5.2-MXFP4 with infera on an AMD MI355X node (SGLANG, TP8).

Every combination below runs the **stock `lmsysorg/sglang` image** with the infera overlay
mounted in — the vendor image is never forked, so following an upstream release is
an image-tag edit.

## 1. Choose the combination

::::{tab-set}

:::{tab-item} Aggregated
:sync: aggregated

One worker does prefill and decode. Start here.

```bash
kubectl apply -f examples/recipes/glm5.2/aggregated/deploy.yaml
kubectl -n infera get pods -w
```

[`glm5.2/mixed/deploy.yaml`](https://github.com/AMD-AGI/Infera/tree/main/examples/recipes/glm5.2/aggregated/deploy.yaml)
:::
:::{tab-item} Aggregated + kvd
:sync: aggregated-kvd

Adds the kvd tiered cache: an L2 arena in pinned host RAM and an L3 tier on a PVC. Worth it when requests share long prefixes.

```bash
kubectl apply -f examples/recipes/glm5.2/aggregated-kvd/deploy.yaml
kubectl -n infera get pods -w
```

[`glm5.2/mixed-kvd/deploy.yaml`](https://github.com/AMD-AGI/Infera/tree/main/examples/recipes/glm5.2/aggregated-kvd/deploy.yaml)
:::
:::{tab-item} Disaggregated
:sync: disaggregated

Prefill and decode on separate nodes, KV handed over by Mooncake.

This combination declares `INFERA_REQUIRE_NATIVE=mooncake`, so it fails at startup rather than serving quietly without it.

```{admonition} Two nodes on a routable RoCE fabric
:class: warning
Substitute `<PREFILL_NODE>` and `<DECODE_NODE>` first. The KV handoff is RDMA with no TCP fallback — on one box this cannot work.
```

```bash
kubectl apply -f examples/recipes/glm5.2/disaggregated/deploy.yaml
kubectl -n infera get pods -w
```

[`glm5.2/pd/deploy.yaml`](https://github.com/AMD-AGI/Infera/tree/main/examples/recipes/glm5.2/disaggregated/deploy.yaml)
:::
:::{tab-item} Disaggregated + kvd
:sync: disaggregated-kvd

Disaggregated, with kvd on each role.

This combination declares `INFERA_REQUIRE_NATIVE=mooncake`, so it fails at startup rather than serving quietly without it.

```{admonition} Two nodes on a routable RoCE fabric
:class: warning
Substitute `<PREFILL_NODE>` and `<DECODE_NODE>` first. The KV handoff is RDMA with no TCP fallback — on one box this cannot work.
```

```bash
kubectl apply -f examples/recipes/glm5.2/disaggregated-kvd/deploy.yaml
kubectl -n infera get pods -w
```

[`glm5.2/pd-kvd/deploy.yaml`](https://github.com/AMD-AGI/Infera/tree/main/examples/recipes/glm5.2/disaggregated-kvd/deploy.yaml)
:::

::::

## 2. Prerequisites

```bash
# nodes must advertise amd.com/gpu
kubectl get nodes -o custom-columns=NODE:.metadata.name,GPU:.status.allocatable.'amd\.com/gpu'

# the operator (provides the InferaDeployment CRD)
helm install infera-operator deploy/operator/helm/infera-operator -n infera-system --create-namespace
```

The weights are expected in the `model-cache` PVC. On k3s, install with `--data-dir`
on a large disk — the default lives under `/var/lib`, and these images plus the
checkpoint fill it, at which point the node goes `DiskPressure` and evicts the
operator before anything serves.

```bash
kubectl apply -f examples/k8s-deployments/model-cache/model-cache.yaml
kubectl apply -f examples/k8s-deployments/model-cache/model-download.yaml
```

```{admonition} Edit the download Job first
:class: important
`model-download.yaml` ships with a small default model. Change the repo id (and the
target directory) to the one this page is about before applying it, or the
deployment will come up and find nothing to load.
```

## 3. Smoke test

```bash
kubectl -n infera port-forward svc/glm52-<combo>-server 8000:8000 &

curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm5.2-mxfp4",
       "messages":[{"role":"user","content":"What is the capital of France?"}],
       "max_tokens":200}' | jq -r '.choices[0].message.content'
```

Expect `The capital of France is Paris.` Keep `max_tokens` generous — both models
here spend 70+ completion tokens on that sentence, and a tight cap truncates it into
something that reads like a broken deployment.

On the kvd combinations, confirm KV is actually landing in the tiers:

```bash
POD=$(kubectl -n infera get pod -o name \
  -l infera.amd.com/deployment=glm52-<combo>,infera.amd.com/service=worker | head -1)
kubectl -n infera exec $POD -c kvd -- \
  /overlay/bin/infera-exec python3 -m infera.kvd.statctl --socket /kvd/kvd.sock
```

`entries` and `long_bytes` must be non-zero after a few requests. Zero usually means
the prompt was too short to fill a chunk, or kvd refused the KV layout — see the
manifest's own KV-dtype note.

## 4. Tear down

```bash
kubectl -n infera delete -f examples/recipes/glm5.2/<combo>/deploy.yaml
rocm-smi --showpids     # a deleted Pod can leave processes holding VRAM
```

## Model-specific gotchas

Each of these is a failure that was hit and diagnosed on this hardware, not a tuning
preference.

| Setting | Why it is not optional |
|---|---|
| `SGLANG_OPT_USE_TILELANG_INDEXER=1`, `SGLANG_OPT_USE_TOPK_V2=0`, `SGLANG_OPT_USE_JIT_NORM=0` | stock SGLang defaults to a CUDA-only DSA top-k JIT kernel that will not build on gfx950 and crashes engine init. |
| `--nsa-prefill-backend tilelang`, `--nsa-decode-backend tilelang` | same reason, on the attention path. |
| `--reasoning-parser glm45` | GLM-5.2 is a thinking model. Without it the answer is still correct but arrives with the chain-of-thought and a raw `</think>` inlined in `content`. |
| SGLang, not vLLM | GLM-5.2's MLA/DSA decode is numerically buggy on vLLM/ROCm — it serves, then degrades to garbage. |

## Source

[`examples/recipes/glm5.2/`](https://github.com/AMD-AGI/Infera/tree/main/examples/recipes/glm5.2) · [all recipes](index)
