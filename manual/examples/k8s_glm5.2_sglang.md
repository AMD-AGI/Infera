# Example: GLM-5.2 on Kubernetes — SGLang (DSA), single node

A **Kubernetes-native** recipe for **GLM-5.2-MXFP4** (`GlmMoeDsaForCausalLM` —
MLA + DeepSeek Sparse Attention): an **`InferaDeployment`** the operator
reconciles into a router + a TP8 **SGLang** worker on one 8-GPU node. SGLang is
the correct and fastest engine for GLM-5.2 on ROCm.

The manifest is
[`examples/k8s-deployments/glm5.2-sglang.yaml`](https://github.com/AMD-AGI/Infera/tree/main/examples/k8s-deployments/glm5.2-sglang.yaml).
See [Kubernetes deployment](../serving/kubernetes.md) for the general flow and
the [Kimi-K3 recipe](k8s_kimi_k3.md) for the vLLM/multimodal single-node case.

```{admonition} Use SGLang, not vLLM, for GLM-5.2
:class: important
GLM-5.2's MLA/DSA decode is **numerically buggy on vLLM** (ROCm) — the engine
loads and serves, but decode degrades to `!!!!` garbage after a few tokens.
**SGLang produces correct, coherent output** and is the fastest engine here.
Validated on k3s (MI355X): worker Ready ~6 min, a `/v1/chat/completions` request
via the router returns a correct answer ("… the capital of France is Paris …");
the same request on the vLLM image returns garbage.
```

```{admonition} What is validated
:class: note
On single-node k3s (MI355X, `amd.com/gpu: 8`), GLM-5.2-MXFP4 at TP8 through this
manifest: the operator reconciled the CR into router + worker, the worker reached
Ready in **~6 min** (~408 GiB of weights plus DSA graph capture), and a
`/v1/chat/completions` request through the router returned a **correct, coherent**
answer.

The same request against the vLLM image on the same hardware returned
`'1!!!!!!!...'` — which is why this recipe is SGLang-only.
```

```{admonition} REQUIRED ROCm env — GLM-5.2 crashes at init without it
:class: warning
Stock SGLang defaults a **CUDA-only DSA top-k JIT kernel** that will not build on
gfx950 (MI355X) and crashes engine init. Force the ROCm tilelang path:
`SGLANG_OPT_USE_TILELANG_INDEXER=1`, `SGLANG_OPT_USE_TOPK_V2=0`,
`SGLANG_OPT_USE_JIT_NORM=0` (+ `SGLANG_USE_AITER=1`,
`SGLANG_ROCM_FUSED_DECODE_MLA=0`), and pass `--nsa-prefill-backend tilelang
--nsa-decode-backend tilelang`. Needs SGLang **0.5.15+** — older ROCm SGLang
can't load GLM-5.2's changed `head_dim`. All wired into the manifest.
```

## Topology

One 8-GPU node, aggregated: a CPU-only **router** (`infera.server`) in front of
one **mixed SGLang worker** at **TP8**, both mounting the model read-only.

| Component | GPUs | What it runs |
|---|---|---|
| `glm52-server` | 0 | `infera.server` — OpenAI endpoint + router, ClusterIP `:8000` |
| `glm52-worker` | 8 | `infera.engine.sglang` — GLM-5.2, `--tp-size 8`, DSA tilelang backends, fp8 KV |

## Prerequisites

- A Kubernetes cluster with one **8× ROCm-GPU node**, `kubectl`/`helm`, the AMD
  GPU device plugin (`amd.com/gpu`), and the infera-operator — see
  [Kubernetes deployment](../serving/kubernetes.md).
- The SGLang engine image (`rocm/infera:sglang-v0.1.1`, SGLang 0.5.15+) present
  in the node's containerd (`k3s ctr images import` for a local image).
- The GLM-5.2-MXFP4 checkpoint reachable inside the pod — a `hostPath` (as in the
  manifest) or the {ref}`model-cache <model-cache-pvc-download-job>` PVC.

## Deploy & verify

```bash
kubectl create namespace infera
# adjust the model hostPath / image in the manifest for your node
kubectl apply -f examples/k8s-deployments/glm5.2-sglang.yaml

kubectl -n infera get inferadeployment -w     # STATE -> ready (~6-18 min: weights + DSA graphs)
kubectl -n infera get pods                     # glm52-server 1/1, glm52-worker 1/1

kubectl -n infera port-forward svc/glm52-server 8000:8000 &
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm5.2-mxfp4","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":60}'
```

A coherent answer confirms router → SGLang DSA worker end to end.

## Notes & gotchas

- **fp8 KV is fine on SGLang** (`--kv-cache-dtype fp8_e4m3`) — SGLang handles the
  fp8 MLA KV correctly for GLM-5.2. (On vLLM, fp8 KV gives immediate garbage and
  even bf16 KV degrades — hence "use SGLang".)
- **Bound the DSA memory.** `--chunked-prefill-size 8192` (the DSA indexer
  profiling OOMs higher) and a modest `--context-length` keep the KV pool in
  check; production long-context runs push `--context-length` to ~400000.
- **Slow load — startupProbe.** ~408 GiB of weights load in ~6-18 min (faster
  from local NVMe than a shared FS); a generous `startupProbe` holds the pod
  non-Ready through the load. See the [Kimi-K3 recipe](k8s_kimi_k3.md) for the
  same pattern.
- **Free the GPUs between runs.** Deleting the `InferaDeployment` does not always
  reap the engine processes; a lingering TP group can hold VRAM and the next
  deploy fails with "free memory … less than desired". Kill stragglers
  (`rocm-smi --showpids`) before redeploying.
