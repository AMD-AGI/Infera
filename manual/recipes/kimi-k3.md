# Kimi-K3

Serve Kimi-K3 with infera on an AMD MI355X node (VLLM, TP8).

Every combination below runs the **stock `vllm/vllm-openai-rocm` image** with the infera overlay
mounted in — the vendor image is never forked, so following an upstream release is
an image-tag edit.

## 1. Choose the combination

::::{tab-set}

:::{tab-item} Mixed
:sync: mixed

One worker does prefill and decode. Start here.

```bash
kubectl apply -f examples/recipes/kimi-k3/mixed/deploy.yaml
kubectl -n infera get pods -w
```

[`kimi-k3/mixed/deploy.yaml`](https://github.com/AMD-AGI/Infera/tree/main/examples/recipes/kimi-k3/mixed/deploy.yaml)
:::
:::{tab-item} Mixed + kvd
:sync: mixed-kvd

Adds the kvd tiered cache: an L2 arena in pinned host RAM and an L3 tier on a PVC. Worth it when requests share long prefixes.

This combination declares `INFERA_REQUIRE_NATIVE=hipfile`, so it fails at startup rather than serving quietly without it.

```bash
kubectl apply -f examples/recipes/kimi-k3/mixed-kvd/deploy.yaml
kubectl -n infera get pods -w
```

[`kimi-k3/mixed-kvd/deploy.yaml`](https://github.com/AMD-AGI/Infera/tree/main/examples/recipes/kimi-k3/mixed-kvd/deploy.yaml)
:::
:::{tab-item} PD
:sync: pd

Prefill and decode on separate nodes, KV handed over by Mooncake. **Two engines
disaggregate differently — pick by which engine you want, not by which is "the"
PD path:**

| Engine | Mechanism | Manifest | Image |
|---|---|---|---|
| vLLM | `--kv-transfer-config` with a `MooncakeConnector` | `pd/deploy.yaml` | `vllm/vllm-openai-rocm:kimi-k3` |
| SGLang | `--disaggregation-mode` + a Mooncake bootstrap handshake | `pd-sglang/deploy.yaml` | `lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727` |

The general `lmsysorg/sglang` tags carry **no** Kimi-K3 support — only that dated
`-k3-` build does. Weights must sit on storage both nodes see at the same path;
at ~1.5 TB a per-node copy is usually not an option.

This combination declares `INFERA_REQUIRE_NATIVE=mooncake`, so it fails at startup rather than serving quietly without it.

```{admonition} Two nodes on a routable RoCE fabric
:class: warning
Substitute `<PREFILL_NODE>` and `<DECODE_NODE>` first. The KV handoff is RDMA with no TCP fallback — on one box this cannot work.
```

```bash
kubectl apply -f examples/recipes/kimi-k3/pd/deploy.yaml
kubectl -n infera get pods -w
```

[`kimi-k3/pd/deploy.yaml`](https://github.com/AMD-AGI/Infera/tree/main/examples/recipes/kimi-k3/pd/deploy.yaml)
:::
:::{tab-item} PD + kvd
:sync: pd-kvd

Disaggregated, with kvd on each role.

This combination declares `INFERA_REQUIRE_NATIVE=mooncake,hipfile`, so it fails at startup rather than serving quietly without it.

```{admonition} Two nodes on a routable RoCE fabric
:class: warning
Substitute `<PREFILL_NODE>` and `<DECODE_NODE>` first. The KV handoff is RDMA with no TCP fallback — on one box this cannot work.
```

```bash
kubectl apply -f examples/recipes/kimi-k3/pd-kvd/deploy.yaml
kubectl -n infera get pods -w
```

[`kimi-k3/pd-kvd/deploy.yaml`](https://github.com/AMD-AGI/Infera/tree/main/examples/recipes/kimi-k3/pd-kvd/deploy.yaml)
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
kubectl -n infera port-forward svc/kimi-k3-<combo>-server 8000:8000 &

curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"kimi-k3",
       "messages":[{"role":"user","content":"What is the capital of France?"}],
       "max_tokens":200}' | jq -r '.choices[0].message.content'
```

Expect `The capital of France is Paris.` Keep `max_tokens` generous — both models
here spend 70+ completion tokens on that sentence, and a tight cap truncates it into
something that reads like a broken deployment.

**Multimodal.** Kimi-K3 is a vision model, so the text test alone does not cover it.
`examples/kimi_k3/mm_test.py` builds a test image with the standard library and posts
it as an image content part:

```bash
python3 examples/kimi_k3/mm_test.py --port 8000 --model kimi-k3
```

On the kvd combinations, confirm KV is actually landing in the tiers:

```bash
POD=$(kubectl -n infera get pod -o name \
  -l infera.amd.com/deployment=kimi-k3-<combo>,infera.amd.com/service=worker | head -1)
kubectl -n infera exec $POD -c kvd -- \
  /overlay/bin/infera-exec python3 -m infera.kvd.statctl --socket /kvd/kvd.sock
```

`entries` and `long_bytes` must be non-zero after a few requests. Zero usually means
the prompt was too short to fill a chunk, or kvd refused the KV layout — see the
manifest's own KV-dtype note.

## 4. Tear down

```bash
kubectl -n infera delete -f examples/recipes/kimi-k3/<combo>/deploy.yaml
rocm-smi --showpids     # a deleted Pod can leave processes holding VRAM
```

## Model-specific gotchas

Each of these is a failure that was hit and diagnosed on this hardware, not a tuning
preference.

| Setting | Why it is not optional |
|---|---|
| `--kv-cache-dtype auto` | infera otherwise injects fp8_e4m3, which selects the batch-1-only `mla_gluon` kernel and warmup dies with `requires batch_size=1, got 128`. |
| `--enable-prefix-caching` is safe here | Kimi-K3 is hybrid, so vLLM runs it in Mamba cache `align` mode and calls that experimental. Measured: 72.7% hit rate on a shared prefix, output still correct. An earlier note claimed it fails engine init — that was an older base image. |
| `--load-format auto` | `fastsafetensors` needs GDS; without it the load stalls in 30-second queue waits. |
| `startupProbe`, not readiness | ~1.5 TB of weights outlives any sane readiness deadline. |

## Source

[`examples/recipes/kimi-k3/`](https://github.com/AMD-AGI/Infera/tree/main/examples/recipes/kimi-k3) · [all recipes](index)
