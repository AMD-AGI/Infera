# Kimi-K3 on Kubernetes

Serve Kimi-K3 with infera on an AMD MI355X node (TP8, vLLM), including its
multimodal and tool-calling paths.

Built as **stock vendor image + overlay + sidecar**: the manifests run the
unmodified `vllm/vllm-openai-rocm` image and mount infera in from an overlay
payload. That separation is not cosmetic here — forking the vLLM base image for
Kimi-K3 is exactly what broke serving for every *other* model in CI. The overlay
keeps Kimi-K3's base choice from being everyone else's problem.

## 1. Choose your combo

| Combo | Serving | KV cache | Manifest |
|---|---|---|---|
| **aggregated** | one worker, prefill + decode | GPU only | [`aggregated/deploy.yaml`](aggregated/deploy.yaml) |
| **aggregated + kvd** | one worker, prefill + decode | + L2 host RAM, L3 on a PVC, **GPU-direct** | [`aggregated-kvd/deploy.yaml`](aggregated-kvd/deploy.yaml) |
| **disaggregated** | prefill and decode on separate nodes | GPU only | [`disaggregated/deploy.yaml`](disaggregated/deploy.yaml) |
| **disaggregated + kvd** | prefill and decode on separate nodes | + kvd per role | [`disaggregated-kvd/deploy.yaml`](disaggregated-kvd/deploy.yaml) |

Start with **aggregated**. Add **kvd** when requests share long prefixes (a common system
prompt, multi-turn chat, RAG). Move to **disaggregated** only if you have two nodes on a
mutually routable RoCE fabric.

Because this recipe runs vLLM, its kvd combos get **GPU-direct L3**: kvd loads the
L3 tier straight into VRAM through hipFile. SGLang's route (`--infera-kvd-socket`)
bounces through host memory instead, so GPU-direct L3 is a vLLM-only capability.

`export COMBO=aggregated` (or `aggregated-kvd`, `disaggregated`, `disaggregated-kvd`) — the commands below use it.

## 2. Prerequisites

**Hardware.** 8× MI355X (or MI300X+) on one node for `aggregated`; two such nodes on a
shared RoCE fabric for `disaggregated`. Kimi-K3's weights are ~1.5 TB — put them on **local
NVMe**, not NFS. Loading over NFS took about an hour here; local disk is minutes.

**Cluster.** Kubernetes 1.28+ with:

```bash
# AMD GPU device plugin — nodes must advertise amd.com/gpu
kubectl get nodes -o custom-columns=NODE:.metadata.name,GPU:.status.allocatable.'amd\.com/gpu'

# the infera operator (provides the InferaDeployment CRD)
helm upgrade --install infera-operator deploy/operator/helm/infera-operator \
  -n infera-system --create-namespace
kubectl -n infera-system rollout status deploy/infera-operator
```

On k3s, install with `--data-dir` on a large disk. The default lives under `/var/lib`,
and 1.5 TB of weights plus these images fills it — the node goes `DiskPressure` and
evicts the operator before anything serves.

`--data-dir` moves the image store only. kubelet keeps its root at
`/var/lib/kubelet` on the OS disk, and the default eviction threshold is
`nodefs.available<10%` of *that* disk — a node with `--data-dir` on a 7 TB NVMe
still sat `DiskPressure` for two days because its 838 GB OS disk had 0 bytes
free. Keep 10% free on both, and check with
`kubectl get --raw /api/v1/nodes/<node>/proxy/stats/summary`, which reports the
two filesystems separately.

**Weights.** The manifests expect the `model-cache` PVC to hold Kimi-K3 at
`/models/Kimi-K3`. Size the PVC for 1.5 TB before you start:

```bash
kubectl apply -f examples/k8s-deployments/model-cache/model-cache.yaml
kubectl apply -f examples/k8s-deployments/model-cache/model-download.yaml
kubectl -n infera wait --for=condition=complete job/model-download --timeout=12h
```

## 3. Deploy

```bash
kubectl create namespace infera --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f examples/recipes/kimi-k3/${COMBO}/deploy.yaml
```

For the `disaggregated` combos, first substitute the two node names:

```bash
sed -e "s/<PREFILL_NODE>/node-a/" -e "s/<DECODE_NODE>/node-b/" \
    examples/recipes/kimi-k3/${COMBO}/deploy.yaml | kubectl apply -f -
```

Weight load takes tens of minutes, which is why the worker uses a `startupProbe`
(`failureThreshold: 120`) instead of a readiness probe — a readiness probe would
declare the Pod dead long before the model is up:

```bash
kubectl -n infera get pods -w
kubectl -n infera logs -f -c main \
  -l infera.amd.com/deployment=kimi-k3-${COMBO},infera.amd.com/service=worker
```

## 4. Smoke test

**Text:**

```bash
kubectl -n infera port-forward svc/kimi-k3-${COMBO}-server 8000:8000 &

curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"kimi-k3",
       "messages":[{"role":"user","content":"What is the capital of France?"}],
       "max_tokens":200}' | jq -r '.choices[0].message.content'
```

Expect `The capital of France is Paris.` Keep `max_tokens` generous: Kimi-K3
spent 73 completion tokens on that sentence, so a 60-token cap truncates it to
the single word `The` and the recipe looks broken when it is not.

**Multimodal** — Kimi-K3 is a vision model, so the text test alone does not cover
it. `examples/kimi_k3/mm_test.py` generates a test image with the standard library
(no Pillow needed) and posts it as an image content part:

```bash
python3 examples/kimi_k3/mm_test.py --port 8000 --model kimi-k3
```

The model should describe the generated image rather than answer generically.

On the kvd combos, confirm KV is actually landing in the tiers:

```bash
POD=$(kubectl -n infera get pod -o name \
  -l infera.amd.com/deployment=kimi-k3-${COMBO},infera.amd.com/service=worker | head -1)
kubectl -n infera exec $POD -c kvd -- \
  /overlay/bin/infera-exec python3 -m infera.kvd.statctl --socket /kvd/kvd.sock
```

`entries` and `long_bytes` must be non-zero. Two things make this read zero even
when everything looks healthy:

- **The prompt was too short.** kvd stores whole chunks. A short request never
  fills one, so nothing is written and nothing is wrong. Send a few thousand tokens.
- **kvd rejected the KV layout**, logging `NOT a plain MLA fp8`. Keep
  `--kv-cache-dtype auto` as the manifest sets it.

## 5. Tear down

```bash
kubectl -n infera delete -f examples/recipes/kimi-k3/${COMBO}/deploy.yaml
```

Then check the GPUs actually released — a deleted Pod can leave processes holding
hundreds of GB of VRAM, and the next deploy OOMs on a box that looks idle:

```bash
rocm-smi --showpids
```

## Model-specific gotchas

These are not tuning preferences; each one is a failure that was hit and diagnosed
on this hardware.

| Setting | Why it is not optional |
|---|---|
| `--kv-cache-dtype auto` | infera otherwise injects fp8_e4m3, which selects the batch-1-only `mla_gluon` kernel: `requires batch_size=1, got 128`. |
| `--enable-prefix-caching` is safe here | Kimi-K3 is hybrid, so vLLM runs it in Mamba cache `align` mode and calls that experimental. Measured: 72.7% hit rate on a shared prefix, output still correct. An earlier note claimed it fails engine init — that was an older base image. |
| `--load-format auto` | `fastsafetensors` needs GDS; without it the load stalls in 30-second queue waits. |
| `startupProbe`, not readiness | weight load outlives any sane readiness deadline. |

## Validation status

| What | Status |
|---|---|
| **`aggregated/deploy.yaml` exactly as written** | **validated end-to-end on k3s (MI355X, 8×`amd.com/gpu`)** — see below |
| kvd sidecar + PVC L3 under vLLM | validated (2 entries / 271 MB on a 3510-token request), on Qwen3-0.6B |
| pd / pd-kvd for this model | structure only; the vLLM MultiConnector wiring is unproven for Kimi-K3 |

The `aggregated` run used the **stock `vllm/vllm-openai-rocm:kimi-k3` image** with the
overlay mounted in — no infera-built engine image anywhere in the Pod:

| | |
|---|---|
| router Ready | ~20 s (`infera.server` running from `/overlay`) |
| weight load | 502 s, 96 shards, 1453 GiB from local XFS NVMe |
| worker Ready | ~11 min including aiter JIT + CUDA graph capture |
| text | `The capital of France is Paris.` (73 completion tokens, `finish_reason: stop`) |
| multimodal | correctly identified the generated crimson image |

That run is also what surfaced the payload-shadowing bug: the engine logged
`ImportError: Numba needs NumPy 2.4 or less. Got NumPy 2.5` on repeat, because
the overlay was shipping numpy 2.5.1 over the base's 2.3.5. Non-fatal here, but
fixed properly in `deploy/overlay/prune_base_dists.py` — the payload now adds
only what the base lacks.

## Source

[`examples/recipes/kimi-k3/`](.) in [AMD-AGI/Infera](https://github.com/AMD-AGI/Infera)
· [all combos](../README.md) · [overlay build](../../../deploy/overlay)
