# GLM-5.2-MXFP4 on Kubernetes

Serve GLM-5.2-MXFP4 with infera on an AMD MI355X node (TP8, SGLang).

Built as **stock vendor image + overlay + sidecar**: the manifests run the
unmodified `lmsysorg/sglang` image and mount infera in from an overlay payload, so
an upstream SGLang bump is a one-line image-tag edit here.

## 1. Choose your combo

| Combo | Serving | KV cache | Manifest |
|---|---|---|---|
| **aggregated** | one worker, prefill + decode | GPU only | [`aggregated/deploy.yaml`](aggregated/deploy.yaml) |
| **aggregated + kvd** | one worker, prefill + decode | + L2 host RAM, L3 on a PVC | [`aggregated-kvd/deploy.yaml`](aggregated-kvd/deploy.yaml) |
| **disaggregated** | prefill and decode on separate nodes | GPU only | [`disaggregated/deploy.yaml`](disaggregated/deploy.yaml) |
| **disaggregated + kvd** | prefill and decode on separate nodes | + kvd per role | [`disaggregated-kvd/deploy.yaml`](disaggregated-kvd/deploy.yaml) |

Start with **aggregated**. Add **kvd** when requests share long prefixes (a common system
prompt, multi-turn chat, RAG) — that is what the L2/L3 tiers pay for. Move to **disaggregated**
only if you have two nodes on a mutually routable RoCE fabric.

`export COMBO=aggregated` (or `aggregated-kvd`, `disaggregated`, `disaggregated-kvd`) — the commands below use it.

```{admonition} What each combo needs from the overlay's native tree
The overlay harvests one native tree per ABI family — `rocm7-py310` for SGLang
bases, `rocm7-py312` for vLLM — and each records what it carries:

| combo | needs | why |
|---|---|---|
| `aggregated` | nothing | |
| `aggregated-kvd` | nothing | SGLang reaches kvd's L2 and file L3 through HiCacheStorage, which is pure Python |
| `disaggregated`, `disaggregated-kvd` | `mooncake` | the KV handoff is Mooncake, a compiled extension |

`hipfile` is **absent from the SGLang tree on purpose** — kvd's GPU-direct L3 is
a vLLM-only path — so its absence here is not a broken payload.

The combos that need something declare it as `INFERA_REQUIRE_NATIVE`, and
`infera-exec` fails at startup if the payload cannot supply it. Without that, a
payload missing Mooncake comes up green and serves with no KV transfer at all.
```

## 2. Prerequisites

```{admonition} kv-aware needs the tokenizer on the ROUTER, not just the workers
:class: warning
The server pod mounts `/models` and passes `--router-tokenizer-path` a **local
path** — both manifests already do this, and both matter. kv-aware tokenizes each
request on the router to compare its prefix against what the workers cached, so
the router needs the same tokenizer files the engines load.

Point it at a hub id instead and it resolves an HF cache directory that may hold
only `tokenizer_config.json`. The load fails, every request hashes to zero
blocks, and kv-aware quietly becomes least-loaded — `--kv-overlap-weight` then
has no effect at any value. Nothing else shows it: the server starts, `/health`
is green, requests succeed. Watch for `kv-aware DEGRADED` in the server log and
`infera_cache_locality_skipped_total{reason="no_tokenizer"}`.
```

**Hardware.** 8× MI355X (or MI300X+) on one node for `aggregated`; two such nodes on a
shared RoCE fabric for `disaggregated`. ~700 GB of host RAM if you enable kvd — its L2 arena is
pinned host memory charged to the sidecar's limit.

**Cluster.** Kubernetes 1.28+ with:

```bash
# AMD GPU device plugin — nodes must advertise amd.com/gpu
kubectl get nodes -o custom-columns=NODE:.metadata.name,GPU:.status.allocatable.'amd\.com/gpu'

# the infera operator (provides the InferaDeployment CRD)
helm install infera-operator oci://docker.io/rocm/infera-operator --version 0.1.0 \
  -n infera-system --create-namespace
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

For the `disaggregated` combos, first substitute the two node names:

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

Expect exactly `The capital of France is Paris.` **Garbage or repeated tokens**
means the ROCm DSA indexer env vars did not take effect — check that
`SGLANG_OPT_USE_TILELANG_INDEXER=1`, `SGLANG_OPT_USE_TOPK_V2=0` and
`SGLANG_OPT_USE_JIT_NORM=0` are set on the worker.

GLM-5.2 is a thinking model, so the manifest passes `--reasoning-parser glm45`
and the trace lands in `reasoning_content`, leaving `content` clean. Without it
the answer is still correct but arrives with the chain-of-thought and a raw
`</think>` inlined — right output, unusable shape.

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
| **`aggregated/deploy.yaml` exactly as written** | **validated end-to-end on k3s (MI355X, 8×`amd.com/gpu`)** — see below |
| kvd sidecar + PVC L3 under SGLang | validated (3674 entries / 421 MB), on Qwen3-0.6B — but see the py310 native-tree note above |
| `disaggregated/deploy.yaml`, `model-cache` swapped for per-node hostPath | **validated cross-node** — both roles Ready in ~4 min, 0 restarts, correct answer, and the decode side logged only `Decode batch` (never `Prefill batch`), so the KV handoff was real rather than failing open |
| `disaggregated` + decode-side DP attention | **validated** — `--dp-size 2 --enable-dp-attention` on the decode role only, Ready in ~3 min, 3/3 requests, both attention ranks active. It saw no port collision, but the run predates the randomised scan start in `free_tcp_port_block` and a cross-node pair cannot produce that collision anyway — it needs two engines on one host |
| `disaggregated/deploy.yaml` **as shipped** | **cannot work** — its `model-cache` PVC is ReadWriteOnce on `local-path`, so it is pinned to one node and the other role cannot mount it. See the manifest header |
| `disaggregated-kvd` | not run |

The `aggregated` run used the **stock `lmsysorg/sglang` image** with the overlay
mounted in — no infera-built engine image anywhere in the Pod:

| | |
|---|---|
| router Ready | ~20 s (`infera.server` running from the overlay's py310 tree) |
| worker Ready | ~13 min (408 GiB over NFS, plus DSA graph capture) |
| output | `The capital of France is Paris.` — correct and coherent |
| reasoning | separated into `reasoning_content`, no `</think>` in `content` |

Weights came over NFS here. On local NVMe expect materially less than 13 min;
the comparable Kimi-K3 run loaded 1453 GiB in 502 s from local disk.

## Source

[`examples/recipes/glm5.2/`](.) in [AMD-AGI/Infera](https://github.com/AMD-AGI/Infera)
· [all combos](../README.md) · [overlay build](../../../deploy/overlay)
