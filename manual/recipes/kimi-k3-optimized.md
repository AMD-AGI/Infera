# Kimi-K3 (optimized build)

Kimi-K3 on MI355X using `johnqin2025/kimi-k3-dspark` — an image carrying an FP8
pre-route / shared-expert kernel cluster, an FP8 latent-MoE tail, a fused MLA
output gate and a tri-projection dispatch — plus an optional **DSpark** draft model
for speculative decoding.

A separate entry from [Kimi-K3](kimi-k3) rather than a combination on it, because
it is a different artifact. Same vLLM commit (`g5f76ae224`) as the stock image, so
the delta is kernels, not engine version.

```{admonition} Every manifest here is a template — fill in the placeholders first
:class: warning
A bare `kubectl apply` on these files is **accepted by the API server** (the
placeholder passes CRD validation) and then fails minutes later at mount time.

| Placeholder | In | How to find it |
|---|---|---|
| `<MODEL_DIR>` | Mixed, Mixed + DSpark | site-specific, **no default** — see below |
| `<NODE>` | Mixed, Mixed + DSpark | a node with 8 free GPUs *that has the weights at that path* |
| `<PREFILL_NODE>`, `<DECODE_NODE>` | PD, PD + DSpark | two nodes on a routable RoCE fabric |
| `<PREFILL_MODEL_DIR>`, `<DECODE_MODEL_DIR>` | PD, PD + DSpark | per node — **they need not be equal** |

**The model path is site-specific and a mixed fleet may not agree with itself.** On
the fleet these were validated on, the two GPU nodes use different paths for the
same weights. Candidates usually look identical (same shard count, same layout)
while only one is local storage, so confirm the one you pick by mounting **that
directory** in a throwaway pod on the node and reading `df -PT` — see the repo
README for the commands. Mounting a parent instead reports the root filesystem and
will call a network-mounted weights directory local.

Leaving a placeholder in has two different outcomes, neither obvious. `<MODEL_DIR>`
creates a Pod that fails at mount time. `<NODE>` creates **no Pod at all** —
`apply` prints `created`, `get pods` shows nothing, and the only error is in the
operator's log in the `infera-system` namespace.

`<NODE>` exists because both the server and the worker mount the weights by
hostPath and are scheduled independently — without it they can land on different
nodes.
```

## 1. Choose the combination

::::{tab-set}

:::{tab-item} Mixed
:sync: mixed

8 GPUs on one node, no speculative decoding. **The default**, and the best tokens
per GPU of the four.

```bash
sed -e 's|<MODEL_DIR>|/mnt/local-nvme/models|' -e 's|<NODE>|node-a|' \
    examples/recipes/kimi-k3-optimized/mixed/deploy.yaml | kubectl apply -f -
```

| Placeholder | What to put there |
|---|---|
| `<MODEL_DIR>` | Directory on that node containing `Kimi-K3/`, mounted into the pod at `/models`. A `hostPath` — the node must hold the weights locally. |
| `<NODE>` | Hostname of the node (`kubernetes.io/hostname`). Pins **both** the server and the worker to it: both mount the weights by `hostPath` and are scheduled independently, so without this they can land on different nodes and the one without the weights fails. |

Start here unless you have a specific reason not to. The other three trade GPUs or
added complexity for latency, and one of them needs a second node.
:::

:::{tab-item} Mixed + DSpark
:sync: mixed-dspark

Same 8 GPUs, plus speculative decoding: a block-diffusion draft model that produces
7 tokens in one parallel pass, verified against the target.

```bash
sed -e 's|<MODEL_DIR>|/mnt/local-nvme/models|' -e 's|<NODE>|node-a|' \
    examples/recipes/kimi-k3-optimized/mixed-dspark/deploy.yaml | kubectl apply -f -
```

Same two placeholders as **Mixed**, but `<MODEL_DIR>` must also contain
`Kimi-K3-DSpark/` alongside `Kimi-K3/`.

The draft is a community checkpoint,
[`Inferact/Kimi-K3-DSpark`](https://huggingface.co/Inferact/Kimi-K3-DSpark) — there
is no official Moonshot draft for Kimi-K3 — and must be downloaded alongside the
target into the same `<MODEL_DIR>`.

Speculation earns its keep at **low concurrency**, where there is idle capacity to
spend on drafting. As concurrency rises the verify batch grows, acceptance falls,
and the advantage decays; at high concurrency plain **Mixed** is the better choice.
Where that crossover sits depends on your workload — measure it on yours rather
than taking a number from here.

```{admonition} The old concurrency ceiling is gone
:class: note
On the `…-20260801` image this died at `c>=16` with `AssertionError: AiterMLA
flattened verify requires a uniform decode query len`. The image pinned here fixes
it — c=16/32/64 all complete with 0 restarts and the assertion never appears — and
speculation is genuinely still on (`num_spec_tokens=7`, CUDA-graph captured,
`running the draft eagerly` count 0), so this is a fix rather than speculation
being quietly disabled.

If you pin the older digest, the old ceiling still applies to you.
```
:::

:::{tab-item} PD
:sync: pd

Prefill and decode on **separate nodes**, 8 GPUs each, KV handed between them over
RDMA.

```bash
sed -e 's|<PREFILL_NODE>|node-a|' -e 's|<DECODE_NODE>|node-b|' \
    -e 's|<PREFILL_MODEL_DIR>|/mnt/local-nvme/models|' \
    -e 's|<DECODE_MODEL_DIR>|/mnt/array/models|' \
    examples/recipes/kimi-k3-optimized/pd/deploy.yaml | kubectl apply -f -
```

| Placeholder | What to put there |
|---|---|
| `<PREFILL_NODE>` | Hostname of the node that runs **prefill** (`kubernetes.io/hostname`). It processes the prompt and produces the KV cache. The router pod is placed here too. |
| `<DECODE_NODE>` | Hostname of the node that runs **decode** — it receives that KV over RDMA and generates the tokens. Must be a *different* node, and the two must be able to reach each other over the RoCE fabric. |
| `<PREFILL_MODEL_DIR>` | Directory **on the prefill node** containing `Kimi-K3/`, mounted into the pod at `/models`. |
| `<DECODE_MODEL_DIR>` | The same, **on the decode node**. Frequently a different path — the nodes do not have to agree, and on a mixed fleet they often do not. |

Both directories are `hostPath` mounts, so each node reads its **own local copy**;
there is no shared volume. Neither path is discoverable from this page — find each
one on its own node, and confirm it is local storage rather than a network mount,
before substituting.

```{admonition} Twice the hardware, and it never wins per GPU
:class: warning
PD does not improve tokens per GPU at any concurrency measured here. What it buys
is **headroom** past what a single node can serve, and **lower latency** at high
concurrency, because decode is no longer interleaved with prefill on the same GPUs.
Below that it is a straight loss on both counts — use **Mixed**.
```

Requires both nodes on a mutually routable RoCE fabric: the KV handoff is RDMA and
there is no TCP fallback. Each node reads its **own local** copy of the weights, and
the two paths need not be the same.
:::

:::{tab-item} PD + DSpark
:sync: pd-dspark

PD's topology plus speculation. **The lowest latency of the four** — the decode role
is not competing with prefill for the same GPUs, so drafting has capacity to use.

```bash
sed -e 's|<PREFILL_NODE>|node-a|' -e 's|<DECODE_NODE>|node-b|' \
    -e 's|<PREFILL_MODEL_DIR>|/mnt/local-nvme/models|' \
    -e 's|<DECODE_MODEL_DIR>|/mnt/array/models|' \
    examples/recipes/kimi-k3-optimized/pd-dspark/deploy.yaml | kubectl apply -f -
```

The placeholders are the same four as **PD** above, with one addition to the
requirement: `Kimi-K3-DSpark/` must sit alongside `Kimi-K3/` in **both**
directories, because both roles load the draft.

```{admonition} `--speculative-config` goes on BOTH roles, and that is not redundancy
:class: warning
The prefiller never samples, so a draft there looks like dead weight. That
configuration was tried and it fails two independent ways:

**Layer lists disagree.** vLLM continues the target's layer numbering into the
draft, so a speculating decoder registers 98 layer names and sends them all to a
prefiller that has only the target's — `KeyError: 'model.layers.93.self_attn'`.

**Block counts disagree.** Fixing the layer lists lands straight on
`pulling kv_caches ... failed: P num blocks less than D`. Speculation recomputes
`max_num_scheduled_tokens` to reserve draft slots, so the decoder's per-request
block accounting differs from a prefiller that does not know speculation is
happening. No layer filtering fixes that — both sides must compute it the same way.

Loading the draft on the prefiller is the **price of that agreement**, not an
oversight. It is never run there, and **both nodes need the draft on disk**.

Both failures **hang rather than error**: all pods Ready, health checks green,
restarts 0, no inference logged, and the client waits until its own timeout.
```
:::

::::

## 2. Prerequisites

```bash
kubectl get nodes -o custom-columns=NODE:.metadata.name,GPU:.status.allocatable.'amd\.com/gpu'

# every manifest hardcodes `namespace: infera`, and nothing else creates it
kubectl create namespace infera --dry-run=client -o yaml | kubectl apply -f -

# on k3s, helm needs KUBECONFIG spelled out — kubectl finds it implicitly, helm does not
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
helm upgrade --install infera-operator deploy/operator/helm/infera-operator \
  -n infera-system --create-namespace

hf download moonshotai/Kimi-K3      --local-dir <MODEL_DIR>/Kimi-K3
hf download Inferact/Kimi-K3-DSpark --local-dir <MODEL_DIR>/Kimi-K3-DSpark   # DSpark only
```

**`<MODEL_DIR>` must be local NVMe.** Kimi-K3's 96 shards load in ~8 min from local
disk and ~95 min from NFS — and the slow path does not merely run late, it exceeds
the ready timeout, so the worker restarts mid-load and never finishes.

For the PD combinations each node needs its **own** copy, and the paths need not
match. On the fleet this was validated on, the tempting common mount (`/mnt/shared`)
was an NFS export of the *other* node's array; `df -hT <dir>` on each node and take
the local device, not the convenient common name.

First start is 10–14 min: the image rebuilds its AITER JIT modules in-container on
top of weight load and CUDA-graph capture.

## 3. Smoke test

```bash
kubectl -n infera port-forward svc/<name>-server 18000:8000 & PF=$!
sleep 3; kill -0 $PF 2>/dev/null || { echo "port-forward failed — try another local port"; exit 1; }

# --max-time is not optional: this system's failure mode is a HANG, not an error.
curl -s --max-time 300 localhost:18000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"kimi-k3","messages":[{"role":"user","content":"What is the capital of France?"}],
       "max_tokens":1024}' | jq -r '.choices[0].message.content'
```

`max_tokens: 1024`, not 200: the model emits a variable-length reasoning preamble,
and at 200 roughly one request in four returns `finish_reason: "length"` having
never reached the answer — which reads like a broken deployment on a healthy one.

Keep `max_tokens` generous — this model spends 70+ tokens on that sentence.

On the DSpark manifests, confirm speculation engaged rather than inferring it from
throughput later:

```bash
kubectl -n infera logs <worker-or-decode-pod> -c main | grep -c 'running the draft eagerly'   # must be 0
```

On the PD manifests, a correct answer proves nothing on its own — a handoff that
fails open just re-prefills locally and returns the same text, faster. Check the
decode side: `External prefix cache hit rate` near 100% with `Avg prompt
throughput` near zero.

## 4. Settings that are not optional

| Setting | Why |
|---|---|
| the `KIMI_K3_*` / `VLLM_ROCM_*` env block | selects the optimized kernels. Without them the MoE asks aiter for a kernel that was never generated — `Invalid FlyDSL kernel name: flydsl_moe1_...` — because there is no Kimi-K3 `tuned_fmoe.csv` |
| `VLLM_ROCM_USE_KIMI_K3_PREROUTE_BF16=0` | must be `0`. A `1` makes the pre-route dispatch take BF16 and shadows the FP8 kernel cluster this image exists for, costing a large fraction of throughput. It is silent in every other respect — the worker starts normally, logs no warning, and median per-token latency barely moves, so latency monitoring will not show it. Only aggregate throughput does |
| `attention_backend: ROCM_AITER_MLA` | the ROCm counterpart of the upstream quick-start's `FLASHINFER_MLA`. Dropping the key instead of translating it is not the fix |
| `--gpu-memory-utilization 0.88` | the draft's weights land after the KV budget is computed; `0.95` dies with 998 MB free trying to allocate 2.32 GiB |
| `INFERA_ENGINE_READY_TIMEOUT=7200` | the 1800 s default is impossible on slow storage, and the worker then restarts mid-load forever — which reads as a crash loop, not as slow storage |
| PD: each node needs its **own local** copy | the `model` volume is a `hostPath`, and the paths need not match between nodes. The fleet's obvious shared mount may be an NFS export of the peer's array — one side then loads for ~95 min and restarts forever |
| PD: never point a misbehaving client at it | the engine validates **after** prefill, so a rejected request has already had its KV computed and queued. 84 requests rejected for one bad field left 424 aborted Mooncake transfers and stalled valid traffic for ~20 minutes with `MooncakeXferMetadata transfer failed: Resource temporarily unavailable`, which reads exactly like a broken fabric |

`ibv_devices` is **not installed** in this image; reading its "not found" as "no
RDMA devices" produced two false TCP-fallback diagnoses here. Ask
`ibv_get_device_list` instead — see the repo README for the one-liner.

## 5. Validation status

Every combination was run end-to-end on the image it pins, with placeholders
substituted (the files are templates). and every
number on this page comes from the same image — Do not carry performance
expectations across a base-image bump: between the two images of this model the
same manifest moved substantially, and in opposite directions at different
concurrencies, so no single correction factor exists.

| What | Status |
|---|---|
| `mixed` | validated |
| `mixed-dspark` | validated; 0 restarts and the concurrency assertion never appears |
| `pd` | validated cross-node; handoff confirmed on the decode side at every concurrency exercised |
| `pd-dspark` | validated cross-node; handoff confirmed on the decode side; needs `--speculative-config` on both roles |
| kvd combinations | not built for this image |
| fp8 KV cache | not measured |

## Source

[`examples/recipes/kimi-k3-optimized/`](https://github.com/AMD-AGI/Infera/tree/main/examples/recipes/kimi-k3-optimized)
· [all recipes](index)
