# Recipes

Ready-to-run deployments for a specific model, in the shape you want to serve it.
Pick the model, pick the combination, `kubectl apply`.

Every recipe runs the **stock vendor image** with the infera overlay mounted in, so
following an upstream vLLM or SGLang release is an image-tag edit rather than a
rebuild.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} GLM-5.2-MXFP4
:link: glm5.2
:link-type: doc

SGLang · TP8 · MI355X

MLA + DeepSeek Sparse Attention. Needs the ROCm tilelang indexer path.
:::

:::{grid-item-card} Kimi-K3
:link: kimi-k3
:link-type: doc

vLLM · TP8 · MI355X

Multimodal, ~1.5 TB of weights, hybrid Mamba.
:::

:::{grid-item-card} Kimi-K3 optimized (DSpark)
:link: kimi-k3-optimized
:link-type: doc

vLLM · TP8 · MI355X

Same engine commit, optimized FP8 kernels. Optional DSpark speculation: 1.9–2.2×
below concurrency 8, and it crashes above it. Cross-node PD validated.
:::

::::

## The four combinations

Each recipe comes in the same four shapes, composing two independent choices:
**how requests are split across GPUs**, and **whether KV survives past the GPU**.

| Combination | Serving | KV cache | Reach for it when |
|---|---|---|---|
| `mixed` | one worker does prefill and decode | GPU only | the default; the simplest thing that works |
| `mixed + kvd` | one worker does prefill and decode | plus L2 host RAM and L3 on a PVC | requests share long prefixes — a common system prompt, multi-turn chat, RAG |
| `pd` | prefill and decode on separate nodes | GPU only | prefill and decode want different batching |
| `pd + kvd` | prefill and decode on separate nodes | plus kvd on each role | both of the above |

One recipe carries an extra axis rather than a fifth combination: **Kimi-K3
optimized** ships `mixed`, `mixed-dspark` and `pd`, because speculative decoding is
a property of that image's draft model, not a serving topology. `pd` is the same
combination as everywhere else; `mixed-dspark` is the extra axis, and it was not
combined with `pd`.

```{admonition} PD needs a routable RoCE fabric
:class: warning
The prefill→decode KV handoff is RDMA, and there is **no TCP fallback**. Both nodes
must sit on a mutually routable RoCE fabric. On a single box, use `mixed`.
```

## What the overlay provides

The overlay carries one native tree per ABI family and records what each provides,
because the families genuinely differ:

| | `mooncake` (PD KV transport) | `hipfile` (kvd GPU-direct L3) |
|---|---|---|
| `rocm7-py312` (vLLM bases) | yes | yes |
| `rocm7-py310` (SGLang bases) | yes | **no — a vLLM-only path** |

Manifests name what they need through `INFERA_REQUIRE_NATIVE`, and `infera-exec`
refuses to start if the payload cannot supply it. Without that check a missing
piece is silent: the Pod comes up green and serves without the tier you deployed
it for.
