# Recipes

Ready-to-run deployments for a specific model, in the shape you want to serve it.
Pick the model, pick the combination, `kubectl apply`.

Recipes run the **stock vendor image** with the infera overlay mounted in, so
following an upstream vLLM or SGLang release is an image-tag edit rather than a
rebuild. The one exception is GLM-5.2-FP8 on gfx942, which needs a rebuilt native
library that no mounted payload can supply; its page says why.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} GLM-5.2-MXFP4
:link: glm5.2
:link-type: doc

SGLang · TP8 · MI355X

MLA + DeepSeek Sparse Attention. Needs the ROCm tilelang indexer path.
:::

:::{grid-item-card} GLM-5.2-FP8 on gfx942
:link: glm5.2-fp8-gfx942
:link-type: doc

SGLang · TP8/DP8 · 1–2 × MI300X

DP-attention and MTP, optionally split over Mooncake RDMA. The one recipe that
builds an engine image instead of mounting the overlay.
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

Same engine commit, optimized FP8 kernels. Optional DSpark speculation — 2.60× at
c=4, decaying to parity by c=32. Cross-node PD validated, with and without
speculation.
:::

::::

:::{admonition} The directory is `aggregated`, the API field still says `mixed`
:class: note
`aggregated` / `disaggregated` is the vocabulary these directories use. The `role:`
field inside each manifest is an API value the operator consumes — `mixed`,
`prefill`, `decode` — and is deliberately unchanged, since renaming it would break
deployed configurations. So `aggregated/deploy.yaml` legitimately contains
`role: mixed`.
:::

## The four combinations

Recipes come in the same four shapes, composing two independent choices:
**how requests are split across GPUs**, and **whether KV survives past the GPU**.

| Combination | Serving | KV cache | Reach for it when |
|---|---|---|---|
| `aggregated` | one worker does prefill and decode | GPU only | the default; the simplest thing that works |
| `aggregated + kvd` | one worker does prefill and decode | plus L2 host RAM and L3 on a PVC | requests share long prefixes — a common system prompt, multi-turn chat, RAG |
| `disaggregated` (PD) | prefill and decode on separate nodes | GPU only | prefill and decode want different batching |
| `disaggregated + kvd` | prefill and decode on separate nodes | plus kvd on each role | both of the above |

One recipe carries an extra axis rather than a fifth combination: **Kimi-K3
optimized** ships `aggregated`, `aggregated-dspark`, `disaggregated` and `disaggregated-dspark`, because
speculative decoding is a property of that image's draft model, not a serving
topology. It composes with `aggregated` and `disaggregated` independently, which is why there are
four rather than a fifth combination.

Where a recipe's four shapes differ in more than those two axes, its page says so.
**GLM-5.2-FP8 on gfx942** is the one that does: its aggregated shapes drop
`hostNetwork`, `privileged` and `/dev/infiniband` outright, because those exist
only to carry KV over RDMA and nothing crosses the fabric when one worker both
prefills and decodes. Its `aggregated + kvd` arm also drops MTP, because on a
worker that both prefills and decodes, MTP and the tiered cache deadlock each
other; its page and its manifest both explain that at the top.

```{admonition} PD needs a routable RoCE fabric
:class: warning
The prefill→decode KV handoff is RDMA, and there is **no TCP fallback**. Both nodes
must sit on a mutually routable RoCE fabric. On a single box, use `aggregated`.
```

:::{admonition} Checking RoCE reachability: bind the source rail
:class: tip
This fabric is **routed L3 RoCEv2**: every NIC gets its own /64, so no two hosts
ever share a subnet — by design, not a fault. An unbound `ping6` picks a default
source, leaves via the wrong rail and reports "No route", which reads as "these
nodes cannot do PD". Bind the matching local rail instead:

```bash
ping6 -I <local-rail-ULA> <peer-rail-ULA>
```

Mooncake binds its QP to a device and GID index, so it takes the working path the
naive ping does not. On this fleet all 8 rails answer at ~0.1 ms once bound.
:::


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
