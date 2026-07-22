---
sd_hide_title: true
---

# ROCm Infera Manual

```{div} sd-text-center sd-fs-2 sd-font-weight-bold
ROCm Infera
```

```{div} sd-text-center sd-fs-5 sd-text-muted
A lightweight inference-serving layer for AMD ROCm GPUs.
```

Infera (the `infera` package) puts an **OpenAI-compatible endpoint** in front
of one or many model workers, and routes each request to the best worker — by
cache locality, by load, or by splitting prefill and decode across GPUs. It runs
**vLLM, SGLang, or ATOM** underneath, and adds an AMD-native **tiered KV cache**
that spills from GPU memory to RAM, NVMe, and the network.

If you have ever run `vllm serve`, you already know 80% of this. Infera is the
thin orchestration layer that turns one model server into a routed fleet.

---

## Start here

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 🚀 Quickstart
:link: getting_started/quickstart
:link-type: doc
Five minutes from `pip install` to a served model and a working `curl`.
:::

:::{grid-item-card} 🧭 Overview
:link: getting_started/overview
:link-type: doc
What Infera is, the three moving parts, and when to reach for it.
:::

:::{grid-item-card} 🔌 Server & router
:link: components/server
:link-type: doc
The OpenAI-compatible endpoints and how requests flow through the router.
:::
::::

## Components

The moving parts, one page each — what they are and how to run them.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🔌 Server & router
:link: components/server
:link-type: doc
OpenAI / Anthropic endpoints + the router (modes, policies).
:::

:::{grid-item-card} ⚙️ Engines
:link: components/engines
:link-type: doc
vLLM / SGLang / ATOM workers — launch, TP, DP-attention.
:::

:::{grid-item-card} 🗄️ KV-Cache Management
:link: components/kvd
:link-type: doc
kvd's tier design, offload/onboard mechanism, and on-disk format.
:::

:::{grid-item-card} ☸️ Operator
:link: components/operator
:link-type: doc
The Kubernetes CRD that reconciles server + prefill/decode pools.
:::
::::

## Feature one-pagers

Each of these is a single, self-contained page — read only the one you need.

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 🎯 KV-aware routing
:link: features/kv_aware_routing
:link-type: doc
Route to the worker that already holds your prompt's prefix.
:::

:::{grid-item-card} ✂️ PD disaggregation
:link: features/pd_disaggregation
:link-type: doc
Run prefill and decode on separate GPUs (or nodes) and stitch them with a KV transfer.
:::

:::{grid-item-card} 🗄️ KV-Cache Offload
:link: features/kv_cache_offload
:link-type: doc
Spill KV to RAM / NVMe / network — the usage modes and per-request control.
:::

:::{grid-item-card} 🧩 DeepSeek-V4 on MI325X
:link: features/mi325-deepseek-v4
:link-type: doc
Run DeepSeek-V4 (Pro/Flash, FP4/FP8) on gfx942 — the support matrix and auto-tuned knobs.
:::

::::
