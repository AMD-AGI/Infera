# Feature matrix

```{admonition} One-pager
:class: tip
At a glance, this table shows which Infera features are available on each
engine. **Legend:** ✅ supported · 🚧 work in progress · blank = not supported.
```

## Overview

| Feature | vLLM | SGLang | ATOM | Source |
| :--- | :---: | :---: | :---: | :--- |
| **Disaggregated Serving (PD)** | ✅ | ✅ | ✅ | [PD Disaggregation][pd] |
| **KV-Aware Routing** | ✅ | ✅ | ✅ | [KV-Aware Routing][kv] |
| **KV-Aware Routing + DP-Attention** | ✅ | ✅ | ✅ | [KV-Aware Routing][kv] |
| **Tiered KV Cache Offload (kvd)** | ✅ | 🚧 | 🚧 | [KV Cache Offload][kvd] |
| **SLA Planner (PD autoscaling)** | ✅ | ✅ | ✅ | [SLA Planner][sla] |
| **Multimodal (image / audio / video)** |  |  |  |  |

KV-cache offload (`kvd`), including AIC GPU-Direct, is **vLLM-only** today.

The SLA planner is engine-agnostic — it reads the server's metrics and resizes
the pools — but it only covers **disaggregated** deployments, and needs a
profiling sweep of your own model to work from.

[pd]: ./pd_disaggregation.md
[kv]: ./kv_aware_routing.md
[kvd]: ./kv_cache_offload.md
[sla]: ./sla_planner.md
