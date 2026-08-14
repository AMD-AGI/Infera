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
| **Multimodal (image / audio / video)** |  |  |  |  |
| **Request Migration** | ✅ | ✅ | ✅ | [Request Migration][mig] |

KV-cache offload (`kvd`), including AIC GPU-Direct, is **vLLM-only** today.
Request migration does not depend on the engine, but does require the NATS
request transport and mixed (non-PD) workers; it is off unless enabled.

[pd]: ./pd_disaggregation.md
[kv]: ./kv_aware_routing.md
[kvd]: ./kv_cache_offload.md
[mig]: ./graceful_shutdown.md#request-migration
