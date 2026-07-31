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
| **Tiered KV Cache Offload (kvd)** | ✅ | ✅ | 🚧 | [KV Cache Offload][kvd] |
| **Multimodal (image / audio / video)** |  |  |  |  |

`kvd` runs on vLLM (`InferaKvdConnector`) and on SGLang (`InferaKvdBackend`, a
`HiCacheStorage` backend). The **AIC GPU-Direct** read path is vLLM-only; SGLang
reads through the daemon's POSIX path.

On SGLang, `kvd` on a **PD decode leg** additionally requires KV events to be
on: a decode leg sets `disable_radix_cache=True` itself, and SGLang rejects that
alongside `--enable-hierarchical-cache`. Infera re-enables the decode radix
cache automatically — but only when KV events are enabled. See
[KV Cache Offload][kvd].

[pd]: ./pd_disaggregation.md
[kv]: ./kv_aware_routing.md
[kvd]: ./kv_cache_offload.md
