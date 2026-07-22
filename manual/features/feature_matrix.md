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
| **Tiered KV Cache Offload (kvd)** | ✅[^gpu-direct] | ✅ | 🚧 | [KV Cache Offload][kvd] |
| **DeepSeek-V4 on MI325X (gfx942)** | ✅[^dsv4-fp4] | ✅[^dsv4-fp8] | ✅[^dsv4-fp8] | [DeepSeek-V4 on MI325X][dsv4mi325] |
| **Multimodal (image / audio / video)** |  |  |  |  |

[^gpu-direct]: The AIC GPU-Direct path is currently supported only by vLLM.
[^dsv4-fp4]: vLLM runs FP4 dsv4 natively; SGLang/ATOM require FP8 (FP4 fails fast).
[^dsv4-fp8]: SGLang/ATOM run FP8 dsv4 (Flash needs MTP, applied automatically); vLLM is FP4-only for dsv4.

[pd]: ./pd_disaggregation.md
[kv]: ./kv_aware_routing.md
[kvd]: ./kv_cache_offload.md
[dsv4mi325]: ./mi325-deepseek-v4.md
