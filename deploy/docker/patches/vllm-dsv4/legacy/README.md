# vllm-dsv4/legacy — deprecated / upstreamed DSv4 patches (not baked into any image)

## Purpose

This directory holds **patches written early in the DSv4 integration (while debugging
against an older vLLM base) that are already resolved upstream on the current verified
stack** — vLLM `0.23.1rc1.dev748+g2dfaae752` + `amd-aiter 0.1.16.post2` — and are
therefore no-ops. They are **not traversed/executed by any Dockerfile**; kept only for
archival and provenance.

> Background: early bring-up was debugged on a private pre-baked image + an older
> Dockerfile, which produced several patches. After rebuilding from the open-source
> `vllm/vllm-openai-rocm:nightly` and verifying markers on real hardware, these turned
> out to be upstream already. They were moved out of the active directory to keep the
> dsv4 image minimal and clean.

## Criteria (marker check inside the `infera/engine-vllm:dsv4` image)

| patch | target file | state on dev748 | verdict |
|---|---|---|---|
| patch_dsv4_aiter_moe.py | rocm_aiter_moe.py | upstream carries `SWIGLUOAI`/`activation_interleave`, defaults gate_mode="" for Silu | no-op (patch self-detects upstream and skips) |
| patch_dsv4_mhc_aiter.py | mhc.py | upstream already does `if HAS_AITER_MHC and hidden%256==0: mhc_pre_aiter` (aiter preferred), no `VLLM_MHC_BACKEND` gate | no-op (old anchor `_mhc_backend()` absent on dev748) |
| patch_dsv4_mhc_aiter.diff | mhc.py | same as above (unified-diff companion to the .py, not executed) | reference only |

## Notes

- These patches may still apply on an **older base** (e.g. earlier vLLM dev301/dev424,
  or an image lacking the upstream fixes above); reference this directory manually to
  reproduce on such a base.
- `patch_dsv4_mhc_aiter`'s functionality is upstream via a **different implementation**
  (aiter as the MHC default rather than an env gate) — "functionally upstream, different
  form" — so the monkey-patch anchor mismatches on dev748 and auto-skips.
- Active DSv4 patches live in the parent directory `../` (the moriio_dsv4 trio).
