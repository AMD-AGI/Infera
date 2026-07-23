# SGLang PD (mooncake / mori) cross-node bring-up on v0.5.15.post1 base

**Ran:** 2026-07-23
**Author:** yihou (via Claude Code)
**Status:** ✅ PASS — both mooncake and mori cross-node PD verified after the mooncake HIP-gate fix.

## Goal

Validate DeepSeek-V4-Pro SGLang **prefill/decode disaggregation** (PD, 1P1D
cross-node) on the newly-upgraded base image
`lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`, over **both** KV-transfer
backends — **mooncake** and **mori** — the infera way (`infera.server` router +
`infera.engine.sglang` + etcd auto-pairing). Follows the repo's
`examples/deepseek_v4/engine/pd_{mooncake,mori}/sglang` kit.

This surfaced (and fixed) a real regression the base bump introduced in mooncake.

**Success criteria:** both legs reach ready, etcd auto-pairs P/D, and a completion
routed through `infera.server` returns a coherent answer (KV actually transferred
prefill→decode over RDMA).

## Result

| Backend | Config | Cross-node smoke | Verdict |
|---|---|---|---|
| mooncake | 1P1D, chi2879→chi2865, DSv4-Pro fp8 tp8 | `Paris` / `1..10` (after HIP-gate fix) | ✅ |
| mori | 1P1D, chi2879→chi2865, DSv4-Pro fp8 tp8 | `Paris` / `1..10` | ✅ |
| mooncake (unpatched) | same | 500 `hipIpcOpenMemHandle failed` | ❌ (root-caused) |

**Headline:** the v0.5.15.post1 base bundles a newer mooncake (upstream #2682)
that installs an **unconditional HIP transport**, which hijacks the cross-node KV
transfer and fails (`hipIpcOpenMemHandle` can't open a peer handle across hosts).
Fix = gate the HIP install behind `MC_ENABLE_HIP_TRANSPORT` (default OFF) + rebuild
mooncake in the sglang image. mori uses a different transport and was unaffected.

## How to reproduce

See `REPRODUCE.md`. TL;DR: build the HIP-gated sglang image, put it on both nodes,
`up.sh` a 1P1D PD (mooncake or mori), poll ready, smoke through the router.

## Folder map

- `REPRODUCE.md` — step-by-step, copy-pasteable
- `environment.md` / `environment_nodes.txt` — exact HW/SW/RDMA the result came from
- `scripts/` — the launch/poll/smoke/teardown scripts used verbatim
- `patches/` — the mooncake HIP-gate fix (`patch_mooncake_sglang.sh`) + fix commit
- `notes.md` — root-cause story, dead-ends, gotchas (the debugging gold)
- `results/` — smoke transcripts + the diagnostic evidence
