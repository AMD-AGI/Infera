# Compatibility matrix

This page lists the exact versions Infera **0.1.0** has been validated against.
Everything here is what we have tested — combinations not listed are outside the
validated set for this release, not necessarily unsupported.

Two ROCm baselines are covered. **ROCm 7.2** is the release baseline: all three
engines, every scenario. **ROCm 10.1** is an early-enablement preview on nightly
vendor bases, validated over a deliberately narrow scope — see
[ROCm 10.1 preview](#rocm-101-preview) for exactly how narrow.

```{admonition} How to read this page
:class: note
Where a component is pinned to a specific image tag, use that exact artifact. The
validated engine images bundle the Infera connector, the `sitecustomize` hook,
and the RDMA transport shims on top of the vendor ROCm base — so the tag is the
tested unit, not just the engine version inside it.
```

## Engines

Infera orchestrates one or more engines. Install at least one. The 0.1.0 release
is validated against these engine versions and images.

::::{tab-set}

:::{tab-item} ROCm 7.2 — release baseline

| | vLLM | SGLang | ATOM |
| --- | --- | --- | --- |
| **Engine version** | 0.25.1 | 0.5.15.post1 | 0.1.4 |
| **Docker image** | `vllm/vllm-openai-rocm:v0.25.1` (ROCm base, digest-pinned) | `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x` | `rocm/atom:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0_atom0.1.4_20260612` |
| **ROCm (image base)** | 7.2.3 | 7.2.0 | 7.2.4 |
| **OS (image base)** | Ubuntu 22.04.5 | Ubuntu 22.04.5 | Ubuntu 24.04.4 |
| **Python (image base)** | 3.12.13 | 3.10.12 | 3.12.3 |
| **PyTorch (image base)** | 2.11.0 | 2.9.1 | 2.10.0 |
| **Build date** | 2026-07-12 | 2026-07-15 | 2026-06-12 |

:::

:::{tab-item} ROCm 10.1 — preview

| | vLLM | SGLang | ATOM |
| --- | --- | --- | --- |
| **Engine version** | 0.26.1.dev0 (`g568afb3a1`) | 0.5.15.post1 (`ge35a33b34a`) | not validated |
| **Docker image** | `rocm/ufb-private:vllm-0.26.0-…-rocm10.1.0a20260807-568afb3a1` | `rocm/ufb-private:sglang-v0.5.15.post1-…-rocm10.1.0a20260811-e35a33b34a` | — |
| **ROCm (image base)** | 10.1.0 (nightly `a20260807`) | 10.1.0 (nightly `a20260811`) | — |
| **OS (image base)** | Ubuntu 24.04.4 | Ubuntu 24.04.4 | — |
| **Python (image base)** | 3.14.7 | 3.14.6 | — |
| **PyTorch (image base)** | 2.12.0+rocm10.1.0a20260807 | 2.12.0+rocm10.1.0a20260811 | — |
| **Build date** | 2026-08-08 | 2026-08-11 | — |

These are nightly tags in an access-gated repository, so the tag alone is not a
version. Build against the digest:

```text
rocm/ufb-private:vllm-0.26.0-ubuntu24.04-py3.14-nightlies-device-all-cdna-rocm10.1.0a20260807-568afb3a1@sha256:11657b464b9c2b7f285a68ca1f622f2cdf3c5ef30a48cc6e560c4141acca8556
rocm/ufb-private:sglang-v0.5.15.post1-ubuntu24.04-py3.14-nightlies-device-all-rocm10.1.0a20260811-e35a33b34a@sha256:e87ecda2624687727d253e0c74863dd7f5008759ab09a9f7bc14e3360de181b6
```

:::

::::

OS / Python / PyTorch above were read from inside each validated base image (not
the tag). See the platform table below for the baseline that applies across all
engines.

```{admonition} Engine images bundle more than the engine
:class: tip
The validated engine images layer the Infera connector, the `sitecustomize`
hook, and the Mooncake / ionic RDMA shims on top of the vendor ROCm base. For
serving, prefer the prebuilt image over a manual `pip install` of the engine.
See [Deployment → Engine images](../serving/deployment.md).
```

## Platform

| Component | Validated version |
| --- | --- |
| GPU | AMD Instinct MI355X (`gfx950`) |
| ROCm | 7.2 — plus a 10.1 nightly preview ([below](#rocm-101-preview)) |
| Operating system | Ubuntu 24.04 (Linux x86-64) |
| Python | 3.10+ |
| Docker | Required — runs `etcd` in dev and builds engine images |
| RDMA NIC | AMD AINIC — required only for cross-node prefill-decode |

## ROCm 10.1 preview

ROCm 10.1 changes how ROCm itself ships. It arrives as a **pip wheel**
(`_rocm_sdk_devel` under site-packages, with `ROCM_HOME` pointing at it) and the
bases carry **no `/opt/rocm`**; Python moves to 3.14. The Infera engine images
restore `/opt/rocm` as a symlink, because third-party code that hardcodes the
path degrades silently rather than erroring.

Validated with **gpt-oss-120b** on 2 × 8 MI355X (`gfx950`), both engines, both
serving topologies:

| Scenario | vLLM | SGLang |
| --- | --- | --- |
| Aggregated — prefill + decode in one worker | ✅ PASS · 169 s | ✅ PASS · 342 s |
| Disaggregated — prefill and decode on separate nodes | ✅ PASS · 166 s | ✅ PASS · 367 s |

Both disaggregated runs were confirmed to carry KV over RDMA rather than
silently falling back to TCP, and all four runs' correctness probes passed.

```{admonition} Outside the ROCm 10.1 preview scope
:class: warning
- **Models** — gpt-oss-120b only. GLM-5.2 and DeepSeek-v4 are not covered: the
  10.1 SGLang base is v0.5.15.post1, while the DSA patch set Infera applies for
  those models is pinned against v0.5.17.
- **aiter sampling (vLLM)** — every 10.1 vLLM run passes `--use-fp64-gumbel`.
  aiter's top-k/top-p kernel does not build against 10.1's hipCUB, so sampling
  falls to the PyTorch implementation. This is an upstream aiter issue; MXFP4
  MoE still runs through aiter.
- **kvd GPU-direct tier** — not built into the 10.1 images.
- **ATOM** — not exercised on 10.1.
- **`gfx942`** — the same bases and scenarios also passed on 8 × MI300X, but
  that path needs image-side patches and aiter overrides which are not part of
  this release's images.
```

## Infera

| Package | Validated version |
| --- | --- |
| `amd-infera` | 0.1.0 |
