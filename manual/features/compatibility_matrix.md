# Compatibility matrix

This page lists the exact versions Infera **0.1.0** has been validated against.
Everything here is what we have tested — combinations not listed are outside the
validated set for this release, not necessarily unsupported.

Three ROCm baselines are covered. **ROCm 7.2** is the release baseline: all three
engines, every scenario. **ROCm 10.1** and **ROCm 10.2** are early-enablement
previews on pre-release vendor bases, each validated over a deliberately narrow
scope — see [ROCm 10.1 preview](#rocm-101-preview) and
[ROCm 10.2 preview](#rocm-102-preview) for exactly how narrow.

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
| **Engine version** | 0.25.1 | 0.5.18 (MI355X) / 0.5.16 (MI325X) | 0.1.4 |
| **Docker image** | `vllm/vllm-openai-rocm:v0.25.1` (ROCm base, digest-pinned) | `lmsysorg/sglang:v0.5.18-rocm720-mi35x` / `lmsysorg/sglang:v0.5.16-rocm720-mi30x` | `rocm/atom:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0_atom0.1.4_20260612` |
| **ROCm (image base)** | 7.2.3 | 7.2.0 | 7.2.4 |
| **OS (image base)** | Ubuntu 22.04.5 | Ubuntu 22.04.5 | Ubuntu 24.04.4 |
| **Python (image base)** | 3.12.13 | 3.10.12 | 3.12.3 |
| **PyTorch (image base)** | 2.11.0 | 2.9.1 | 2.10.0 |

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

:::{tab-item} ROCm 10.2 — preview

| | vLLM | SGLang | ATOM |
| --- | --- | --- | --- |
| **Engine version** | not validated | 0.5.21 (`gaaaf7c860d`) | not validated |
| **Docker image** | — | `rocm/ufb-private:sglang-v0.5.21-…-rocm10.2.0rc0-aaaf7c860d` | — |
| **ROCm (image base)** | — | 10.2.0 (pre-release `rc0`) | — |
| **OS (image base)** | — | Ubuntu 24.04.5 | — |
| **Python (image base)** | — | 3.14.10 | — |
| **PyTorch (image base)** | — | 2.14.0+rocm10.2.0rc0 | — |
| **Build date** | — | 2026-10-29 | — |

This is a pre-release tag in an access-gated repository, so the tag alone is not
a version. Build against the digest:

```text
rocm/ufb-private:sglang-v0.5.21-ubuntu24.04-py3.14-prereleases-device-all-rocm10.2.0rc0-aaaf7c860d@sha256:3385b5c137e4fe6d482d508d80ce661387a905b3dc60983a5e8bd5113b5189ea
```

The Infera engine image for this base is built from
`deploy/docker/Dockerfile.sglang.rocm102`, not `Dockerfile.sglang`.

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
| ROCm | 7.2 — plus a 10.1 nightly preview ([below](#rocm-101-preview)) and a 10.2 pre-release preview ([below](#rocm-102-preview)) |
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

## ROCm 10.2 preview

ROCm 10.2 keeps 10.1's packaging: ROCm ships as a **pip wheel** (`rocm-sdk-*`
`10.2.0rc0`, `_rocm_sdk_devel` under site-packages) and Python is 3.14. Unlike
the 10.1 bases, the 10.2 SGLang base already carries `/opt/rocm` as a symlink
to `_rocm_sdk_devel`.

The 10.2 SGLang base is **v0.5.21**, between the v0.5.18 and v0.5.22 recipes,
so the engine image has its own Dockerfile, `Dockerfile.sglang.rocm102`. It
applies the v0.5.22 patch set with these differences:

- **GLM NextN fused shared experts** — backported. It is upstream only from
  v0.5.22, where the DSA patch script just asserts it.
- **Mooncake KV wait event** — the `mooncake/conn.py` consumer half is added.
  v0.5.21 records the early-send event but never waits on it, so chunked
  prefill over Mooncake can read half-written KV without it.
- **MLA absorb-BMM config** — dropped. This build has no `_absorb_bmm_config`.
- **Decode radix under speculation** — the patch defaults off. The opt-in it
  enables (`SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1`) is unavailable on this
  image.

Validated with **GLM-5.2-MXFP4** on 1 × 8 MI355X (`gfx950`), SGLang only, both
serving topologies, using the `examples/sglang_mix_glm5.2` and
`examples/sglang_2p1d_glm5.2` kits:

| Scenario | SGLang |
| --- | --- |
| Aggregated — TP4 + DP4 attention + EAGLE MTP + HiCache, one worker on 4 GPUs | ✅ PASS · 6/6 |
| Disaggregated — 1 × TP1·PP4 prefill + TP4 / DP4-attention decode, on one node | ✅ PASS · 6/6 |

Each run is the kit's `smoke.sh`: worker registration, completion, chat,
streaming, and a ~66k-token needle prompt sent cold and again from the prefix
cache. The disaggregated run additionally answered two fresh, uncached
~99.5k-token needle prompts correctly, and its prefill and decode both moved KV
through Mooncake's RDMA transport over the ionic rails, with no TCP fallback.

```{admonition} Outside the ROCm 10.2 preview scope
:class: warning
- **Engines** — SGLang only. vLLM and ATOM are not exercised on 10.2.
- **Models** — GLM-5.2-MXFP4 only. gpt-oss-120b, the 10.1 preview model, is
  not re-run on 10.2.
- **Cross-node prefill-decode** — not covered. Prefill (GPUs 0-3) and decode
  (GPUs 4-7) ran on the same node, still over Mooncake RDMA.
- **Accuracy and performance** — smoke checks only. GSM8K and AgentX were not
  run on 10.2.
- **First-request warm-up** — the first long prompts after a cold start pay a
  one-time AITER / Triton JIT cost. A ~66k-token prompt took 530 s the first
  time, while a fresh ~99.5k-token prompt took 4.4 s once warm. Warm up with
  long prompts before taking traffic.
- **`gfx942`** — not exercised on 10.2.
```

## Infera

| Package | Validated version |
| --- | --- |
| `amd-infera` | 0.1.0 |
