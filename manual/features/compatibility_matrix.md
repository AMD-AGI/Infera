# Compatibility matrix

This page lists the exact versions Infera **0.1.0** has been validated against.
Everything here is what we have tested — combinations not listed are outside the
validated set for this release, not necessarily unsupported.

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

| | vLLM | SGLang | ATOM |
| --- | --- | --- | --- |
| **Engine version** | 0.25.1 | 0.5.13 | 0.1.4 |
| **Docker image** | `vllm/vllm-openai-rocm` (ROCm nightly base, digest-pinned) | `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` | `rocm/atom:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0_atom0.1.4_20260612` |
| **ROCm (image base)** | ROCm nightly | 7.2.0 | 7.2.4 |
| **OS (image base)** | — | — | Ubuntu 24.04 |
| **Python (image base)** | — | — | 3.12 |
| **PyTorch (image base)** | — | — | 2.10.0 |
| **Build date** | — | 2026-06-12 | 2026-06-12 |

Fields marked `—` are not encoded in the released image tag and are not
separately stated here — see the platform table below for the validated baseline
that applies across all engines.

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
| ROCm | 7.2 |
| Operating system | Ubuntu 24.04 (Linux x86-64) |
| Python | 3.10+ |
| Docker | Required — runs `etcd` in dev and builds engine images |
| RDMA NIC | AMD AINIC — required only for cross-node prefill-decode |

## Infera

| Package | Validated version |
| --- | --- |
| `amd-infera` | 0.1.0 |
