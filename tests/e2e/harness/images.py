###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Which engine image an e2e run builds, per (engine, GPU architecture).

WHAT: one table mapping ``(engine, arch)`` to ``(image tag, Dockerfile)``.

WHY: the PD-disaggregated conftests and ``tests/run_tests.sh`` both need that
answer, and hardcoding it twice lets them drift.

CONTEXT: only SGLang differs by architecture. Its vendor bases are arch-split
(``mi35x`` / ``mi30x``) and ``build_mooncake_sglang.sh`` compiles Mooncake for
one arch. vLLM and ATOM pin no arch anywhere — their bases are multi-arch and
``build_aiter_rocm.sh`` keeps aiter at ``GPU_ARCHS=native``, JIT-compiling
against the live GPU at first import — so both architectures produce the
identical image and a second tag would only buy a redundant rebuild.
"""

from __future__ import annotations

from .arch import target_arch

_IMAGES: dict[tuple[str, str], tuple[str, str]] = {
    ("sglang", "gfx950"): (
        "infera/engine-sglang:test-local",
        "deploy/docker/Dockerfile.sglang",
    ),
    ("sglang", "gfx942"): (
        "infera/engine-sglang-gfx942:test-local",
        "deploy/docker/Dockerfile.sglang.gfx942",
    ),
    ("vllm", "gfx950"): ("infera/engine-vllm:test-local", "deploy/docker/Dockerfile.vllm"),
    ("vllm", "gfx942"): ("infera/engine-vllm:test-local", "deploy/docker/Dockerfile.vllm"),
    ("atom", "gfx950"): ("infera/engine-atom:test-local", "deploy/docker/Dockerfile.atom"),
    ("atom", "gfx942"): ("infera/engine-atom:test-local", "deploy/docker/Dockerfile.atom"),
}


def engine_image(engine: str, arch: str | None = None) -> tuple[str, str]:
    """``(image_tag, dockerfile)`` for ``engine`` on ``arch`` (default: this run's target).
    Raises on an unregistered pair — inheriting gfx950's image builds one that cannot load."""
    arch = arch or target_arch()
    try:
        return _IMAGES[(engine, arch)]
    except KeyError:
        raise RuntimeError(
            f"no e2e image registered for engine={engine!r} arch={arch!r}; "
            f"add the pair to {__name__}._IMAGES"
        ) from None
