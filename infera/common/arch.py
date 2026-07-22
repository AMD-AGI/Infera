###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""GPU architecture detection for the ROCm/CDNA fleet.

A single, stable place to answer "what GPU is this?" for both the Python engine
wrappers and the launch scripts. Uses torch's device properties (the engines
already depend on torch) rather than parsing ``amd-smi`` text/JSON, which is an
unstable interface across ROCm releases. The compute-capability -> gfx mapping
matches AMD's convention (mirrors Primus-Turbo's ``core/utils``):

    (9, 4)  -> gfx942  (MI300 / MI325X, CDNA3)
    (9, 5)  -> gfx950  (MI355X, CDNA4)

Result is cached: the GPU arch is fixed for the process lifetime.
"""

from __future__ import annotations

import functools

# Compute capability (major, minor) -> gfx arch name.
_CC_TO_GFX: dict[tuple[int, int], str] = {
    (9, 4): "gfx942",
    (9, 5): "gfx950",
}


@functools.lru_cache(maxsize=1)
def _compute_capability() -> tuple[int, int] | None:
    """(major, minor) compute capability of the current GPU, or None.

    Returns None on CPU-only / no-GPU hosts (and if torch is unimportable) so
    callers no-op safely.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return (props.major, props.minor)
    except Exception:
        return None


def gpu_arch() -> str | None:
    """The current GPU's gfx arch name (e.g. ``"gfx942"``), or None if unknown.

    Unknown = no GPU, torch unavailable, or a capability not in the CDNA map.
    """
    cc = _compute_capability()
    if cc is None:
        return None
    return _CC_TO_GFX.get(cc)


def is_gfx942() -> bool:
    """True iff the current GPU is gfx942 (MI300 / MI325X, CDNA3)."""
    return _compute_capability() == (9, 4)


def is_gfx950() -> bool:
    """True iff the current GPU is gfx950 (MI355X, CDNA4)."""
    return _compute_capability() == (9, 5)


if __name__ == "__main__":
    # `python -m infera.common.arch` prints the gfx arch (e.g. "gfx942") so
    # launch scripts can read it from inside the ROCm container without parsing
    # amd-smi. Prints nothing and exits non-zero when the arch is unknown.
    import sys

    arch = gpu_arch()
    if arch is None:
        sys.exit(1)
    print(arch)
