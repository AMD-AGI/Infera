###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Infera inference/serving projection + tuning.

Measure sparsely on one GPU, transport analytically to every serving recipe
(TP/EP/PP, batch, concurrency, dtype), then let the tuning agent search the
recipe space. Vendored from Primus under :mod:`infera.projection._vendor`.
"""

__all__ = ["launch_projection_from_cli"]


def launch_projection_from_cli(args, overrides):
    from infera.projection._vendor.projection_core.core.projection.inference_projection import (
        launch_projection_from_cli as _impl,
    )
    return _impl(args, overrides)
