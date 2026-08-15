###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""inferasim -- Infera's inference/serving simulation + projection tool.

inferasim is a workload-driven simulate-then-verify surface for the serving
stack. Measure sparsely on one GPU, transport analytically to every
serving recipe (TP/EP/PP, batch, concurrency, dtype), simulate arrival-driven
serving with the discrete-event engine, then let the tuning agent search the
recipe space -- validating only the shortlist on real GPUs (Hyperloom/Infera).
"""

__all__ = ["launch_projection_from_cli"]


def launch_projection_from_cli(args, overrides):
    from infera.projection.core.projection.inference_projection import (
        launch_projection_from_cli as _impl,
    )
    return _impl(args, overrides)
