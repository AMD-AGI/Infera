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

import os as _os


def _install_env_aliases() -> None:
    """Back-compat: canonical env vars are ``INFERASIM_*``; legacy ``PRIMUS_*``
    names are still honored.

    We mirror the two prefixes bidirectionally (only filling names that aren't
    already set) so a value set under either prefix is visible to code reading
    the other. This runs on package import, before any submodule reads the
    environment, so both ``inferasim`` and pre-rebrand ``primus`` env vars work.
    """
    for key, val in list(_os.environ.items()):
        if key.startswith("PRIMUS_"):
            _os.environ.setdefault("INFERASIM_" + key[len("PRIMUS_"):], val)
        elif key.startswith("INFERASIM_"):
            _os.environ.setdefault("PRIMUS_" + key[len("INFERASIM_"):], val)


_install_env_aliases()

__all__ = ["launch_projection_from_cli"]


def launch_projection_from_cli(args, overrides):
    from infera.projection.core.projection.inference_projection import (
        launch_projection_from_cli as _impl,
    )
    return _impl(args, overrides)
