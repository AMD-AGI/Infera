# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Choose the candidate explicitly; the catalog tool never activates it."""


def select(use_candidate=False):
    if use_candidate:
        from candidate import run
    else:
        from original import run
    return run
