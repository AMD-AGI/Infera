# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""o11y side-cars: things that watch a run and may never fail one.

The prefix they install into is `env_mgr.prefix`, not here: it is an `env_mgr`
layout that o11y happens to be the first consumer of.
"""
