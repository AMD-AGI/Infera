# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
import pytest

from env_mgr.layer import LAYER_ORDER, layer_index


def test_layer_order():
    assert LAYER_ORDER == ("system", "workspace", "project", "repo", "worktree")


def test_layer_index():
    assert layer_index("system") == 0
    assert layer_index("worktree") == 4


def test_layer_index_unknown_raises():
    with pytest.raises(ValueError):
        layer_index("galaxy")
