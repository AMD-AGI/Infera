# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Suite-wide guards. Currently one: **the test suite owns no host state.**"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True, scope="session")
def _prefix_is_never_the_operators(tmp_path_factory) -> Iterator[None]:
    """`AGENT_SYS_HOME` points into `tmp` for the whole session.

    **Measured, and it was not theoretical.** A plain `pytest` wrote **128 MB**
    into `$AGENT_SYS_HOME` — which, with no override, is the operator's real
    `~/.infera_agent_sys`. Any test reaching `cli.main.main(["run", ...])`
    without patching `ensure_installed` runs the actual recipe: a 45 MB download
    and a 133 MB binary. Three files did it, and nothing said so.

    Session-scoped and autouse because the next test to do it will be written by
    someone who did not know — a per-test patch is a rule people forget, and the
    failure is a green suite that happens to have downloaded 45 MB. A test that
    wants its own prefix still sets one; function-scoped `monkeypatch` wins.
    """
    root = tmp_path_factory.mktemp("agent_sys_home")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("AGENT_SYS_HOME", str(root))
        yield
    assert "AGENT_SYS_HOME" not in os.environ or os.environ["AGENT_SYS_HOME"] != str(root)
