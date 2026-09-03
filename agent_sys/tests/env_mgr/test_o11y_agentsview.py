# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The AgentsView side-car: which port, and every way it is allowed to fail."""

from __future__ import annotations

import socket

from env_mgr.o11y import agentsview


def test_the_default_port_is_18888() -> None:
    assert agentsview.DEFAULT_PORT == 18888
    assert agentsview.resolve_port(None, {}) == 18888


def test_the_environment_beats_the_default() -> None:
    assert agentsview.resolve_port(None, {"AGENTSVIEW_PORT": "9001"}) == 9001


def test_the_flag_beats_the_environment() -> None:
    assert agentsview.resolve_port(9002, {"AGENTSVIEW_PORT": "9001"}) == 9002


def test_an_unparseable_environment_value_falls_back_to_the_default() -> None:
    assert agentsview.resolve_port(None, {"AGENTSVIEW_PORT": "not-a-port"}) == 18888


def test_port_is_free_says_no_when_something_is_listening() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        taken = s.getsockname()[1]
        assert agentsview.port_is_free(taken) is False


def test_port_is_free_says_yes_when_nothing_is() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    assert agentsview.port_is_free(free) is True
