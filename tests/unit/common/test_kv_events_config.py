###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Reading the publisher port out of a caller-supplied --kv-events-config."""

from __future__ import annotations

import json

import pytest

from infera.common.kv_events_config import explicit_kv_events_port


def _config(port: int, **extra) -> str:
    return json.dumps(
        {"publisher": "zmq", "endpoint": f"tcp://*:{port}", "topic": "kv-events", **extra}
    )


@pytest.mark.parametrize(
    ("argv", "port"),
    [
        (["--tp-size", "4"], None),
        (["--kv-events-config", _config(5560), "--tp-size", "4"], 5560),
        ([f"--kv-events-config={_config(5570)}"], 5570),
        (["--kv-events-config", _config(5580, enable_kv_cache_events=True)], 5580),
    ],
)
def test_explicit_port_is_read_from_either_flag_form(argv, port):
    assert explicit_kv_events_port(argv) == port


def test_a_config_without_a_port_is_rejected():
    with pytest.raises(ValueError, match="endpoint with a port"):
        explicit_kv_events_port(["--kv-events-config", json.dumps({"publisher": "zmq"})])


def test_a_topic_the_router_does_not_subscribe_to_is_rejected():
    with pytest.raises(ValueError, match="only subscribes to 'kv-events'"):
        explicit_kv_events_port(["--kv-events-config", _config(5560, topic="")])
