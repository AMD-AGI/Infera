###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Importing the kvd connector must register it by NAME in vLLM's factory.

PD + kvd wraps the kvd connector and an RDMA transport in vLLM's
``MultiConnector``. MultiConnector reconstructs each child's stats via
``KVConnectorFactory.get_connector_class_by_name()``, which only consults the
factory registry — it does NOT honor ``kv_connector_module_path``. Without a
name registration, the engine crashes on the first connector-stats record:
``ValueError: Connector 'InferaKvdConnector' is not registered``.
"""

from __future__ import annotations

import pytest


def test_kvd_connector_self_registers_by_name() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("vllm")

    # Import side effect: module registers "InferaKvdConnector" by name.
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    import infera.engine.vllm.kvd_connector as kvd

    cls = KVConnectorFactory.get_connector_class_by_name("InferaKvdConnector")
    assert cls is kvd.InferaKvdConnector
