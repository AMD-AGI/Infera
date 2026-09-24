###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""How each worker entrypoint wires the readiness port and startup checks.

The entrypoints import their engine at module load, so these read the source
rather than importing it, the same way test_shutdown_order does.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from infera.engine import readiness

ROOT = Path(inspect.getfile(readiness)).parents[2]

# Entrypoint, and the call that installs its SIGTERM handler.
ENTRYPOINTS = (
    ("infera/engine/sglang/__main__.py", "_supervise_engine"),
    ("infera/engine/vllm/__main__.py", "add_signal_handler"),
    ("infera/engine/atom/__main__.py", "add_signal_handler"),
)


def _call_lines(tree: ast.AST, name: str) -> list[int]:
    """Source lines of every call to ``name``, as a function or a method."""
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Name) and func.id == name) or (
            isinstance(func, ast.Attribute) and func.attr == name
        ):
            lines.append(node.lineno)
    return sorted(lines)


def _function(tree: ast.AST, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"no function {name}")


@pytest.mark.parametrize(("entrypoint", "_signal_call"), ENTRYPOINTS)
def test_every_entrypoint_opens_and_closes_the_readiness_port(entrypoint, _signal_call):
    # The operator probes every single-node worker on this port regardless of
    # backend, so an entrypoint that never opens it sits NotReady forever.
    tree = ast.parse((ROOT / entrypoint).read_text())
    assert _call_lines(tree, "serve_readiness_best_effort"), f"{entrypoint} never opens it"
    assert _call_lines(tree, "close_readiness"), f"{entrypoint} never closes it"


@pytest.mark.parametrize(("entrypoint", "signal_call"), ENTRYPOINTS)
def test_signal_handlers_are_installed_before_the_readiness_port_opens(entrypoint, signal_call):
    # A SIGTERM between the two would take the default action and kill the
    # process without closing the port or deregistering.
    tree = ast.parse((ROOT / entrypoint).read_text())
    handlers = _call_lines(tree, signal_call)
    opened = _call_lines(tree, "serve_readiness_best_effort")
    assert handlers and opened
    assert handlers[0] < opened[0], (
        f"{entrypoint} opens the readiness port before it handles SIGTERM"
    )


def test_decode_checks_its_discovery_config_before_loading_weights():
    # The decode's prefill probe runs after the weights are in, so a discovery
    # config it can never resolve must fail before the load, as prefill does.
    tree = ast.parse((ROOT / "infera/engine/sglang/__main__.py").read_text())
    main = _function(tree, "main")
    guards = _call_lines(main, "should_verify_prefill")
    starts = _call_lines(main, "start")
    assert guards, "main() never checks the decode discovery config"
    assert starts and guards[0] < starts[0], "the decode check runs after engine.start()"


@pytest.mark.parametrize(("entrypoint", "_signal_call"), ENTRYPOINTS)
def test_the_engine_check_dials_the_bind_address(entrypoint, _signal_call):
    # config.host is the advertised address, which need not be reachable from
    # inside the pod; the engine is waited on at its bind address, and the
    # readiness check must dial the same one.
    tree = ast.parse((ROOT / entrypoint).read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "engine_health_check"
    ]
    assert calls, f"{entrypoint} never builds an engine check"
    for call in calls:
        host = call.args[0]
        assert not (
            isinstance(host, ast.Attribute)
            and host.attr == "host"
            and isinstance(host.value, ast.Name)
            and host.value.id == "config"
        ), f"{entrypoint} dials the advertised host"
