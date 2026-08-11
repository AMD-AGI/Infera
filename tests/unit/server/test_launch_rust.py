###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""What the launcher hands the Rust binary.

This module had no tests, and it is the only thing standing between a user's
flags and an `execvp` -- so a malformed argv is not a wrong result, it is a
process that never starts. The gate it applies is equally load-bearing: it is
what tells someone their configuration needs the Python backend, instead of
letting the binary fail somewhere less legible.
"""

from __future__ import annotations

import argparse

import pytest

from infera.server.launch_rust import exec_rust


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        host="0.0.0.0",
        port=8000,
        router_backend="rust",
        router_mode="auto",
        router_policy="round-robin",
        discovery_backend="kubernetes",
        etcd_endpoint=None,
        etcd_prefix="/infera/workers/",
        k8s_label_selector="app=worker",
        k8s_namespace=None,
        request_transport="nats",
        kv_event_transport="nats",
        nats_server=None,
        nats_req_idle_timeout=None,
        nats_req_max_duration=None,
        nats_req_max_pending=None,
        request_max_retries=1,
        breaker_failure_threshold=3,
        breaker_cooldown_s=5.0,
        breaker_max_cooldown_s=60.0,
        enable_profiling=False,
        router_tokenizer_path=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _argv(monkeypatch, **overrides) -> list[str]:
    """The argv the launcher would exec, with the exec itself intercepted."""
    captured: list[list[str]] = []

    def fake_exec(binary, argv):
        captured.append(argv)
        raise SystemExit(0)  # exec never returns; neither does this

    monkeypatch.setattr("infera.server.launch_rust.os.execvp", fake_exec)
    monkeypatch.setattr("infera.server.launch_rust._find_binary", lambda: "/fake/infera-router")
    with pytest.raises(SystemExit):
        exec_rust(_args(**overrides))
    assert captured, "the launcher returned without exec'ing"
    return captured[0]


def _value_of(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


def test_every_argument_is_a_string(monkeypatch):
    """execvp rejects anything else with a TypeError, so a None here is not a
    wrong value -- it is a router that cannot start at all."""
    argv = _argv(monkeypatch)
    bad = [a for a in argv if not isinstance(a, str)]
    assert bad == [], f"non-string arguments would abort the exec: {bad}"


def test_the_kubernetes_path_does_not_pass_an_etcd_endpoint(monkeypatch):
    """It has no default, and this backend exists to serve deployments that do
    not have one."""
    argv = _argv(monkeypatch)
    assert "--etcd-endpoint" not in argv
    assert _value_of(argv, "--k8s-label-selector") == "app=worker"


def test_the_etcd_path_passes_its_endpoint(monkeypatch):
    argv = _argv(monkeypatch, discovery_backend="etcd", etcd_endpoint="http://etcd:2379")
    assert _value_of(argv, "--etcd-endpoint") == "http://etcd:2379"
    assert "--k8s-label-selector" not in argv


def test_transports_are_forwarded_rather_than_left_to_a_default(monkeypatch):
    """Both sides default to nats, but the launcher forwards what Python
    resolved -- a deployment must not depend on the two defaults agreeing."""
    argv = _argv(monkeypatch, request_transport="http", kv_event_transport="zmq")
    assert _value_of(argv, "--request-transport") == "http"
    assert _value_of(argv, "--kv-event-transport") == "zmq"


def test_nats_tuning_is_forwarded_when_given(monkeypatch):
    argv = _argv(monkeypatch, nats_req_max_pending=8, nats_req_idle_timeout=30.0)
    assert _value_of(argv, "--nats-req-max-pending") == "8"
    assert _value_of(argv, "--nats-req-idle-timeout-s") == "30.0"


def test_nats_tuning_is_omitted_when_unset(monkeypatch):
    """Unset means "defer": the Rust side reads the same environment variables,
    and passing None would override that with the string "None"."""
    argv = _argv(monkeypatch)
    assert "--nats-req-max-pending" not in argv
    assert "--nats-req-idle-timeout-s" not in argv


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"router_mode": "direct"}, "GAIE direct mode"),
        ({"enable_profiling": True}, "profiling"),
        ({"discovery_backend": "consul"}, "--discovery-backend consul"),
        ({"request_transport": "grpc"}, "--request-transport grpc"),
        ({"kv_event_transport": "kafka"}, "--kv-event-transport kafka"),
        ({"router_policy": "least-loaded"}, "--router-policy least-loaded"),
    ],
)
def test_unsupported_configurations_are_refused_by_name(monkeypatch, overrides, expected):
    """Naming the offending flag is the point: the alternative is a binary that
    exits with something the reader has to map back to their own config."""
    monkeypatch.setattr("infera.server.launch_rust._find_binary", lambda: "/fake/infera-router")
    with pytest.raises(SystemExit) as exc:
        exec_rust(_args(**overrides))
    assert expected in str(exc.value)


def test_kubernetes_without_a_selector_is_refused(monkeypatch):
    """An empty selector would match every Pod in the namespace and register
    anything carrying the annotation."""
    monkeypatch.setattr("infera.server.launch_rust._find_binary", lambda: "/fake/infera-router")
    with pytest.raises(SystemExit) as exc:
        exec_rust(_args(k8s_label_selector=""))
    assert "--k8s-label-selector" in str(exc.value)


def test_etcd_without_an_endpoint_is_refused(monkeypatch):
    monkeypatch.setattr("infera.server.launch_rust._find_binary", lambda: "/fake/infera-router")
    with pytest.raises(SystemExit) as exc:
        exec_rust(_args(discovery_backend="etcd", etcd_endpoint=""))
    assert "--etcd-endpoint" in str(exc.value)
