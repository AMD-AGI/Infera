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
from pathlib import Path

import pytest

from infera.server.args import parse_server_args
from infera.server.launch_rust import _find_binary, exec_rust


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
        kv_overlap_weight=1.0,
        kv_prefill_overlap_weight=None,
        kv_decode_overlap_weight=None,
        kv_default_chat_template_kwargs=None,
        kv_per_worker_template_kwargs=True,
        session_affinity="off",
        session_affinity_ttl_seconds=3600,
        decode_input_demand=None,
        decode_request_weight=None,
        pd_prefill_guard_release=None,
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


def test_rust_only_routing_switches_are_forwarded_only_when_given(monkeypatch):
    """Unset defers to the INFERA_* variables the Rust side reads itself."""
    argv = _argv(monkeypatch)
    assert "--decode-input-demand" not in argv
    assert "--decode-request-weight" not in argv
    assert "--pd-prefill-guard-release" not in argv
    argv = _argv(
        monkeypatch,
        router_policy="kv-aware",
        decode_input_demand="shadow",
        decode_request_weight=0,
        pd_prefill_guard_release="completion",
    )
    assert _value_of(argv, "--decode-input-demand") == "shadow"
    assert _value_of(argv, "--decode-request-weight") == "0"
    assert _value_of(argv, "--pd-prefill-guard-release") == "completion"


@pytest.mark.parametrize(
    "flags",
    [
        ["--decode-input-demand", "on"],
        ["--decode-request-weight", "180000"],
        ["--pd-prefill-guard-release", "completion"],
    ],
)
def test_rust_only_routing_switches_need_the_rust_backend(flags, capsys):
    argv = ["--router-tokenizer-path", "tokenizer", *flags]
    with pytest.raises(SystemExit):
        parse_server_args(["--router-backend", "python", *argv])
    assert "requires --router-backend rust" in capsys.readouterr().err
    args = parse_server_args(["--router-backend", "rust", *argv])
    given = (args.decode_input_demand, args.decode_request_weight, args.pd_prefill_guard_release)
    assert flags[1] in [str(v) for v in given if v is not None]


def test_decode_request_weight_rejects_negative(capsys):
    with pytest.raises(SystemExit):
        parse_server_args(["--router-backend", "rust", "--decode-request-weight", "-1"])
    assert "must be >= 0" in capsys.readouterr().err


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"router_mode": "direct"}, "GAIE direct mode"),
        ({"enable_profiling": True}, "profiling"),
        ({"discovery_backend": "consul"}, "--discovery-backend consul"),
        ({"request_transport": "grpc"}, "--request-transport grpc"),
        ({"kv_event_transport": "kafka"}, "--kv-event-transport kafka"),
        ({"router_policy": "least-loaded"}, "--router-policy least-loaded"),
        ({"enable_sla_metrics": False}, "--no-enable-sla-metrics"),
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


def test_a_default_kv_aware_run_passes_no_template_kwargs_flags(monkeypatch):
    """Both new flags default to the same thing on both sides, so a default run
    must not name them.

    This is a compatibility property, not tidiness. The launcher and the binary
    ship in one image today, but nothing enforces that -- a mounted binary, a
    staged rollout, a debug build -- and an argument the binary does not know is
    not a degraded feature, it is clap refusing to start the router. Passing a
    default explicitly buys nothing and spends exactly that.
    """
    argv = _argv(monkeypatch, router_policy="kv-aware")
    assert "--kv-per-worker-template-kwargs" not in argv
    assert "--kv-default-chat-template-kwargs" not in argv
    assert _value_of(argv, "--kv-overlap-weight") == "1.0"


def test_turning_per_worker_template_kwargs_off_is_forwarded(monkeypatch):
    """The half that must still reach the binary. Off is a deliberate
    non-default, and a binary too old to understand it would silently ignore the
    request -- there, failing to start is the honest outcome."""
    argv = _argv(monkeypatch, router_policy="kv-aware", kv_per_worker_template_kwargs=False)
    assert _value_of(argv, "--kv-per-worker-template-kwargs") == "false"


def test_fleet_template_kwargs_are_forwarded_verbatim(monkeypatch):
    """The JSON is validated upstream in args.py; the launcher must not reshape
    it, because the router re-parses this exact string."""
    blob = '{"reasoning_effort": "high"}'
    argv = _argv(monkeypatch, router_policy="kv-aware", kv_default_chat_template_kwargs=blob)
    assert _value_of(argv, "--kv-default-chat-template-kwargs") == blob


def test_session_affinity_is_forwarded_only_when_on(monkeypatch):
    """Off is the binary's default too; leaving it out keeps older binaries starting."""
    assert "--session-affinity" not in _argv(monkeypatch)
    argv = _argv(monkeypatch, session_affinity="both", session_affinity_ttl_seconds=60)
    assert _value_of(argv, "--session-affinity") == "both"
    assert _value_of(argv, "--session-affinity-ttl-seconds") == "60"


def test_the_new_kv_flags_stay_inside_the_kv_aware_gate(monkeypatch):
    """A round-robin router does not get kv-aware arguments even when the
    namespace carries them -- the gate is what keeps the two policies' argv
    surfaces independent."""
    argv = _argv(monkeypatch, kv_per_worker_template_kwargs=False)
    assert "--kv-per-worker-template-kwargs" not in argv


def _router_at(path: Path, mode: int = 0o755) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    path.chmod(mode)
    return path


@pytest.fixture
def lookup(monkeypatch, tmp_path):
    """A bundled slot and a PATH entry, both empty until a test fills them."""
    monkeypatch.delenv("INFERA_ROUTER_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "path"))
    bundled = tmp_path / "wheel" / "_bin" / "infera-router"
    monkeypatch.setattr("infera.server.launch_rust._BUNDLED_BINARY", bundled)
    return bundled, tmp_path / "path" / "infera-router"


def test_the_bundled_router_wins_over_one_on_path(lookup):
    """The wheel's own router is the one this launcher's argv was written for; a
    copy on PATH -- an image's /usr/local/bin, say -- may be from any release."""
    bundled, on_path = lookup
    _router_at(on_path)
    _router_at(bundled)
    assert _find_binary() == str(bundled)


def test_without_a_bundled_router_path_is_used(lookup):
    """Source installs (every image and the overlay) carry no bundled binary."""
    _, on_path = lookup
    _router_at(on_path)
    assert _find_binary() == str(on_path)


def test_infera_router_bin_overrides_the_bundled_router(lookup, monkeypatch):
    bundled, _ = lookup
    _router_at(bundled)
    monkeypatch.setenv("INFERA_ROUTER_BIN", "/custom/infera-router")
    assert _find_binary() == "/custom/infera-router"


def test_a_bundled_router_that_cannot_execute_is_named(lookup):
    """Falling through to PATH would silently run a different router."""
    bundled, on_path = lookup
    _router_at(on_path)
    _router_at(bundled, mode=0o644)
    with pytest.raises(SystemExit) as exc:
        _find_binary()
    assert str(bundled) in str(exc.value)
