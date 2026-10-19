###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""``parse_atom_args`` — ``--kv-transfer-config`` → disagg mode + meta.

ATOM itself is not importable in CPU unit tests, so its ``EngineArgs``
parser is stubbed with the few flags infera reads.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from typing import Any

import pytest

from infera.common.worker_pool import DisaggMode
from infera.engine.atom.args import AtomWorkerArgs, parse_atom_args

_MOONCAKE_PRODUCER = {
    "kv_connector": "mooncake",
    "kv_role": "kv_producer",
    "handshake_port": 21301,
}
_MOONCAKE_CONSUMER = {"kv_connector": "mooncake", "kv_role": "kv_consumer"}
_OFFLOAD = {"kv_connector": "lmcache_offload", "kv_role": "offload"}


class _StubEngineArgs:
    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model", type=str, default="m")
        parser.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
        parser.add_argument("--data-parallel-size", type=int, default=1)
        parser.add_argument("--enable-dp-attention", action="store_true")
        parser.add_argument("--block-size", type=int, default=16)
        parser.add_argument("--kv-transfer-config", type=str, default=None)


@pytest.fixture(autouse=True)
def _stub_atom(monkeypatch: pytest.MonkeyPatch) -> None:
    arg_utils = types.ModuleType("atom.model_engine.arg_utils")
    arg_utils.EngineArgs = _StubEngineArgs  # type: ignore[attr-defined]
    for name in ("atom", "atom.model_engine"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "atom.model_engine.arg_utils", arg_utils)


def _parse(kv_cfg: Any) -> AtomWorkerArgs:
    return parse_atom_args(
        [
            "--etcd-endpoint=127.0.0.1:2379",
            "--advertise-host=10.0.0.1",
            "--server-port=19001",
            "--port=29500",
            "-tp=8",
            f"--kv-transfer-config={json.dumps(kv_cfg)}",
        ]
    )


def _multi(*connectors: Any) -> dict[str, Any]:
    return {"kv_connector": "multi", "connectors": list(connectors)}


def test_plain_mooncake_producer_is_prefill() -> None:
    args = _parse(_MOONCAKE_PRODUCER)
    assert args.disagg_mode is DisaggMode.PREFILL
    assert args.disagg_meta == {
        "protocol": "atom-mooncake",
        "params": {
            "host": "10.0.0.1",
            "http_port": 19001,
            "handshake_port": 21301,
            "tp_size": 8,
            "dp_size": 1,
        },
    }


@pytest.mark.parametrize(
    "kv_cfg",
    [
        _multi(_MOONCAKE_PRODUCER, _OFFLOAD),
        _multi(_OFFLOAD, _MOONCAKE_PRODUCER),
        _multi("junk", None, _OFFLOAD, _MOONCAKE_PRODUCER),
        _multi({"kv_connector": "x", "kv_role": "kv_both"}, _MOONCAKE_PRODUCER),
    ],
    ids=["producer-first", "offload-first", "non-dict-entries", "kv-both-first"],
)
def test_multi_resolves_to_mooncake_producer(kv_cfg: dict[str, Any]) -> None:
    expected = _parse(_MOONCAKE_PRODUCER)
    args = _parse(kv_cfg)
    assert args.disagg_mode is DisaggMode.PREFILL
    assert args.disagg_meta == expected.disagg_meta


@pytest.mark.parametrize(
    "kv_cfg",
    [_multi(_OFFLOAD), _multi(), {"kv_connector": "multi"}, _OFFLOAD, {}],
    ids=["multi-offload-only", "multi-empty", "multi-no-list", "offload", "empty"],
)
def test_no_pd_connector_is_mixed(kv_cfg: dict[str, Any]) -> None:
    args = _parse(kv_cfg)
    assert args.disagg_mode is DisaggMode.MIXED
    assert args.disagg_meta == {}


def test_consumer_unaffected() -> None:
    plain = _parse(_MOONCAKE_CONSUMER)
    assert plain.disagg_mode is DisaggMode.DECODE
    assert plain.disagg_meta["protocol"] == "atom-mooncake"
    assert plain.disagg_meta["params"]["handshake_port"] == 6301
    multi = _parse(_multi(_OFFLOAD, _MOONCAKE_CONSUMER))
    assert (multi.disagg_mode, multi.disagg_meta) == (plain.disagg_mode, plain.disagg_meta)


def test_atom_argv_forwarded_verbatim() -> None:
    kv_cfg = _multi(_MOONCAKE_PRODUCER, _OFFLOAD)
    args = _parse(kv_cfg)
    assert f"--kv-transfer-config={json.dumps(kv_cfg)}" in args.atom_argv
    assert "--etcd-endpoint=127.0.0.1:2379" not in args.atom_argv


def _parse_parallel(*flags: str, kv_cfg: Any = None) -> AtomWorkerArgs:
    argv = ["--etcd-endpoint=127.0.0.1:2379", "--server-port=19001", "--port=29500", *flags]
    if kv_cfg is not None:
        argv.append(f"--kv-transfer-config={json.dumps(kv_cfg)}")
    return parse_atom_args(argv)


@pytest.mark.parametrize(
    ("flags", "dp_size"),
    [
        (("-tp=4",), None),
        (("-tp=4", "--enable-dp-attention"), 4),
        (("-tp=4", "--enable-dp-attention", "--data-parallel-size=2"), 8),
        (("-tp=1", "--data-parallel-size=2"), 2),
    ],
)
def test_mixed_worker_registers_its_dp_ranks(flags: tuple[str, ...], dp_size: int | None) -> None:
    # Under DP-attention every TP rank is its own DP rank (ATOM's
    # global_engine_count), so the router can address each one.
    assert _parse_parallel(*flags).dp_size == dp_size


def test_pd_worker_does_not_register_dp_ranks() -> None:
    # A PD prefill picks its own rank and reports it in the hand-off.
    args = _parse_parallel("-tp=4", "--enable-dp-attention", kv_cfg=_MOONCAKE_PRODUCER)
    assert args.disagg_mode is DisaggMode.PREFILL
    assert args.dp_size is None
