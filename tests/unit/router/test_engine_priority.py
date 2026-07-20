###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for infera/router/engine_priority.py — translating
CacheHints to engine-specific priority/retention fields on the
request body before forwarding to a worker."""

from __future__ import annotations

from infera.common.worker_pool import EngineType
from infera.router.cache_control import CacheHints, Retention
from infera.router.engine_priority import inject_engine_priority


def _hints(retention: Retention, *, explicit: bool = True) -> CacheHints:
    return CacheHints(
        retention=retention,
        session_id=None,
        explicit_hint_seen=explicit,
    )


# ----------------------------------------------------------------------
# SGLang — priority integer field
# ----------------------------------------------------------------------


def test_sglang_long_sets_priority_100():
    body = {"model": "m", "messages": []}
    out = inject_engine_priority(body, _hints(Retention.LONG), EngineType.SGLANG)
    assert out["priority"] == 100


def test_sglang_short_sets_priority_50():
    body = {"model": "m"}
    out = inject_engine_priority(body, _hints(Retention.SHORT), EngineType.SGLANG)
    assert out["priority"] == 50


def test_sglang_no_hint_defaults_to_long_priority():
    # No cache_control (retention NONE) now maps to the router default (long),
    # so it is retained rather than evict-first; priority follows long=100.
    body = {"model": "m"}
    out = inject_engine_priority(body, _hints(Retention.NONE), EngineType.SGLANG)
    assert out["priority"] == 100


def test_sglang_client_priority_wins_over_default():
    """Deliberate client-side priority should not be overwritten."""
    body = {"model": "m", "priority": 999}
    out = inject_engine_priority(body, _hints(Retention.LONG), EngineType.SGLANG)
    assert out["priority"] == 999


def test_sglang_does_not_mutate_input_body():
    body = {"model": "m"}
    out = inject_engine_priority(body, _hints(Retention.LONG), EngineType.SGLANG)
    assert "priority" not in body  # original untouched
    assert out["priority"] == 100


def test_sglang_also_stashes_infera_retention_string():
    """SGLang's `priority` field affects radix
    eviction but NOT kvd retention. We additionally stash the retention
    string in the body so a worker-side middleware can extract it and
    bridge to `set_request_retention_hint`."""
    body = {"model": "m"}
    out = inject_engine_priority(body, _hints(Retention.LONG), EngineType.SGLANG)
    assert out["infera_retention"] == "long"
    # Short and none round-trip too.
    assert (
        inject_engine_priority({}, _hints(Retention.SHORT), EngineType.SGLANG)["infera_retention"]
        == "short"
    )
    # No-hint (NONE) now defaults to long (cache-by-default), not "none".
    assert (
        inject_engine_priority({}, _hints(Retention.NONE), EngineType.SGLANG)["infera_retention"]
        == "long"
    )


def test_sglang_client_infera_retention_wins_over_default():
    """A client (or upstream proxy) that explicitly stamps
    infera_retention on the body must not be overwritten."""
    body = {"model": "m", "infera_retention": "long"}
    out = inject_engine_priority(body, _hints(Retention.SHORT), EngineType.SGLANG)
    assert out["infera_retention"] == "long"  # untouched
    assert out["priority"] == 50  # but our priority default still applies


# ----------------------------------------------------------------------
# vLLM — placeholder retention field for future connector
# ----------------------------------------------------------------------


def test_vllm_attaches_retention_under_private_key():
    body = {"model": "m"}
    out = inject_engine_priority(body, _hints(Retention.LONG), EngineType.VLLM)
    assert out["_infera_retention"] == "long"
    assert "priority" not in out  # vLLM doesn't get SGLang's field


def test_vllm_short_attaches_short():
    body = {"model": "m"}
    out = inject_engine_priority(body, _hints(Retention.SHORT), EngineType.VLLM)
    assert out["_infera_retention"] == "short"


def test_vllm_client_retention_wins_over_default():
    body = {"model": "m", "_infera_retention": "none"}
    out = inject_engine_priority(body, _hints(Retention.LONG), EngineType.VLLM)
    assert out["_infera_retention"] == "none"


# ----------------------------------------------------------------------
# Phase A.1: vLLM kv_transfer_params is the in-band channel that
# actually survives the OpenAI → vLLM API boundary. We must put the
# retention there too (the top-level `_infera_retention` survives
# pre-validation in our server but gets stripped by vLLM's pydantic
# schema; kv_transfer_params is the schema-blessed dict that gets
# plumbed through to Request.kv_transfer_params).
# ----------------------------------------------------------------------


def test_vllm_writes_retention_into_kv_transfer_params():
    body = {"model": "m"}
    out = inject_engine_priority(body, _hints(Retention.LONG), EngineType.VLLM)
    assert out["kv_transfer_params"] == {"infera_retention": "long"}


def test_vllm_preserves_existing_kv_transfer_params_keys():
    """Operator may already be using kv_transfer_params for disagg/PD
    or some other purpose. We add `infera_retention` without
    clobbering siblings."""
    body = {
        "model": "m",
        "kv_transfer_params": {"engine_id": "abc", "kv_role": "kv_both"},
    }
    out = inject_engine_priority(body, _hints(Retention.LONG), EngineType.VLLM)
    assert out["kv_transfer_params"]["engine_id"] == "abc"
    assert out["kv_transfer_params"]["kv_role"] == "kv_both"
    assert out["kv_transfer_params"]["infera_retention"] == "long"


def test_vllm_client_supplied_infera_retention_in_kv_transfer_params_wins():
    """If the client (or a deliberate test/debug knob) already set
    `kv_transfer_params={"infera_retention": "..."}`, our injection
    shouldn't overwrite — same convention as other fields."""
    body = {
        "model": "m",
        "kv_transfer_params": {"infera_retention": "none"},
    }
    out = inject_engine_priority(body, _hints(Retention.LONG), EngineType.VLLM)
    assert out["kv_transfer_params"]["infera_retention"] == "none"


def test_vllm_kv_transfer_params_field_is_a_fresh_dict():
    """Modifying the output's kv_transfer_params must not mutate the
    caller's input dict (we copy before adding our key)."""
    inner = {"engine_id": "abc"}
    body = {"model": "m", "kv_transfer_params": inner}
    out = inject_engine_priority(body, _hints(Retention.LONG), EngineType.VLLM)
    assert out["kv_transfer_params"] is not inner
    # Original input dict unchanged.
    assert inner == {"engine_id": "abc"}


# ----------------------------------------------------------------------
# Unknown engine — no-op
# ----------------------------------------------------------------------


def test_atom_engine_is_noop():
    """ATOM doesn't have a defined cache mechanism yet; injection is a no-op."""
    body = {"model": "m"}
    out = inject_engine_priority(body, _hints(Retention.LONG), EngineType.ATOM)
    assert "priority" not in out
    assert "_infera_retention" not in out
    assert out == body  # equal, not identical (still a copy)


# ----------------------------------------------------------------------
# Returns a new dict, not a reference
# ----------------------------------------------------------------------


def test_output_is_separate_dict_object():
    body = {"model": "m"}
    out = inject_engine_priority(body, _hints(Retention.LONG), EngineType.SGLANG)
    assert out is not body
    # Mutating out doesn't touch body.
    out["extra"] = "x"
    assert "extra" not in body
