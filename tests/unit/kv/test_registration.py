###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for the registration extension: KvRegistrationMetadata,
WorkerInfo.kv passthrough, CanaryVerifier, and Registry's
canary-rejection behavior on _upsert.

The Registry tests stub `_upsert` directly rather than going through
etcd — _upsert is the function under test for these scenarios.
"""

from __future__ import annotations

import json

import pytest

from infera.common.discovery import Registry
from infera.common.worker_pool import (
    CanaryMismatch,
    CanaryVerifier,
    KvRegistrationMetadata,
    WorkerInfo,
)

# ----------------------------------------------------------------------
# KvRegistrationMetadata round-trip
# ----------------------------------------------------------------------


def test_kv_metadata_round_trip() -> None:
    kv = KvRegistrationMetadata(
        engine_block_size=1,
        index_block_size=64,
        tokenizer="Qwen/Qwen3-0.6B",
        tokenizer_digest="0123456789abcdef",
        tokenizer_canary=[9707, 11, 1879, 1235],
        supports_events=True,
        event_version=1,
        events_endpoint="tcp://10.0.0.5:5557",
        tiers=["device", "host"],
    )
    d = kv.to_dict()
    assert d["engine_block_size"] == 1
    assert d["index_block_size"] == 64
    assert d["tokenizer_canary"] == [9707, 11, 1879, 1235]

    restored = KvRegistrationMetadata.from_dict(d)
    assert restored == kv


def test_kv_metadata_from_dict_applies_defaults() -> None:
    """Optional fields default sensibly when absent from the dict."""
    kv = KvRegistrationMetadata.from_dict(
        {
            "engine_block_size": 16,
            "index_block_size": 64,
            "tokenizer": "tok",
            "tokenizer_digest": "abc",
            "tokenizer_canary": [1, 2, 3],
        }
    )
    assert kv.supports_events is True
    assert kv.event_version == 1
    assert kv.events_endpoint is None
    assert kv.tiers == ["device"]


def test_kv_metadata_coerces_types() -> None:
    """JSON round-trip may turn tuples into lists; from_dict normalizes."""
    kv = KvRegistrationMetadata.from_dict(
        {
            "engine_block_size": "16",  # str → int
            "index_block_size": 64,
            "tokenizer": "tok",
            "tokenizer_digest": "abc",
            "tokenizer_canary": (1, 2, 3),  # tuple → list
        }
    )
    assert kv.engine_block_size == 16
    assert kv.tokenizer_canary == [1, 2, 3]


# ----------------------------------------------------------------------
# CanaryVerifier
# ----------------------------------------------------------------------


def test_canary_verifier_first_call_records_reference() -> None:
    v = CanaryVerifier()
    v.verify(model_name="m", worker_id="w1", canary=[1, 2, 3])
    assert v.reference("m") == (1, 2, 3)


def test_canary_verifier_matching_subsequent_accepted() -> None:
    v = CanaryVerifier()
    v.verify(model_name="m", worker_id="w1", canary=[1, 2, 3])
    v.verify(model_name="m", worker_id="w2", canary=[1, 2, 3])  # no raise


def test_canary_verifier_mismatch_raises() -> None:
    v = CanaryVerifier()
    v.verify(model_name="m", worker_id="w1", canary=[1, 2, 3])
    with pytest.raises(CanaryMismatch) as exc_info:
        v.verify(model_name="m", worker_id="w2", canary=[1, 2, 9])
    err = exc_info.value
    assert err.model_name == "m"
    assert err.first_worker_id == "w1"
    assert err.new_worker_id == "w2"
    assert err.expected == (1, 2, 3)
    assert err.got == (1, 2, 9)


def test_canary_verifier_different_models_isolated() -> None:
    v = CanaryVerifier()
    v.verify(model_name="m1", worker_id="w1", canary=[1, 2, 3])
    v.verify(model_name="m2", worker_id="w2", canary=[7, 8, 9])
    # Both still valid; different models.
    v.verify(model_name="m1", worker_id="w3", canary=[1, 2, 3])
    v.verify(model_name="m2", worker_id="w4", canary=[7, 8, 9])


def test_canary_verifier_forget_resets() -> None:
    v = CanaryVerifier()
    v.verify(model_name="m", worker_id="w1", canary=[1, 2, 3])
    v.forget("m")
    assert v.reference("m") is None
    # Fresh registration sets a new reference (different canary OK now).
    v.verify(model_name="m", worker_id="w2", canary=[9, 9, 9])
    assert v.reference("m") == (9, 9, 9)


def test_canary_mismatch_message_truncates_long_canaries() -> None:
    v = CanaryVerifier()
    long_a = list(range(1000))
    long_b = list(range(1, 1001))
    v.verify(model_name="m", worker_id="w1", canary=long_a)
    with pytest.raises(CanaryMismatch) as exc_info:
        v.verify(model_name="m", worker_id="w2", canary=long_b)
    # The message should be tractable — not 4000 chars of token IDs.
    assert len(str(exc_info.value)) < 500


# ----------------------------------------------------------------------
# Registry._upsert — canary integration
# ----------------------------------------------------------------------


def _payload(
    *,
    worker_id: str,
    model_name: str = "m",
    canary: list[int] | None = None,
    kv: bool = True,
) -> bytes:
    payload: dict = {
        "worker_id": worker_id,
        "url": f"http://{worker_id}",
        "model_name": model_name,
        "engine": "sglang",
        "disagg_mode": "mixed",
        "disagg_meta": {},
    }
    if kv:
        payload["kv"] = {
            "engine_block_size": 1,
            "index_block_size": 64,
            "tokenizer": "tok",
            "tokenizer_digest": "abc",
            "tokenizer_canary": canary if canary is not None else [1, 2, 3],
        }
    return json.dumps(payload).encode("utf-8")


def _registry() -> Registry:
    # No-op endpoint; we only call _upsert in these tests.
    return Registry("localhost:0")


def test_registry_upsert_accepts_first_worker() -> None:
    reg = _registry()
    reg._upsert(reg._prefix + "w1", _payload(worker_id="w1", canary=[1, 2, 3]))
    assert reg.pool.get("w1") is not None
    assert reg.pool.get("w1").kv is not None
    assert reg.pool.get("w1").kv.tokenizer_canary == [1, 2, 3]
    assert reg.canary_rejections == 0


def test_registry_upsert_accepts_matching_canary() -> None:
    reg = _registry()
    reg._upsert(reg._prefix + "w1", _payload(worker_id="w1", canary=[1, 2, 3]))
    reg._upsert(reg._prefix + "w2", _payload(worker_id="w2", canary=[1, 2, 3]))
    assert reg.pool.get("w1") is not None
    assert reg.pool.get("w2") is not None
    assert reg.canary_rejections == 0


def test_registry_upsert_rejects_mismatched_canary() -> None:
    reg = _registry()
    reg._upsert(reg._prefix + "w1", _payload(worker_id="w1", canary=[1, 2, 3]))
    reg._upsert(reg._prefix + "w2", _payload(worker_id="w2", canary=[1, 2, 9]))
    assert reg.pool.get("w1") is not None
    assert reg.pool.get("w2") is None  # rejected
    assert reg.canary_rejections == 1


def test_registry_upsert_canary_per_model_name() -> None:
    """Different model_names have independent canaries."""
    reg = _registry()
    reg._upsert(reg._prefix + "w1", _payload(worker_id="w1", model_name="mA", canary=[1, 2, 3]))
    reg._upsert(reg._prefix + "w2", _payload(worker_id="w2", model_name="mB", canary=[7, 8, 9]))
    assert reg.pool.get("w1") is not None
    assert reg.pool.get("w2") is not None


def test_registry_upsert_accepts_worker_without_kv_block() -> None:
    """Pre-Phase-1 workers (or workers that opt out) register normally;
    they just don't participate in the kv index."""
    reg = _registry()
    reg._upsert(reg._prefix + "w1", _payload(worker_id="w1", kv=False))
    assert reg.pool.get("w1") is not None
    assert reg.pool.get("w1").kv is None
    assert reg.canary_rejections == 0


def test_registry_delete_forgets_canary_when_last_worker_for_model_leaves() -> None:
    reg = _registry()
    reg._upsert(reg._prefix + "w1", _payload(worker_id="w1", model_name="mA", canary=[1, 2, 3]))
    # Simulate a DELETE event.
    msg = {
        "result": {
            "events": [
                {
                    "type": "DELETE",
                    "kv": {
                        "key": __import__("base64")
                        .b64encode((reg._prefix + "w1").encode())
                        .decode()
                    },
                }
            ]
        }
    }
    reg._dispatch_watch(msg)
    # After last worker leaves, canary reference should be forgotten.
    assert reg._canary.reference("mA") is None
    # A new worker for the same model can now register with a different canary.
    reg._upsert(reg._prefix + "w2", _payload(worker_id="w2", model_name="mA", canary=[9, 9, 9]))
    assert reg.pool.get("w2") is not None
    assert reg.canary_rejections == 0


def test_registry_delete_keeps_canary_when_others_remain() -> None:
    reg = _registry()
    reg._upsert(reg._prefix + "w1", _payload(worker_id="w1", model_name="mA", canary=[1, 2, 3]))
    reg._upsert(reg._prefix + "w2", _payload(worker_id="w2", model_name="mA", canary=[1, 2, 3]))
    # Delete w1 only.
    msg = {
        "result": {
            "events": [
                {
                    "type": "DELETE",
                    "kv": {
                        "key": __import__("base64")
                        .b64encode((reg._prefix + "w1").encode())
                        .decode()
                    },
                }
            ]
        }
    }
    reg._dispatch_watch(msg)
    # w2 still serves mA → canary persists.
    assert reg._canary.reference("mA") == (1, 2, 3)
    # Mismatched newcomer still rejected.
    reg._upsert(reg._prefix + "w3", _payload(worker_id="w3", model_name="mA", canary=[9, 9, 9]))
    assert reg.pool.get("w3") is None
    assert reg.canary_rejections == 1


def test_registry_accepts_custom_canary_verifier() -> None:
    """Operator can pre-seed a verifier — useful for tests and for
    explicit per-fleet canary management."""
    verifier = CanaryVerifier()
    verifier.verify(model_name="mA", worker_id="seed", canary=[42])
    reg = Registry("localhost:0", canary_verifier=verifier)
    reg._upsert(reg._prefix + "w1", _payload(worker_id="w1", model_name="mA", canary=[42]))
    assert reg.pool.get("w1") is not None
    reg._upsert(reg._prefix + "w2", _payload(worker_id="w2", model_name="mA", canary=[99]))
    assert reg.pool.get("w2") is None  # rejected


# ----------------------------------------------------------------------
# WorkerInfo backwards compat
# ----------------------------------------------------------------------


def test_worker_info_default_kv_is_none() -> None:
    info = WorkerInfo(worker_id="w1", url="http://x", model_name="m")
    assert info.kv is None


# ----------------------------------------------------------------------
# Registry lifecycle callbacks
# ----------------------------------------------------------------------


def test_on_worker_added_fires_on_new_registration() -> None:
    seen: list[WorkerInfo] = []
    reg = Registry("localhost:0", on_worker_added=lambda info: seen.append(info))
    reg._upsert(reg._prefix + "w1", _payload(worker_id="w1", canary=[1, 2, 3]))
    assert len(seen) == 1
    assert seen[0].worker_id == "w1"
    assert seen[0].kv is not None


def test_on_worker_added_does_not_fire_on_update() -> None:
    """Re-registration via lease refresh shouldn't refire — the listener
    typically does add-subscriber and we don't want duplicate sockets."""
    seen: list[WorkerInfo] = []
    reg = Registry("localhost:0", on_worker_added=lambda info: seen.append(info))
    reg._upsert(reg._prefix + "w1", _payload(worker_id="w1", canary=[1, 2, 3]))
    reg._upsert(reg._prefix + "w1", _payload(worker_id="w1", canary=[1, 2, 3]))  # update
    assert len(seen) == 1


def test_on_worker_added_does_not_fire_on_canary_rejection() -> None:
    """A rejected registration must not look like an add to listeners."""
    seen: list[WorkerInfo] = []
    reg = Registry("localhost:0", on_worker_added=lambda info: seen.append(info))
    reg._upsert(reg._prefix + "w1", _payload(worker_id="w1", canary=[1, 2, 3]))
    reg._upsert(reg._prefix + "w2", _payload(worker_id="w2", canary=[9, 9, 9]))  # rejected
    assert [info.worker_id for info in seen] == ["w1"]


def test_on_worker_removed_fires_on_delete() -> None:
    import base64

    # on_worker_removed callback receives the worker_id string (aligned
    # with the KvEventClient.on_worker_removed signature).
    removed: list[str] = []
    reg = Registry("localhost:0", on_worker_removed=lambda worker_id: removed.append(worker_id))
    reg._upsert(reg._prefix + "w1", _payload(worker_id="w1", canary=[1, 2, 3]))
    msg = {
        "result": {
            "events": [
                {
                    "type": "DELETE",
                    "kv": {"key": base64.b64encode((reg._prefix + "w1").encode()).decode()},
                }
            ]
        }
    }
    reg._dispatch_watch(msg)
    assert removed == ["w1"]


def test_callback_exception_does_not_break_watch_loop() -> None:
    """If a listener raises, the registry should log + continue, not crash."""

    def bad(_info):
        raise RuntimeError("listener bug")

    reg = Registry("localhost:0", on_worker_added=bad)
    # Should not raise.
    reg._upsert(reg._prefix + "w1", _payload(worker_id="w1", canary=[1, 2, 3]))
    # Worker still landed in the pool.
    assert reg.pool.get("w1") is not None
