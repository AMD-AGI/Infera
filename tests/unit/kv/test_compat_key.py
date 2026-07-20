###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import pytest

from infera.kv.compat_key import (
    CANARY_STRING,
    compute_compat_key,
    compute_tokenizer_digest,
    tokenize_canary,
)
from infera.kv.types import AttentionGroup, AttentionRole


def _ag_full(family: str = "qwen3") -> AttentionGroup:
    return AttentionGroup(family=family, role=AttentionRole.INDEXABLE)


def test_compat_key_is_16_hex() -> None:
    k = compute_compat_key(
        model_id="Qwen/Qwen3-0.6B",
        kv_dtype="bfloat16",
        quant=None,
        index_block_size=64,
        attention_groups=[_ag_full()],
        tokenizer_digest="0123456789abcdef",
    )
    assert len(k) == 16
    assert all(c in "0123456789abcdef" for c in k)


def test_compat_key_deterministic() -> None:
    args = dict(
        model_id="Qwen/Qwen3-0.6B",
        kv_dtype="bfloat16",
        quant=None,
        index_block_size=64,
        attention_groups=[_ag_full()],
        tokenizer_digest="0123456789abcdef",
    )
    assert compute_compat_key(**args) == compute_compat_key(**args)


def test_compat_key_canonical_json_order_invariant() -> None:
    # Different ways of constructing equivalent input → same key.
    k1 = compute_compat_key(
        model_id="m",
        kv_dtype="bfloat16",
        quant=None,
        index_block_size=64,
        attention_groups=[
            AttentionGroup(
                family="m",
                role=AttentionRole.INDEXABLE,
                layer_indices=(2, 0, 1),
            )
        ],
        tokenizer_digest="abc",
    )
    k2 = compute_compat_key(
        model_id="m",
        kv_dtype="bfloat16",
        quant=None,
        index_block_size=64,
        attention_groups=[
            AttentionGroup(
                family="m",
                role=AttentionRole.INDEXABLE,
                layer_indices=(0, 1, 2),
            )
        ],
        tokenizer_digest="abc",
    )
    assert k1 == k2


def test_compat_key_sensitivity() -> None:
    base = dict(
        model_id="m",
        kv_dtype="bfloat16",
        quant=None,
        index_block_size=64,
        attention_groups=[_ag_full()],
        tokenizer_digest="abc",
    )
    base_key = compute_compat_key(**base)

    # Each of the included fields must change the key.
    for field, new in [
        ("model_id", "m2"),
        ("kv_dtype", "float16"),
        ("quant", "fp8-w8a8"),
        ("index_block_size", 32),
        ("tokenizer_digest", "abd"),
    ]:
        changed = dict(base)
        changed[field] = new
        assert compute_compat_key(**changed) != base_key, f"{field} should affect key"


def test_compat_key_attention_groups_sensitivity() -> None:
    base = dict(
        model_id="m",
        kv_dtype="bfloat16",
        quant=None,
        index_block_size=64,
        attention_groups=[_ag_full()],
        tokenizer_digest="abc",
    )
    base_key = compute_compat_key(**base)
    # Different attention groups → different key.
    different = dict(base)
    different["attention_groups"] = [
        _ag_full(),
        AttentionGroup(
            family="m",
            role=AttentionRole.SLIDING,
            window=128,
        ),
    ]
    assert compute_compat_key(**different) != base_key


def test_compat_key_excludes_tp_pp_engine_version() -> None:
    """compat_key must NOT depend on tp/pp/engine_version. We don't pass them
    to the function in the first place — guard by checking the signature
    via inspect."""
    import inspect

    sig = inspect.signature(compute_compat_key)
    params = set(sig.parameters.keys())
    forbidden = {"tp", "pp", "rank", "engine_version", "infera_version"}
    assert not (params & forbidden), f"compat_key must not include {params & forbidden}"


def test_tokenizer_digest_file(tmp_path) -> None:
    p = tmp_path / "tokenizer.json"
    p.write_text("hello world")
    d = compute_tokenizer_digest(p)
    assert len(d) == 16
    # Determinism.
    assert compute_tokenizer_digest(p) == d
    # Sensitivity.
    p.write_text("hello world!")
    assert compute_tokenizer_digest(p) != d


def test_tokenizer_digest_dir(tmp_path) -> None:
    # Hashes are sensitive to file content and to the set of files.
    d = tmp_path / "tok"
    d.mkdir()
    (d / "tokenizer.json").write_text("a")
    (d / "special_tokens_map.json").write_text("b")
    digest = compute_tokenizer_digest(d)
    assert len(digest) == 16

    # Adding a hidden file (starts with .) doesn't affect the digest.
    (d / ".cache").write_text("noise")
    assert compute_tokenizer_digest(d) == digest

    # Changing a file does.
    (d / "tokenizer.json").write_text("aa")
    assert compute_tokenizer_digest(d) != digest


def test_tokenizer_digest_missing_path_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_tokenizer_digest(tmp_path / "nothing-here.json")


def test_tokenizer_digest_dir_ignores_weights(tmp_path) -> None:
    # A model directory holds tokenizer files alongside huge weight shards.
    # The digest must depend ONLY on tokenizer files, never weights —
    # otherwise it would (slowly) hash hundreds of GB. Regression for the
    # Kimi case: tiktoken.model + tokenization_kimi.py, no tokenizer.json.
    d = tmp_path / "model"
    d.mkdir()
    (d / "tiktoken.model").write_text("ranks")
    (d / "tokenizer_config.json").write_text("{}")
    (d / "tokenization_kimi.py").write_text("class TikTokenTokenizer: ...")
    (d / "config.json").write_text('{"hidden_size": 1}')
    (d / "model-00001-of-00002.safetensors").write_bytes(b"\x00" * 4096)
    (d / "model-00002-of-00002.safetensors").write_bytes(b"\x01" * 4096)

    digest = compute_tokenizer_digest(d)
    assert len(digest) == 16

    # Mutating a weight shard must NOT change the digest.
    (d / "model-00001-of-00002.safetensors").write_bytes(b"\xff" * 8192)
    assert compute_tokenizer_digest(d) == digest

    # Mutating a tokenizer file MUST change the digest.
    (d / "tiktoken.model").write_text("ranks-v2")
    assert compute_tokenizer_digest(d) != digest


def test_tokenizer_digest_dir_no_tokenizer_files_raises(tmp_path) -> None:
    # A directory with only weights / unrelated files has no tokenizer to
    # fingerprint; fail loudly rather than return a degenerate empty hash.
    d = tmp_path / "weights-only"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"\x00" * 16)
    (d / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        compute_tokenizer_digest(d)


def test_canary_string_is_stable() -> None:
    # If this test fails, you bumped CANARY_STRING. That's a wire-protocol
    # change that invalidates every registered worker. Bump event_version
    # and document the migration in 19-trust-and-deployment.md.
    assert isinstance(CANARY_STRING, str)
    assert len(CANARY_STRING) > 20  # not trivial
    # Spot-check a few markers must be present
    assert "Hello" in CANARY_STRING
    assert "你好" in CANARY_STRING
    assert "<|endoftext|>" in CANARY_STRING


def test_tokenize_canary_with_stub() -> None:
    """tokenize_canary doesn't depend on a real tokenizer; only the encode method."""

    class StubTokenizer:
        calls: list[tuple[str, bool]] = []

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            self.calls.append((text, add_special_tokens))
            return [1, 2, 3, 4]

    stub = StubTokenizer()
    ids = tokenize_canary(stub)
    assert ids == [1, 2, 3, 4]
    # Must call with add_special_tokens=False per the design rationale.
    assert stub.calls == [(CANARY_STRING, False)]
