###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for infera.common.tokenizer.resolve_tokenizer_path.

The resolver gives both consumers (front-door tokenize cache + kv-aware
BlockHasher) an unambiguous local path regardless of whether the operator
passed a local path or an HF model id. These tests cover:

  - local file passthrough (.json)
  - local dir passthrough
  - HF id → snapshot_download (mocked; no network in tests)
  - clear error when neither resolves
  - clear error when huggingface_hub is missing

We also pin the CLI contract (parse_server_args makes the flag required).
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from infera.common.tokenizer import resolve_tokenizer_path
from infera.server.args import parse_server_args

# ----------------------------------------------------------------------
# Local path passthrough
# ----------------------------------------------------------------------


def test_local_file_passthrough(tmp_path):
    tok = tmp_path / "tokenizer.json"
    tok.write_text("{}")
    assert resolve_tokenizer_path(str(tok)) == str(tok)


def test_local_dir_passthrough(tmp_path):
    assert resolve_tokenizer_path(str(tmp_path)) == str(tmp_path)


# ----------------------------------------------------------------------
# HF id → snapshot_download
# ----------------------------------------------------------------------


def test_hf_id_resolves_via_snapshot_download(tmp_path):
    """Non-local string is treated as HF id and dispatched to snapshot_download
    with allow_patterns restricted to tokenizer / chat_template files."""
    fake_snapshot_dir = str(tmp_path / "snapshots" / "abc123")
    captured = {}

    def fake_snapshot_download(*, repo_id, allow_patterns, **_):
        captured["repo_id"] = repo_id
        captured["allow_patterns"] = allow_patterns
        return fake_snapshot_dir

    # Inject a fake huggingface_hub module so the lazy import inside the
    # resolver picks up our stub instead of the real package.
    fake_hf_hub = type(sys)("huggingface_hub")
    fake_hf_hub.snapshot_download = fake_snapshot_download
    with patch.dict(sys.modules, {"huggingface_hub": fake_hf_hub}):
        out = resolve_tokenizer_path("Qwen/Qwen3-0.6B")
    assert out == fake_snapshot_dir
    assert captured["repo_id"] == "Qwen/Qwen3-0.6B"
    # Crucial: weights must NOT be pulled — restrict to tokenizer+chat_template.
    assert captured["allow_patterns"] == ["tokenizer*", "chat_template*"]


def test_hf_id_resolution_failure_is_wrapped(tmp_path):
    """snapshot_download raising must surface as a ValueError that names the
    offending value — bare HF stack traces are unhelpful to operators."""
    fake_hf_hub = type(sys)("huggingface_hub")

    def boom(**_):
        raise RuntimeError("404 — repo not found")

    fake_hf_hub.snapshot_download = boom
    with patch.dict(sys.modules, {"huggingface_hub": fake_hf_hub}):
        with pytest.raises(ValueError, match="not-a-real/repo"):
            resolve_tokenizer_path("not-a-real/repo")


def test_resolver_errors_when_hf_hub_missing(tmp_path):
    """huggingface_hub not installed + non-local path → clear ValueError."""
    # Force the inner `from huggingface_hub import snapshot_download` to fail.
    with patch.dict(sys.modules, {"huggingface_hub": None}):
        with pytest.raises(ValueError, match="huggingface_hub is missing"):
            resolve_tokenizer_path("Qwen/Qwen3-0.6B")


# ----------------------------------------------------------------------
# CLI contract: --router-tokenizer-path is required
# ----------------------------------------------------------------------


def test_router_tokenizer_path_is_required():
    """Missing --router-tokenizer-path must fail argument parsing.

    Tokenization is always on at the front door, so declaring the tokenizer
    up-front avoids the silent "short model name → cache_hits=0" trap.
    """
    with pytest.raises(SystemExit):
        parse_server_args(["--etcd-endpoint", "127.0.0.1:2379"])


def test_router_tokenizer_path_accepts_value():
    args = parse_server_args(
        [
            "--etcd-endpoint",
            "127.0.0.1:2379",
            "--router-tokenizer-path",
            "Qwen/Qwen3-0.6B",
        ]
    )
    assert args.router_tokenizer_path == "Qwen/Qwen3-0.6B"
