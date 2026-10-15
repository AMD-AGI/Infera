###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for GLM-5.3 thinking-budget constants (no SGLang import)."""

from __future__ import annotations

import json
import pickletools
import re
from pathlib import Path

from infera.engine.sglang.thinking_budget_const import (
    GLM53_NEW_LINE_TOKEN_ID,
    GLM53_THINKING_BUDGET_PROCESSOR_STR,
    GLM53_THINKING_END_TOKEN_ID,
    GLM53_THINKING_START_TOKEN_ID,
    is_glm53_model,
    resolve_thinking_budget,
    resolve_thinking_budget_action,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_is_glm53_model_matches_common_ids():
    assert is_glm53_model("glm-5-3")
    assert is_glm53_model("GLM_5.3-FP8")
    assert is_glm53_model("infera-glm53-1p1d")
    assert not is_glm53_model("qwen3-32b")
    assert not is_glm53_model(None)
    assert not is_glm53_model(123)


def test_resolve_thinking_budget_clamps_to_max_tokens_minus_one():
    assert resolve_thinking_budget(1024, 4096) == 1024
    assert resolve_thinking_budget(10000, 8192) == 8191
    assert resolve_thinking_budget(1, 2) == 1


def test_processor_str_targets_glm53_class():
    payload = json.loads(GLM53_THINKING_BUDGET_PROCESSOR_STR)
    hex_blob = payload["callable"]
    names = [
        arg
        for op, arg, _ in pickletools.genops(bytes.fromhex(hex_blob))
        if op.name == "SHORT_BINUNICODE"
    ]
    assert names == [
        "infera.engine.sglang.thinking_budget",
        "Glm53ThinkingBudgetLogitProcessor",
    ]


def test_processor_str_matches_rust_anthropic_constant():
    rust = (_REPO_ROOT / "rust/router/src/anthropic.rs").read_text(encoding="utf-8")
    match = re.search(
        r"const GLM53_THINKING_BUDGET_PROCESSOR_STR: &str = concat!\((.*?)\);",
        rust,
        flags=re.DOTALL,
    )
    assert match is not None, "Rust GLM53_THINKING_BUDGET_PROCESSOR_STR not found"
    rust_hex = "".join(re.findall(r'"([0-9a-f]+)"', match.group(1)))
    py_hex = json.loads(GLM53_THINKING_BUDGET_PROCESSOR_STR)["callable"]
    assert rust_hex == py_hex
    assert rust_hex


def test_glm53_token_ids_are_stable():
    # Documented GLM-5.3 special-token ids used by the hard-cap processor.
    assert GLM53_THINKING_START_TOKEN_ID == 154841
    assert GLM53_THINKING_END_TOKEN_ID == 154842
    assert GLM53_NEW_LINE_TOKEN_ID == 198


def test_budget_action_skips_when_history_closed_think_and_no_open_block():
    start, end, nl = 1, 2, 3
    # Multi-turn prompt with a closed thinking block and no open think.
    origin = [10, start, 11, end, 12]
    assert resolve_thinking_budget_action(origin, [20, 21], 2, start, end, nl) is None


def test_budget_action_counts_output_after_prompt_open_think():
    start, end, nl = 1, 2, 3
    # Template opened <think> in the prompt; count only generated tokens.
    origin = [10, start]
    assert resolve_thinking_budget_action(origin, [20], 2, start, end, nl) is None
    assert resolve_thinking_budget_action(origin, [20, 21], 2, start, end, nl) == "newline"
    assert resolve_thinking_budget_action(origin, [20, 21, nl], 2, start, end, nl) == "end"


def test_budget_action_uses_last_open_think_across_history():
    start, end, nl = 1, 2, 3
    # Closed history think, then a new open think in the prompt.
    origin = [start, 11, end, start]
    assert resolve_thinking_budget_action(origin, [20, 21], 2, start, end, nl) == "newline"


def test_budget_action_ignores_prompt_literal_without_output_exhaustion():
    start, end, nl = 1, 2, 3
    # Literal <think> early in the prompt must not force an immediate end.
    origin = [start, 10, 11, 12]
    assert resolve_thinking_budget_action(origin, [], 2, start, end, nl) is None
    assert resolve_thinking_budget_action(origin, [20], 2, start, end, nl) is None


def test_thinking_budget_module_imports_when_sglang_present():
    pytest = __import__("pytest")
    try:
        import sglang  # noqa: F401
    except ImportError:
        pytest.skip("sglang not installed")
    from infera.engine.sglang.thinking_budget import Glm53ThinkingBudgetLogitProcessor

    assert Glm53ThinkingBudgetLogitProcessor.THINKING_START_TOKEN_ID == 154841
    assert callable(Glm53ThinkingBudgetLogitProcessor())
