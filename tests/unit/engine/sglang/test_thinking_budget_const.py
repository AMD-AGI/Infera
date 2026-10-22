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

import pytest

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


def test_resolve_thinking_budget_leaves_room_for_close_and_answer():
    assert resolve_thinking_budget(1024, 4096) == 1024
    assert resolve_thinking_budget(10000, 8192) == 8192 - 2 - 1024
    assert resolve_thinking_budget(31999, 32000) == 32000 - 2 - 1024
    assert resolve_thinking_budget(1, 4) == 1
    assert resolve_thinking_budget(100, 10) == 10 - 2 - 2


@pytest.mark.parametrize(
    "budget_tokens,max_tokens",
    [(31999, 32000), (100000, 32000), (10, 10), (5, 20), (4, 4), (4096, 4096)],
)
def test_hard_cap_closes_thinking_and_leaves_answer_room(budget_tokens, max_tokens):
    start, end, nl = 100, 101, 102
    prompt = [1, 2, start]
    budget = resolve_thinking_budget(budget_tokens, max_tokens)
    output: list[int] = []
    while len(output) < max_tokens:
        action = resolve_thinking_budget_action(prompt, output, budget, start, end, nl)
        output.append(nl if action == "newline" else end if action == "end" else 7)
        if output[-1] == end:
            break
    assert end in output
    assert max_tokens - len(output) >= 1


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


class _CountingList(list):
    """List that counts element reads so tests can detect prompt rescans."""

    reads = 0

    def __iter__(self):
        type(self).reads += len(self)
        return super().__iter__()

    def __getitem__(self, key):
        type(self).reads += 1
        return super().__getitem__(key)


def _reference_action(origin, output, budget, start, end, nl):
    """Full-scan reference matching the original semantics."""
    ids = [*origin, *output]
    start_index = -1
    for i, tok in enumerate(ids):
        if tok == start:
            start_index = i
        elif tok == end:
            start_index = -1
    if start_index < 0:
        return None
    count = len(output) if start_index < len(origin) else len(ids) - start_index - 1
    if count < budget:
        return None
    return "newline" if not output or output[-1] != nl else "end"


def test_incremental_state_matches_full_scan():
    import random

    start, end, nl = 100, 101, 102
    rng = random.Random(7)
    for _ in range(300):
        origin = [rng.choice([1, 2, start, end, nl]) for _ in range(rng.randint(0, 12))]
        budget = rng.randint(0, 6)
        state: dict = {}
        output: list[int] = []
        for _ in range(rng.randint(0, 25)):
            output.append(rng.choice([7, 8, start, end, nl]))
            got = resolve_thinking_budget_action(origin, output, budget, start, end, nl, state)
            assert got == _reference_action(origin, output, budget, start, end, nl)


def test_incremental_state_does_not_rescan_prompt():
    start, end, nl = 100, 101, 102
    origin = _CountingList([1, 2, 3, start] + [5] * 5000)
    state: dict = {}
    output: list[int] = []
    resolve_thinking_budget_action(origin, output, 1000, start, end, nl, state)
    reads_after_first = _CountingList.reads
    for tok in range(50):
        output.append(7 + tok)
        resolve_thinking_budget_action(origin, output, 1000, start, end, nl, state)
    assert _CountingList.reads == reads_after_first


def test_incremental_state_recovers_when_output_is_reset():
    start, end, nl = 100, 101, 102
    origin = [1, start]
    state: dict = {}
    assert resolve_thinking_budget_action(origin, [7, 8, 9], 3, start, end, nl, state) == "newline"
    assert resolve_thinking_budget_action(origin, [7], 3, start, end, nl, state) is None


# ---------------------------------------------------------------------------
# EAGLE verify: one request owns several logits rows (chain draft, topk == 1)
# ---------------------------------------------------------------------------

_S = GLM53_THINKING_START_TOKEN_ID
_E = GLM53_THINKING_END_TOKEN_ID
_NL = GLM53_NEW_LINE_TOKEN_ID


def _spec(origin, output, committed, cands, budget, history=(), state=None):
    from infera.engine.sglang.thinking_budget_const import resolve_thinking_budget_spec_actions

    return resolve_thinking_budget_spec_actions(
        origin_input_ids=origin,
        output_ids=output,
        candidates=cands,
        seq_len=len(origin) + committed - 1,
        history=list(history),
        thinking_budget=budget,
        start_id=_S,
        end_id=_E,
        newline_id=_NL,
        state=state,
    )


def test_spec_actions_far_from_budget_are_all_none():
    assert _spec([5, _S], [1, 2], 2, [2, 7, 8, 9], 100) == [None] * 4


def test_spec_actions_force_newline_then_end_inside_one_block():
    assert _spec([5, _S], [1, 2], 2, [2, 7, 8, _NL], 4) == [None, None, "newline", "end"]


def test_spec_actions_stop_forcing_after_a_natural_end_in_the_draft():
    assert _spec([5, _S], [1, 2], 2, [2, 7, _E, 9], 4) == [None] * 4


def test_spec_actions_count_committed_tokens_missing_from_output_ids():
    # Two tokens are committed but not yet in output_ids; the last one is the bonus.
    history = [([2, 3, 8, 6], 3)]
    assert _spec([5, _S], [1, 2], 4, [8, 7, 9, _NL], 6, history) == [
        None,
        None,
        "newline",
        "end",
    ]


def test_spec_actions_do_not_force_when_committed_tokens_cannot_be_recovered():
    # Two unseen committed tokens and no history: the first could be </think>.
    assert _spec([5, _S], [1, 2], 4, [_NL, 10, 11, 12], 3) == [None] * 4
    assert _spec([5, _S], [1, 2], 4, [9, 10, 11, 12], 3) == [None] * 4


def test_spec_actions_reuse_a_scan_state_without_mutating_it():
    from infera.engine.sglang.thinking_budget_const import resolve_thinking_budget_action

    state: dict = {}
    resolve_thinking_budget_action([5, _S], [1, 2], 4, _S, _E, _NL, state)
    snapshot = dict(state)
    with_state = _spec([5, _S], [1, 2], 2, [2, 7, 8, _NL], 4, state=state)
    assert state == snapshot
    assert with_state == _spec([5, _S], [1, 2], 2, [2, 7, 8, _NL], 4)


def test_spec_actions_treat_a_shorter_committed_len_as_no_lag():
    assert _spec([5, _S], [1, 2], 0, [2, 7, 8, _NL], 4) == [None, None, "newline", "end"]


def test_spec_actions_none_when_thinking_is_closed_or_budget_negative():
    assert _spec([], [_S, 1, _E], 3, [_E, 7, 8, 9], 1) == [None] * 4
    assert _spec([5, _S], [1, 2], 2, [2, 7, 8, 9], -1) == [None] * 4


def test_spec_actions_length_follows_candidates():
    assert len(_spec([5, _S], [1, 2], 2, [2, 7], 100)) == 2


def test_spec_actions_see_a_close_in_committed_tokens_missing_from_output_ids():
    # Visible output [1, 2]; the previous step committed [newline, </think>, 9].
    # seq_len grew by 3 over that step, so the chain [2, newline, </think>, 9]
    # contributed its two accepted drafts plus the new bonus token 9.
    history = [([2, _NL, _E, 9], 3)]
    assert _spec([5, _S], [1, 2], 5, [9, 10, 11, 12], 2, history) == [None] * 4


def test_spec_actions_rebuild_the_tail_across_several_steps():
    # Two unseen steps: [3, 4] then [_E, 9]; the close is two steps back.
    history = [([2, 3, 4, 8], 3), ([8, _E, 9, 6], 5)]
    assert _spec([5, _S], [1, 2], 7, [9, 10, 11, 12], 3, history) == [None] * 4


def test_spec_actions_fall_back_to_the_bonus_token_when_history_is_inconsistent():
    history = [([2, 3, 4, 8], 4)]
    # seq_len did not grow over the recorded step, so it cannot be decoded and only
    # the bonus token is known: count 3 at row 0, then the budget of 4 is reached.
    assert _spec([5, _S], [1, 2], 3, [9, 7, 8, 6], 4, history) == [
        None,
        "newline",
        "newline",
        "newline",
    ]


def test_spec_actions_open_marker_in_the_candidates_is_counted():
    assert _spec([5], [9], 1, [9, _S, 7, 8], 1) == [None, None, "newline", "newline"]


class _FakeStep:
    """Host-side view of one verify step; counts host copies."""

    def __init__(self, rows, seqs):
        self.rows = rows
        self.seqs = seqs
        self.copies = 0

    def cpu(self):
        self.copies += 1
        return self.rows, self.seqs


class _FakeLogits:
    """Records the forced token per row without needing torch."""

    def __init__(self, rows, vocab=200000):
        self.shape = (rows, vocab)
        self.forced = {}

    def __setitem__(self, key, value):
        row, col = key
        if isinstance(col, slice):
            self.forced[row] = None
        else:
            self.forced[row] = col


@pytest.fixture
def processor_cls(monkeypatch):
    """The processor class built on a stub SGLang base, with module state restored."""
    import sys
    import types

    import infera.engine.sglang as pkg

    name = "infera.engine.sglang.thinking_budget"
    saved_module = sys.modules.pop(name, None)
    saved_attr = pkg.__dict__.pop("thinking_budget", None)

    base_mod = types.ModuleType("sglang.srt.sampling.custom_logit_processor")

    class ThinkingBudgetLogitProcessor:
        pass

    base_mod.ThinkingBudgetLogitProcessor = ThinkingBudgetLogitProcessor
    for mod_name in ("sglang", "sglang.srt", "sglang.srt.sampling"):
        monkeypatch.setitem(sys.modules, mod_name, types.ModuleType(mod_name))
    monkeypatch.setitem(sys.modules, "sglang.srt.sampling.custom_logit_processor", base_mod)
    from infera.engine.sglang.thinking_budget import Glm53ThinkingBudgetLogitProcessor

    yield Glm53ThinkingBudgetLogitProcessor

    sys.modules.pop(name, None)
    pkg.__dict__.pop("thinking_budget", None)
    if saved_module is not None:
        sys.modules[name] = saved_module
    if saved_attr is not None:
        pkg.thinking_budget = saved_attr


def _spec_param(origin, output, budget, cands, seq_len, n=4, history=(), step=None, row=0):
    req = type("Req", (), {"origin_input_ids": origin, "output_ids": output})()
    return {
        "thinking_budget": budget,
        "__req__": req,
        "__spec_num_tokens__": n,
        "__spec_step__": step or _FakeStep([cands], [seq_len]),
        "__spec_row__": row,
        "__spec_history__": [(_FakeStep([chain], [seq]), 0) for chain, seq in history],
    }


def test_processor_forces_tokens_on_the_right_rows_of_a_spec_block(processor_cls):
    proc = processor_cls()
    # Request 0 hits the budget at row 2; request 1 is far from it.
    # seq_len = prompt_len + committed_outputs - 1 with the bonus token excluded.
    p0 = _spec_param([5, _S], [1, 2], 4, [2, 7, 8, _NL], seq_len=2 + 2 - 1)
    p1 = _spec_param([5, _S], [1, 2], 100, [2, 7, 8, 9], seq_len=2 + 2 - 1)
    logits = _FakeLogits(rows=8)
    proc(logits, [p0, p1])
    assert logits.forced == {2: _NL, 3: _E}


def test_processor_without_spec_keys_keeps_single_row_behaviour(processor_cls):
    proc = processor_cls()
    req = type("Req", (), {"origin_input_ids": [5, _S], "output_ids": [1, 2, 3, 4]})()
    logits = _FakeLogits(rows=1)
    proc(logits, [{"thinking_budget": 4, "__req__": req}])
    assert logits.forced == {0: _NL}


def test_processor_skips_a_block_whose_row_count_does_not_match(processor_cls):
    proc = processor_cls()
    p = _spec_param([5, _S], [1, 2], 4, [2, 7, 8, _NL], seq_len=3)
    logits = _FakeLogits(rows=5)
    proc(logits, [p])
    assert logits.forced == {}


def test_processor_one_row_block_uses_the_committed_bonus_token(processor_cls):
    proc = processor_cls()
    # Visible output [1, 2], committed bonus 7, budget 3: the next token must be a newline.
    p = _spec_param([5, _S], [1, 2], 3, [7], seq_len=2 + 3 - 1, n=1)
    logits = _FakeLogits(rows=1)
    proc(logits, [p])
    assert logits.forced == {0: _NL}


def test_processor_reads_back_when_the_open_marker_is_unseen(processor_cls):
    proc = processor_cls()
    p = _spec_param([5], [9], 1, [9, _S, 7, 8], seq_len=1 + 1 - 1)
    logits = _FakeLogits(rows=4)
    proc(logits, [p])
    assert logits.forced == {2: _NL, 3: _NL}


def test_processor_sees_a_think_block_opened_inside_unseen_tokens(processor_cls):
    proc = processor_cls()
    # No <think> in the prompt; the model opened one in the previous step, which is
    # not in output_ids yet. Budget 2 is reached inside this block.
    history = [([9, _S, 7, 8], 1)]
    p = _spec_param([5], [9], 2, [8, 10, 11, 12], seq_len=1 + 3 - 1, history=history)
    logits = _FakeLogits(rows=4)
    proc(logits, [p])
    assert logits.forced == {1: _NL, 2: _NL, 3: _NL}


def test_processor_uses_history_to_see_a_close_in_unseen_tokens(processor_cls):
    proc = processor_cls()
    history = [([2, _NL, _E, 9], 3)]
    p = _spec_param([5, _S], [1, 2], 2, [9, 10, 11, 12], seq_len=2 + 5 - 1, history=history)
    logits = _FakeLogits(rows=4)
    proc(logits, [p])
    assert logits.forced == {}


def test_processor_skips_the_readback_when_the_budget_is_out_of_reach(processor_cls):
    proc = processor_cls()
    step = _FakeStep([[2, 7, 8, 9]], [3])
    p = _spec_param([5, _S], [1, 2], 1000, [2, 7, 8, 9], seq_len=3, step=step)
    proc(_FakeLogits(rows=4), [p])
    assert step.copies == 0


def test_processor_shares_one_host_copy_between_requests_of_a_step(processor_cls):
    proc = processor_cls()
    # One step object for the whole batch; the host copy is made by the step, not per request.
    step = _FakeStep([[2, 7, 8, _NL], [2, 7, 8, _NL]], [3, 3])
    p0 = _spec_param([5, _S], [1, 2], 4, None, None, step=step, row=0)
    p1 = _spec_param([5, _S], [1, 2], 4, None, None, step=step, row=1)
    logits = _FakeLogits(rows=8)
    proc(logits, [p0, p1])
    assert logits.forced == {2: _NL, 3: _E, 6: _NL, 7: _E}


def test_processor_declares_spec_row_support(processor_cls):
    assert processor_cls.SUPPORTS_SPEC_VERIFY_ROWS is True


@pytest.mark.parametrize(
    ("algorithm", "topk", "expected"),
    [
        ("EAGLE", 2, True),
        ("EAGLE3", 4, True),
        ("NEXTN", 2, True),
        ("EAGLE", 1, False),
        ("EAGLE", None, False),
        (None, 4, False),
        ("NGRAM", 4, False),
    ],
)
def test_eagle_topk_disables_thinking_budget(algorithm, topk, expected):
    from infera.engine.sglang.thinking_budget_const import eagle_topk_disables_thinking_budget

    assert eagle_topk_disables_thinking_budget(algorithm, topk) is expected
