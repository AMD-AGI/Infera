###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The SGLang patch that applies custom logit processors in the EAGLE verify step."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from collections import deque
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PATCH = (
    _REPO_ROOT
    / "deploy/docker/patches/sglang_eagle_custom_logit/patch_eagle_custom_logit_processor.py"
)

_SOURCE = """import logging

logger = logging.getLogger(__name__)


def eagle_sample(
    verify_input,
    batch,
    logits_output,
    grammar_mask=None,
):
    sampling_info = batch.sampling_info
    next_token_logits = logits_output.next_token_logits

    # Apply grammar mask if provided
    if grammar_mask is not None:
        grammar_mask.apply(next_token_logits)

    candidates = 1
    return candidates
"""


@pytest.fixture(scope="module")
def patch_mod():
    spec = importlib.util.spec_from_file_location("eagle_custom_logit_patch", _PATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_patch_source_adds_the_helper_and_the_call(patch_mod):
    patched = patch_mod.patch_source(_SOURCE)
    assert patched.count("_infera_apply_custom_logit_processor(") == 2
    assert patched.index("grammar_mask.apply") < patched.rindex(
        "_infera_apply_custom_logit_processor("
    )
    compile(patched, "eagle_utils.py", "exec")


def test_patch_source_is_idempotent(patch_mod):
    once = patch_mod.patch_source(_SOURCE)
    assert patch_mod.patch_source(once) == once


def test_patch_source_refuses_a_partial_patch(patch_mod):
    broken = patch_mod.patch_source(_SOURCE).replace(patch_mod.CALL, "")
    with pytest.raises(ValueError, match="partial"):
        patch_mod.patch_source(broken)


def test_patch_source_leaves_a_module_without_eagle_sample_untouched(patch_mod):
    assert patch_mod.patch_source("x = 1\n") == "x = 1\n"


def test_patch_source_rejects_an_unknown_layout(patch_mod):
    with pytest.raises(ValueError, match="unexpected eagle_sample layout"):
        patch_mod.patch_source(_SOURCE.replace("# Apply grammar mask if provided", "# changed"))


# ---- the helper, executed against fakes ------------------------------------


class _Rows:
    """Tensor stand-in: 2-D rows or a flat list."""

    def __init__(self, data):
        self.data = data

    def reshape(self, bs, n):
        flat = self.data
        return _Rows([flat[i * n : (i + 1) * n] for i in range(bs)])

    def clone(self):
        return _Rows([list(r) if isinstance(r, list) else r for r in self.data])

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    def tolist(self):
        return self.data


class _Sup:
    SUPPORTS_SPEC_VERIFY_ROWS = True


class _Unsup:
    pass


class _ProcessorEntry:
    """Represent the ProcessorEntry shape used by current SGLang releases."""

    def __init__(self, processor):
        self.processor = processor


@pytest.fixture
def helper(patch_mod, monkeypatch):
    calls = []

    sampler = types.ModuleType("sglang.srt.layers.sampler")

    def apply_custom_logit_processor(logits, sampling_info, num_tokens_in_batch=1):
        calls.append(
            {
                "processors": [
                    type(value.processor if hasattr(value, "processor") else value[0])
                    for value in sampling_info.custom_logit_processor.values()
                ],
                "n": num_tokens_in_batch,
                "params": [None if p is None else dict(p) for p in sampling_info.custom_params],
            }
        )

    sampler.apply_custom_logit_processor = apply_custom_logit_processor
    for name in ("sglang", "sglang.srt", "sglang.srt.layers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.sampler", sampler)

    ns = {"logging": logging, "__name__": "eagle_utils_stub"}
    ns["logger"] = logging.getLogger("eagle_utils_stub")
    exec(patch_mod.patch_source(_SOURCE), ns)
    return ns["_infera_apply_custom_logit_processor"], calls, ns


def _call(fn, processors, params, topk=1, n=4, flat=None, seqs=None):
    bs = len(params)
    info = types.SimpleNamespace(
        custom_logit_processor={i: (p, object()) for i, p in enumerate(processors)},
        custom_params=params,
    )
    batch = types.SimpleNamespace(seq_lens=_Rows(seqs or [10] * bs))
    verify = types.SimpleNamespace(
        draft_token_num=n,
        tree_topk=topk,
        draft_token=_Rows(flat or list(range(bs * n))),
    )
    fn(object(), info, batch, verify)
    return info


def test_helper_passes_the_block_size_and_stashes_the_step(helper):
    fn, calls, _ = helper
    params = [{"a": 1}, None, {"b": 2}]
    _call(fn, [_Sup()], params, n=2, flat=list(range(6)), seqs=[5, 6, 7])
    assert calls[0]["n"] == 2
    stashed = calls[0]["params"][0]
    assert stashed["__spec_num_tokens__"] == 2
    assert stashed["__spec_row__"] == 0
    assert calls[0]["params"][2]["__spec_row__"] == 2
    assert stashed["__spec_step__"].cpu() == ([[0, 1], [2, 3], [4, 5]], [5, 6, 7])
    # The stash is removed afterwards; the step joins the history.
    assert set(params[0]) == {"a", "__spec_history__"}
    step, row = params[0]["__spec_history__"][0]
    assert row == 0 and step is stashed["__spec_step__"]
    assert params[1] is None


def test_helper_keeps_the_last_four_steps(helper):
    fn, _, _ = helper
    params = [{}]
    for _ in range(6):
        _call(fn, [_Sup()], params, n=1, flat=[7])
    assert isinstance(params[0]["__spec_history__"], deque)
    assert len(params[0]["__spec_history__"]) == 4


def test_helper_shares_one_host_copy_across_rows(helper):
    fn, calls, _ = helper
    params = [{}, {}]
    _call(fn, [_Sup()], params, n=2, flat=[1, 2, 3, 4], seqs=[5, 6])
    step = calls[0]["params"][0]["__spec_step__"]
    assert calls[0]["params"][1]["__spec_step__"] is step
    first = step.cpu()
    assert step.cpu() is first


def test_helper_only_runs_processors_that_support_spec_rows(helper):
    fn, calls, _ = helper
    info = _call(fn, [_Sup(), _Unsup()], [{}])
    assert calls[0]["processors"] == [_Sup]
    assert [type(p) for p, _ in info.custom_logit_processor.values()] == [_Sup, _Unsup]


def test_helper_supports_processor_entry_values(helper):
    fn, calls, _ = helper
    params = [{}]
    info = types.SimpleNamespace(
        custom_logit_processor={0: _ProcessorEntry(_Sup())},
        custom_params=params,
    )
    batch = types.SimpleNamespace(seq_lens=_Rows([10]))
    verify = types.SimpleNamespace(
        draft_token_num=1,
        tree_topk=1,
        draft_token=_Rows([7]),
    )

    fn(object(), info, batch, verify)

    assert calls[0]["processors"] == [_Sup]
    assert isinstance(info.custom_logit_processor[0], _ProcessorEntry)


def test_helper_skips_when_no_processor_supports_spec_rows(helper):
    fn, calls, _ = helper
    params = [{}]
    _call(fn, [_Unsup()], params)
    assert calls == []
    assert params == [{}]


def test_helper_skips_tree_drafts_and_warns_once(helper, caplog):
    fn, calls, _ = helper
    with caplog.at_level(logging.WARNING, logger="eagle_utils_stub"):
        _call(fn, [_Sup()], [{}], topk=4, n=4)
        _call(fn, [_Sup()], [{}], topk=4, n=4)
    assert calls == []
    assert sum("speculative-eagle-topk" in r.message for r in caplog.records) == 1


def test_helper_treats_a_one_row_block_as_a_chain_whatever_the_topk(helper):
    fn, calls, _ = helper
    _call(fn, [_Sup()], [{}], topk=4, n=1, flat=[7])
    assert calls[0]["n"] == 1
