#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Apply request-scoped custom logit processors in the EAGLE verify step."""

import argparse
import importlib.util
import py_compile
from pathlib import Path

MARKER = "_infera_apply_custom_logit_processor"

HELPER = '''_INFERA_SPEC_PARAM_KEYS = ("__spec_num_tokens__", "__spec_step__", "__spec_row__")
_INFERA_SPEC_HISTORY_STEPS = 4
_infera_topk_warned = False


class _InferaSpecStep:
    """Draft chains and sequence lengths of one verify step, copied to the host once."""

    def __init__(self, candidates, seq_lens):
        self._candidates = candidates
        self._seq_lens = seq_lens
        self._host = None

    def cpu(self):
        if self._host is None:
            self._host = (self._candidates.tolist(), self._seq_lens.tolist())
            self._candidates = self._seq_lens = None
        return self._host


def _infera_apply_custom_logit_processor(next_token_logits, sampling_info, batch, verify_input):
    """Run multi-row custom logit processors over the rows of a chain verify block.

    Each request owns draft_token_num rows. The current step (draft chains and
    sequence lengths of the batch) and the request's row in it are exposed to the
    processors through the request's custom params for the duration of the call.
    The same pair from previous steps is kept in __spec_history__. Only processors
    that set SUPPORTS_SPEC_VERIFY_ROWS run here. Tree drafts (topk > 1) are left
    untouched.
    """
    global _infera_topk_warned
    num_tokens = verify_input.draft_token_num
    if verify_input.tree_topk != 1 and num_tokens != 1:
        if not _infera_topk_warned:
            _infera_topk_warned = True
            logger.warning(
                "custom logit processors are not applied under EAGLE with "
                "--speculative-eagle-topk > 1"
            )
        return
    from collections import deque

    from sglang.srt.layers.sampler import apply_custom_logit_processor

    all_processors = sampling_info.custom_logit_processor
    supported = {
        key: value
        for key, value in all_processors.items()
        if getattr(value[0], "SUPPORTS_SPEC_VERIFY_ROWS", False)
    }
    if not supported:
        return
    step = _InferaSpecStep(
        verify_input.draft_token.reshape(len(batch.seq_lens), num_tokens).clone(),
        batch.seq_lens.clone(),
    )
    stashed = []
    for i, params in enumerate(sampling_info.custom_params):
        if params is None:
            continue
        params["__spec_num_tokens__"] = num_tokens
        params["__spec_step__"] = step
        params["__spec_row__"] = i
        if "__spec_history__" not in params:
            params["__spec_history__"] = deque(maxlen=_INFERA_SPEC_HISTORY_STEPS)
        stashed.append(params)
    sampling_info.custom_logit_processor = supported
    try:
        apply_custom_logit_processor(
            next_token_logits, sampling_info, num_tokens_in_batch=num_tokens
        )
    finally:
        sampling_info.custom_logit_processor = all_processors
        for params in stashed:
            params["__spec_history__"].append((step, params["__spec_row__"]))
            for key in _INFERA_SPEC_PARAM_KEYS:
                params.pop(key, None)


'''

DEF_ANCHOR = "def eagle_sample(\n"
CALL_ANCHOR = (
    "    # Apply grammar mask if provided\n"
    "    if grammar_mask is not None:\n"
    "        grammar_mask.apply(next_token_logits)\n"
    "\n"
)
CALL = (
    "    if sampling_info.has_custom_logit_processor:\n"
    f"        {MARKER}(\n"
    "            next_token_logits, sampling_info, batch, verify_input\n"
    "        )\n"
    "\n"
)


def patch_source(source: str) -> str:
    if MARKER in source:
        if source.count(f"{MARKER}(") != 2 or source.count(CALL_ANCHOR + CALL) != 1:
            raise ValueError("partial EAGLE custom logit patch; refusing to guess")
        return source
    if DEF_ANCHOR not in source:
        # No EAGLE verify step in this module: nothing to patch.
        return source
    if source.count(DEF_ANCHOR) != 1 or source.count(CALL_ANCHOR) != 1:
        raise ValueError("unexpected eagle_sample layout; update or disable this patch")
    if "\nlogger = logging.getLogger(__name__)\n" not in source:
        raise ValueError("eagle_utils has no module logger")
    patched = source.replace(DEF_ANCHOR, HELPER + DEF_ANCHOR, 1)
    patched = patched.replace(CALL_ANCHOR, CALL_ANCHOR + CALL, 1)
    compile(patched, "eagle_utils.py", "exec")
    return patched


def apply(target: Path) -> bool:
    if not target.exists():
        return False
    original = target.read_text()
    patched = patch_source(original)
    changed = patched != original
    if changed:
        target.write_text(patched)
    # Do not leave a timestamp-valid pyc from an earlier image layer.
    for cached in (target.parent / "__pycache__").glob(target.stem + ".*.pyc"):
        cached.unlink()
    py_compile.compile(
        str(target), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH
    )
    return changed


def installed_target() -> Path:
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("cannot locate the installed SGLang package")
    return Path(next(iter(spec.submodule_search_locations))) / "srt/speculative/eagle_utils.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, help="override the eagle_utils.py path")
    args = parser.parse_args()
    target = args.target if args.target is not None else installed_target()
    changed = apply(target)
    print(
        f"[eagle-custom-logit] {'patched' if changed else 'already present or not applicable'}: {target}"
    )


if __name__ == "__main__":
    main()
