#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Apply request-scoped custom logit processors in the EAGLE verify step."""

import argparse
import importlib.util
import py_compile
from pathlib import Path

MARKER = "_infera_apply_custom_logit_processor"

HELPER = '''_INFERA_SPEC_PARAM_KEYS = ("__spec_num_tokens__", "__spec_candidates__", "__spec_seq_len__")
_infera_topk_warned = False


def _infera_apply_custom_logit_processor(next_token_logits, sampling_info, batch, verify_input):
    """Run custom logit processors over the rows of a chain verify block.

    Each request owns draft_token_num rows. The draft chain and committed length
    are exposed to the processor through the request's custom params for the
    duration of the call. Tree drafts (topk > 1) are left untouched.
    """
    global _infera_topk_warned
    if verify_input.tree_topk != 1:
        if not _infera_topk_warned:
            _infera_topk_warned = True
            logger.warning(
                "custom logit processors are not applied under EAGLE with "
                "--speculative-eagle-topk > 1"
            )
        return
    from sglang.srt.layers.sampler import apply_custom_logit_processor

    num_tokens = verify_input.draft_token_num
    candidates = verify_input.draft_token.reshape(len(batch.seq_lens), num_tokens)
    stashed = []
    for i, params in enumerate(sampling_info.custom_params):
        if params is None:
            continue
        params["__spec_num_tokens__"] = num_tokens
        params["__spec_candidates__"] = candidates[i]
        params["__spec_seq_len__"] = batch.seq_lens[i]
        stashed.append(params)
    try:
        apply_custom_logit_processor(
            next_token_logits, sampling_info, num_tokens_in_batch=num_tokens
        )
    finally:
        for params in stashed:
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
    if source.count(DEF_ANCHOR) != 1 or source.count(CALL_ANCHOR) != 1:
        raise ValueError("unexpected eagle_sample layout; update or disable this patch")
    if "\nlogger = logging.getLogger(__name__)\n" not in source:
        raise ValueError("eagle_utils has no module logger")
    patched = source.replace(DEF_ANCHOR, HELPER + DEF_ANCHOR, 1)
    patched = patched.replace(CALL_ANCHOR, CALL_ANCHOR + CALL, 1)
    compile(patched, "eagle_utils.py", "exec")
    return patched


def apply(target: Path) -> bool:
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
    print(f"[eagle-custom-logit] {'patched' if changed else 'already present'}: {target}")


if __name__ == "__main__":
    main()
