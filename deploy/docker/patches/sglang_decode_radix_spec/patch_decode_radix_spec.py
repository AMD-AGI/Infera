#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Add the explicit EAGLE/NEXTN top-k-1 Decode radix opt-in to SGLang."""

import argparse
import importlib.util
import py_compile
from pathlib import Path

HELPER = """def _infera_decode_radix_spec_allowed(cfg) -> bool:
    return (
        os.environ.get("SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC", "0") == "1"
        and str(cfg.speculative_algorithm).upper() in ("EAGLE", "NEXTN")
        and cfg.speculative_eagle_topk == 1
    )


"""


def patch_source(source: str) -> str:
    for name in ("server_args", "cfg"):
        guard = f"            if {name}.speculative_algorithm is not None:\n"
        gated = (
            f"            if {name}.speculative_algorithm is not None and not "
            f"_infera_decode_radix_spec_allowed({name}):\n"
        )
        rejection = (
            "                raise ValueError(\n"
            '                    "--disaggregation-decode-enable-radix-cache is incompatible "\n'
            '                    "with speculative decoding "\n'
            f'                    f"(--speculative-algorithm {{{name}.speculative_algorithm}})"\n'
            "                )\n"
        )
        if HELPER in source and source.count(gated + rejection) == 1:
            return source
        if source.count(guard + rejection) == 1:
            if "_infera_decode_radix_spec_allowed" in source:
                raise ValueError("partial Decode radix/spec patch; refusing to guess")
            anchor = "def handle_pd_disaggregation(server_args: ServerArgs) -> None:\n"
            if source.count(anchor) != 1 or "\nimport os\n" not in source:
                raise ValueError("unexpected SGLang PD hook layout")
            patched = source.replace(guard + rejection, gated + rejection, 1)
            patched = patched.replace(anchor, HELPER + anchor, 1)
            compile(patched, "pd_disaggregation_hook.py", "exec")
            return patched
    raise ValueError("unsupported SGLang Decode radix guard; update or disable this patch")


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


def installed_hook() -> Path:
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("cannot locate the installed SGLang package")
    return (
        Path(next(iter(spec.submodule_search_locations)))
        / "srt/arg_groups/pd_disaggregation_hook.py"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, help="override the installed PD hook path")
    args = parser.parse_args()
    target = args.target if args.target is not None else installed_hook()
    changed = apply(target)
    print(f"[decode-radix-spec] {'patched' if changed else 'already present'}: {target}")


if __name__ == "__main__":
    main()
