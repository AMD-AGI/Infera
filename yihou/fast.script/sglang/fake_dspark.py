"""Allow synthetic local handoff in the pinned SGLang DSpark PD validator."""

from __future__ import annotations

import inspect
import runpy
import sys
from types import FunctionType

_ORIGINAL = (
    '            or cfg.disaggregation_transfer_backend not in ("mooncake", "nixl", "mori")\n'
)
_FAKE_DECODE = (
    '            or (cfg.disaggregation_transfer_backend not in ("mooncake", "nixl", "mori")'
    ' and not (cfg.disaggregation_mode == "decode"'
    ' and cfg.disaggregation_transfer_backend == "fake"))\n'
)


def allow_fake_decode(validator: FunctionType) -> None:
    lines, first_line = inspect.getsourcelines(validator)
    source = "".join(lines)
    if source.count(_ORIGINAL) != 1:
        raise RuntimeError("Unsupported SGLang DSpark PD validator source.")

    # Keep traceback line numbers and the original object registered by hooks.
    source = "\n" * (first_line - 1) + source.replace(_ORIGINAL, _FAKE_DECODE)
    patched = {}
    code = compile(source, validator.__code__.co_filename, "exec")
    exec(code, validator.__globals__, patched)
    validator.__code__ = patched[validator.__name__].__code__


def main() -> None:
    args = sys.argv[1:]
    for option, expected in (
        ("--disaggregation-mode", "decode"),
        ("--disaggregation-transfer-backend", "fake"),
        ("--speculative-algorithm", "DSPARK"),
    ):
        if args.count(option) == 1:
            index = args.index(option) + 1
            if args[index : index + 1] == [expected]:
                continue
        raise SystemExit(f"Fake DSpark launcher requires {option} {expected}.")

    from sglang.srt.arg_groups.deepseek_v4_hook import validate_deepseek_v41_features

    allow_fake_decode(validate_deepseek_v41_features)
    print("Enabled DSpark validator compatibility for local fake-PD decode.", file=sys.stderr)
    runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
