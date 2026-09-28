#!/usr/bin/env python3
"""CPU-only check of decode MTP + DPA + radix + HiCache at argument resolution.

Parses the GLM-5.2 decode command line (repro_decode_args.DECODE_ARGV plus the
harness's HiCache flags) with Infera, then runs SGLang's PD, HiCache and cache
compatibility hooks in pipeline order on the argv launch_server would receive.
No engine, GPU, or weights.

Usage (inside the infera-sglang image, model mounted read-only):
    python3 repro_decode_hicache_args.py
Set SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1 to exercise the patched opt-in.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from repro_decode_args import DECODE_ARGV, FLAG

HICACHE_ARGV = [
    "--enable-hierarchical-cache", "--hicache-ratio", "1.5",
    "--hicache-write-policy", "write_through", "--hicache-io-backend", "kernel",
    "--hicache-mem-layout", "page_first",
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from infera.engine.sglang.args import parse_sglang_args
    from sglang.srt.arg_groups.hicache_hook import handle_hicache
    from sglang.srt.arg_groups.kv_cache_hook import handle_cache_compatibility
    from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation
    from sglang.srt.server_args import ServerArgs

    args = parse_sglang_args([*DECODE_ARGV, *HICACHE_ARGV])
    out = {"flag_forwarded_to_sglang": FLAG in args.sglang_argv}
    p = argparse.ArgumentParser(add_help=False)
    ServerArgs.add_cli_args(p)
    sa = ServerArgs.from_cli_args(p.parse_args(args.sglang_argv))
    for hook in (handle_pd_disaggregation, handle_hicache, handle_cache_compatibility):
        try:
            hook(sa)
            out[hook.__name__] = "ok"
        except Exception as e:  # noqa: BLE001 - the error is the result
            out[hook.__name__] = f"{type(e).__name__}: {e}"
            break
    print("RESULT " + json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
