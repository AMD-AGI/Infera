#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Stop infera appending the decode radix cache when speculative decoding is on.

WHAT: with KV events enabled, `infera/engine/sglang/args.py` appends
`--disaggregation-decode-enable-radix-cache` to the decode leg so the router can
steer repeats to the rank holding the prefix. SGLang **hard-forbids** that flag
under speculative decoding:

    sglang/srt/arg_groups/pd_disaggregation_hook.py
        if server_args.speculative_algorithm is not None:
            raise ValueError(
                "--disaggregation-decode-enable-radix-cache is incompatible "
                "with speculative decoding (--speculative-algorithm EAGLE)")

so `kv-aware` and `EAGLE/MTP` cannot both be switched on: the decode leg dies in
argument parsing before it loads a single weight.

Not a conflict between the merged workstreams — no line of this changed. It is
pre-existing infera code meeting a configuration nobody had run: the kvaware/kvd
validation never enabled MTP, and the PD+DPA+MTP validation drove
`sglang.launch_server` directly, bypassing the infera wrapper that appends the
flag. The merge is simply the first time both switches are on at once.

DISTINCT from `patch_infera_decode_kvd_skip.py`, and neither subsumes the other.
That one is gated on **kvd** (`--infera-kvd-socket`) and stops
`--enable-hierarchical-cache` from being appended. This one is gated on
**kvaware** (`--enable-kv-events`). A decode leg running `KVAWARE=1 MTP=1` with
kvd off still needs this patch.

FIX: extend the existing gate (which already excludes non-mooncake backends for
the same "SGLang rejects it" reason) to also exclude speculative decoding, and
say so in the log rather than dropping the flag silently.

CONSEQUENCE, deliberately not hidden: on the decode leg SGLang then takes its
`else` branch and forces `disable_radix_cache = True` ("KV cache is forced as
chunk cache for decode server"). The decode leg therefore contributes little or
nothing to the router's KV view under MTP; prefix-aware routing runs on the
prefill-side view, which is where prefix reuse is actually decided. Measure the
decode-side `cache-view` block count rather than assuming either way.

BELONGS AS A SOURCE COMMIT, not a build-time patch: unlike the sglang diffs this
edits infera's own code. It is shipped as a script here only so the running
experiment containers can pick it up without a rebuild.

Self-locating and idempotent.
"""

import importlib.util
import sys
from pathlib import Path

_TAG = "[decode-radix-vs-mtp]"

_OLD = """    if (
        known.enable_kv_events
        and sglang_parsed.disaggregation_mode == "decode"
        and getattr(sglang_parsed, "disaggregation_transfer_backend", None) == "mooncake"
        and "--disaggregation-decode-enable-radix-cache" not in remaining
    ):
        remaining.append("--disaggregation-decode-enable-radix-cache")
"""

_NEW = '''    if (
        known.enable_kv_events
        and sglang_parsed.disaggregation_mode == "decode"
        and getattr(sglang_parsed, "disaggregation_transfer_backend", None) == "mooncake"
        and "--disaggregation-decode-enable-radix-cache" not in remaining
    ):
        # SGLang rejects the decode radix cache outright under speculative
        # decoding (pd_disaggregation_hook: "incompatible with speculative
        # decoding"), so appending it would kill an EAGLE/MTP decode leg during
        # argument parsing. Skip it and let SGLang fall back to its chunk cache;
        # prefix-aware routing then runs on the prefill-side view.
        if getattr(sglang_parsed, "speculative_algorithm", None) is not None:
            logger.info(
                "kv-events on, but --disaggregation-decode-enable-radix-cache is "
                "incompatible with --speculative-algorithm %s; not appending it. "
                "The decode leg will use SGLang's chunk cache and contribute "
                "little to the router KV view; prefix-aware routing runs on the "
                "prefill-side view.",
                sglang_parsed.speculative_algorithm,
            )
        else:
            remaining.append("--disaggregation-decode-enable-radix-cache")
'''


def _infera_dirs() -> list[Path]:
    """Every infera package copy that could be imported (see the bigram patch)."""
    roots: list[Path] = []
    seen: set[Path] = set()

    def _add(d: Path) -> None:
        d = d.resolve()
        if d.is_dir() and (d / "engine" / "sglang" / "args.py").is_file() and d not in seen:
            seen.add(d)
            roots.append(d)

    spec = importlib.util.find_spec("infera")
    if spec and spec.origin:
        _add(Path(spec.origin).parent)
    _add(Path("/opt/infera/infera"))
    return roots


def main() -> int:
    roots = _infera_dirs()
    if not roots:
        print(f"{_TAG} no infera package found — cannot patch")
        return 1

    touched = 0
    for root in roots:
        f = root / "engine" / "sglang" / "args.py"
        src = f.read_text()
        if "incompatible with --speculative-algorithm" in src:
            print(f"{_TAG} {root}: already present — skipping")
            continue
        if src.count(_OLD) != 1:
            print(f"{_TAG} anchor not found exactly once in {f}")
            print(f"{_TAG} infera drifted — re-cut the patch, nothing written")
            return 1
        f.write_text(src.replace(_OLD, _NEW, 1))
        print(f"{_TAG} patched {f}")
        touched += 1

    if touched:
        print(f"{_TAG} decode radix cache is no longer appended under speculative decoding")
    return 0


if __name__ == "__main__":
    sys.exit(main())
