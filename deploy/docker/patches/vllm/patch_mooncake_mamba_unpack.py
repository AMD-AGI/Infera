#!/usr/bin/env python3
"""Mooncake connector: don't assume a Mamba layer has exactly two state tensors.

register_kv_caches() destructures every MambaSpec layer as a 2-tuple:

    if isinstance(layer_spec, MambaSpec):
        conv, _ = cache_or_caches
        cache_list = [conv]

That holds for Mamba2 (conv state, SSM state) and fails for anything else.
Kimi-K3 uses KDA linear attention, whose layers carry a different number of state
tensors, so PD dies during KV registration for every rank at once:

    mooncake_connector.py:1678 in register_kv_caches
      conv, _ = cache_or_caches
    ValueError: too many values to unpack (expected 2)

MambaSpec.shapes is already a variable-length tuple of shapes, so the two-tensor
assumption is the connector's, not the spec's. Take the first state tensor —
which is what the original code kept — without constraining how many follow.

Self-locating and idempotent: re-running is a no-op, and it no-ops if upstream
fixes this, so the patch can stay in place across a base bump.
"""

import sys
from pathlib import Path

OLD = """            if isinstance(layer_spec, MambaSpec):
                conv, _ = cache_or_caches
                cache_list = [conv]"""

NEW = """            if isinstance(layer_spec, MambaSpec):
                # A Mamba-family layer does not necessarily carry exactly two
                # state tensors: that is Mamba2's shape (conv, ssm). KDA linear
                # attention (Kimi-K3) carries a different number, and `conv, _ =`
                # then raises "too many values to unpack" on every rank. Keep the
                # first state tensor, as before, without fixing the arity.
                cache_list = [cache_or_caches[0]]"""


def main() -> int:
    import vllm

    target = (
        Path(vllm.__file__).parent
        / "distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"
    )
    if not target.is_file():
        print(f"[patch] {target} not found; skipping")
        return 0
    src = target.read_text()
    if NEW in src:
        print("[patch] mooncake mamba unpack: already applied")
        return 0
    if OLD not in src:
        print("[patch] mooncake mamba unpack: anchor absent (upstream changed?); skipping")
        return 0
    target.write_text(src.replace(OLD, NEW))
    print(f"[patch] mooncake mamba unpack: applied to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
