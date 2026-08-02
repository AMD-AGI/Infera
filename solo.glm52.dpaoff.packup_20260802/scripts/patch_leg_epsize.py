#!/usr/bin/env python3
"""EP_DECOUPLE: make --ep-size independent of DP-attention in glm52_leg.sh.

WHY
    The DPA-off latency experiment must change EXACTLY ONE thing: how attention
    is sharded (DP8 -> pure TP8). As shipped, `--ep-size` lives inside the
    `if DPA = 1` branch, so DPA=0 silently drops it and sglang resolves
    ep_size 8 -> 1. That collapses MoE expert-parallelism at the same time,
    making the run a two-variable change whose result cannot be attributed to
    DPA at all. (Verified offline: no --ep-size flag -> ep_size = 1.)

WHAT
    Move `--ep-size "$TP"` out of the DPA branch so it is always passed.
    Behaviour with DPA=1 is byte-identical -- the same flag with the same value
    is still emitted, just from one line up. Only the DPA=0 path changes.

IDEMPOTENT
    Re-running is a no-op. Exits 2 if the anchor text has drifted.
"""
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "glm52_leg.sh"
src = open(PATH).read()

MARK = "EP_DECOUPLE"
if MARK in src:
    print(f"already patched ({MARK} present) - no change")
    sys.exit(0)

OLD = '''DP_ARGS=()
if [ "$DPA" = "1" ]; then
  DP_ARGS+=(--dp-size "$TP" --enable-dp-attention --ep-size "$TP")'''

NEW = '''DP_ARGS=()
# EP_DECOUPLE: --ep-size is passed UNCONDITIONALLY, not only under DP-attention.
# Rationale: the DPA-off latency experiment must vary attention sharding ALONE.
# Left inside the DPA branch, DPA=0 drops the flag and sglang resolves
# ep_size 8 -> 1, collapsing MoE expert-parallelism too -- a second variable
# that would make the measured delta unattributable. With DPA=1 this emits the
# exact same flag as before, so the DPA-on command line is unchanged.
DP_ARGS+=(--ep-size "$TP")
if [ "$DPA" = "1" ]; then
  DP_ARGS+=(--dp-size "$TP" --enable-dp-attention)'''

if OLD not in src:
    print("ANCHOR NOT FOUND - the leg script has drifted. Refusing to patch.",
          file=sys.stderr)
    sys.exit(2)

src = src.replace(OLD, NEW, 1)
open(PATH, "w").write(src)
print(f"patched OK - {MARK} occurrences:", src.count(MARK))
