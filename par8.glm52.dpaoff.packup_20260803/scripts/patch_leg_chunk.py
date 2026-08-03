#!/usr/bin/env python3
"""CHUNK_PASSTHROUGH: let an outer CHUNK= reach the leg.

start_leg.sh hardcodes ISL=8192 TP=8 in its docker-exec env block and never
passes CHUNK. glm52_leg.sh:73 then derives it:

    if [ "$DPA" = "1" ]; then CHUNK="${CHUNK:-$((ISL * TP))}"   # 65536
                        else CHUNK="${CHUNK:-8192}"; fi

sglang only divides chunked_prefill_size by dp_size when enable_dp_attention is
true (server_args.py:4902). So:

    DPA=1 -> 65536 passed, engine divides by 8 -> 8192 per rank
    DPA=0 ->  8192 passed, engine does NOT divide -> 8192 per forward

Both land on 8192, which is why the DPA-off solo run was comparable. But it also
means a DPA-off run CANNOT be given a larger chunk from outside: the outer
CHUNK= is dropped on the floor by start_leg.sh, silently, and the leg boots with
8192 while the operator believes it is running 16384.

This patch adds CHUNK to the env block with the same default-preserving idiom
used by DPA_PASSTHROUGH, so unset CHUNK keeps today's behaviour exactly.
"""
import re, sys, pathlib

p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "start_leg.sh")
s = p.read_text()

if "CHUNK_PASSTHROUGH" in s:
    print("already patched"); sys.exit(0)

# 1. declare CHUNK next to DPA, defaulting to empty (= leg derives it as today)
anchor = 'DPA="${DPA:-1}"'
if anchor not in s:
    print("ERROR: DPA anchor not found", file=sys.stderr); sys.exit(1)
s = s.replace(anchor, anchor + '\n'
    '# CHUNK_PASSTHROUGH: --chunked-prefill-size was underivable from outside.\n'
    '# Empty default = glm52_leg.sh derives it exactly as before (ISL*TP under\n'
    '# DPA, 8192 without), so unset CHUNK is a no-op. Set it only when the\n'
    '# derivation is wrong for the deployment -- notably DPA=0, where sglang no\n'
    '# longer divides by dp_size and the 8192 default is a per-forward value.\n'
    'CHUNK="${CHUNK:-}"', 1)

# 2. forward it into the container env
old = '  CTX=262144 ISL=8192 TP=8 DPA="$DPA" CUDA_GRAPH_BS=128 MAX_RUNNING=2048 \\'
if old not in s:
    print("ERROR: env block not found", file=sys.stderr); sys.exit(1)
s = s.replace(old,
    '  CTX=262144 ISL=8192 TP=8 DPA="$DPA" CUDA_GRAPH_BS=128 MAX_RUNNING=2048 \\\n'
    '  ${CHUNK:+CHUNK="$CHUNK"} \\', 1)

# 3. surface it in the launcher echo so the log records what was actually used
s = s.replace(
    'echo "[$TAG] $ROLE launched on $(hostname) mtp=$MTP ctx=262144 gmu=$GMU -> $LOG"',
    'echo "[$TAG] $ROLE launched on $(hostname) mtp=$MTP ctx=262144 gmu=$GMU dpa=$DPA chunk=${CHUNK:-<derived>} -> $LOG"', 1)

p.write_text(s)
print("patched OK - CHUNK_PASSTHROUGH occurrences:", s.count("CHUNK_PASSTHROUGH"))
