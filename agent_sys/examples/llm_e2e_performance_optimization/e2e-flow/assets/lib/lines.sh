#!/bin/sh
# Who is running what, and is a card set free. **Use this instead of composing
# the check inline.**
#
# **Why this exists.** The check "is this node free" was written ad hoc at least
# five times, and **four of them silently matched their own command line** and
# returned a plausible wrong answer:
#
#   a `pgrep` loop whose body contained the node name      -> false positive
#   a loop whose body contained the run pid                -> false positive
#   `ps --forest | grep` matching its own `eval` string    -> false positive
#   `ps -eo args | grep -F node=… | grep -F gpu=… | grep -c`
#        -> `2`, which is the two greps in the pipeline. Four idle cards would
#           have sat another cycle if that had been believed.
#
# Every one produced an answer that looked right. The conclusion worth keeping:
# **the fix is not knowing better, it is never composing the check inline** — a
# rule that changes the command rather than asking someone to remember
# something mid-pipeline.
#
# Usage:
#   sh assets/lib/lines.sh                 # every live chain: pid, age, node, cards
#   sh assets/lib/lines.sh <node>          # only that node
#   sh assets/lib/lines.sh <node> 4        # does a line DECLARE that half? rc=0 yes, rc=1 no
#
# **The `DECLARES` column is what a line asked for, NOT what is occupied**, and
# the two can differ: a line declaring `gpu_devices=4,5,6,7` while its engine
# sits on cards 0-3 at 232 GB each is opposite sets. Reading this script's output
# as occupancy then reports the occupied half as free, and only a `rocm-smi` at
# the precondition stops a bring-up landing on top of a live chain.
#
# **So `rc=1` means "nobody declared it", not "the cards are free."** For the
# occupancy question there is exactly one authority and it is on the node:
#
#     spur exec <jobid> rocm-smi --showmemuse | grep -E "^GPU\[[0-7]\].*VRAM%"
#
# Use this script to find collisions between declarations. Use `rocm-smi` before
# you put an engine anywhere.
#
# `rc` is the answer for the third form, so it composes into `if`/`&&` without
# anyone re-reading the output.
set -eu

NODE="${1:-}"
HALF="${2:-}"

# **One `ps`, filtered in awk.** No `grep` in the pipeline at all, so there is
# nothing for the check to match itself against — that is the whole point and it
# is why this is not `ps | grep | grep -v grep`, which works but leaves the trap
# one edit away.
_lines() {
    ps -eo pid=,etime=,args= | awk -v want="$NODE" '
        index($0, "run_with_long_stall") == 0 { next }
        {
            pid = $1; age = $2
            node = ""; cards = ""; mock = ""; pkg = ""
            for (i = 3; i <= NF; i++) {
                if ($i ~ /^node=/)         { node  = substr($i, 6) }
                if ($i ~ /^gpu_devices=/)  { cards = substr($i, 13) }
                if ($i ~ /^mock_stages=/)  { mock  = substr($i, 13) }
                if ($i == "--package")     { pkg   = $(i+1) }
            }
            # A row with no `node=` is a process caught mid-fork -- the last
            # residue of the self-match class this file exists to kill. Drop it
            # rather than print a blank line someone has to interpret.
            if (node == "") next
            if (want != "" && node != want) next
            n = split(pkg, parts, "/"); pkg = parts[n]
            printf "%-9s %-10s %-20s %-12s %-16s %s\n", pid, age, node, cards, mock, pkg
        }'
}

if [ -n "$HALF" ]; then
    # "is the half starting at card $HALF busy on $NODE" -- rc, not prose.
    if _lines | awk -v h="$HALF" '{ split($4, c, ","); if (c[1] == h) found = 1 }
                                  END { exit(found ? 0 : 1) }'; then
        echo "DECLARED by a live line: $NODE cards ${HALF}-.." >&2
        exit 0
    fi
    echo "not declared by any live line: $NODE cards ${HALF}-.." >&2
    echo "  NOT the same as free -- a line can occupy cards it did not declare." >&2
    echo "  Check the cards: spur exec <jobid> rocm-smi --showmemuse" >&2
    exit 1
fi

printf '%-9s %-10s %-20s %-12s %-16s %s\n' PID AGE NODE DECLARES MOCK PACKAGE
_lines
