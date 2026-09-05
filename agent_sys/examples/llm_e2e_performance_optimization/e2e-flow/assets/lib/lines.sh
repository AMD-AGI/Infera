#!/bin/sh
# Who is running what, and is a card set free. **Use this instead of composing
# the check inline.**
#
# **Why this exists.** On 2026-09-05 the check "is this node free" was written
# ad hoc at least five times by two people, and **four of them silently matched
# their own command line** and returned a plausible wrong answer:
#
#   m3   a `pgrep` loop whose body contained `088`            -> false positive
#   m3   a loop whose body contained the run pid              -> false positive
#   m3   `ps --forest | grep` matching its own `eval` string  -> false positive
#   m3   a cleanup loop containing `088`, ten minutes after   -> "2822004 still on 088"
#        writing their own entry in RUN-PLAN 3a about this
#   lead `ps -eo args | grep -F node=… | grep -F gpu_devices=4 | grep -c`
#        -> `2`, which is the two greps in the pipeline. Four idle cards would
#           have sat another cycle if I had believed it.
#
# Every one produced an answer that looked right. m3's conclusion is the one
# worth keeping: **the fix is not knowing better, it is never composing the
# check inline** — which is `CLAUDE.md`'s tier-1 shape, a rule that changes the
# command rather than asking someone to remember something mid-pipeline.
#
# Usage:
#   sh assets/lib/lines.sh                 # every live chain: pid, age, node, cards
#   sh assets/lib/lines.sh <node>          # only that node
#   sh assets/lib/lines.sh <node> 4        # is the 4,5,6,7 half busy? rc=0 busy, rc=1 free
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
        echo "busy: $NODE cards ${HALF}-.." >&2
        exit 0
    fi
    echo "free: $NODE cards ${HALF}-.." >&2
    exit 1
fi

printf '%-9s %-10s %-20s %-12s %-16s %s\n' PID AGE NODE CARDS MOCK PACKAGE
_lines
