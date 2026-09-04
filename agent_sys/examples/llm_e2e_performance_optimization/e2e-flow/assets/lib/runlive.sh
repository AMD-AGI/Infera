#!/bin/sh
# Which agent-sys RUNS were alive at the instant this ran, and for which run root.
#
# Companion to `runprobe.py` (what a run has escalated) and `read_events.py`
# (what a run's event store says). This one answers only: **was a run process
# present**, which neither of those can, because a run that is not running
# leaves an event store that looks exactly like a run that is merely quiet.
#
# ## Why it exists as a FILE
#
# It lived as a one-line shell snippet pasted into messages, and on 2026-09-04
# that cost a wrong published claim: the form that was *run* used a snapshot
# file, the form that was *pasted* used `awk … - -`, and those behave
# differently (see below). **A tool that is quoted rather than executed drifts
# between the quoting and the running.** There is now one form and it is this.
#
# ## The four defects this fixes, all measured 2026-09-04
#
# The naive `ps … | grep agent_sys.cli.main` has all four:
#
#   1 **FALSE NEGATIVE — `agent-sys run` is invisible.** `bin/agent-sys` is a
#     console script (`from cli.main import main`); its process line carries no
#     `agent_sys.cli.main` at all. Both launch forms are in live use —
#     `CLAUDE.md` writes `agent-sys run`, `CONTRACT.md` §9 writes
#     `python3 -m agent_sys.cli.main`. Missing a live run reads as "stopped",
#     which is the escalating direction. Found by m3.
#   2 **`show` is not a run and matched.** It type-checks yaml and dispatches
#     nothing, runs in ~1 s, and is run dozens of times a day by owners editing
#     specs. Every one was a window reporting a run that was not one. Found by m3.
#   3 **One run is up to three lines** — `zsh -c` -> `timeout 7200` -> `python3`.
#     A wrapper's command line contains the run's **verbatim**, so *no textual
#     exclusion can separate them*; a regex written specifically to try still
#     kept 5 of 6 shapes. Leaves are therefore selected by **process tree**.
#   4 **Line count has no fixed relation to run count.** Two live runs were once
#     seen as 3 + 1 lines, asymmetric because only one was wrapped. So the count
#     cannot be recovered by dividing. Found by m3.
#
# ## Known limits — read before trusting a result
#
# * `grep -v grep` filters **by content**, so a genuine run whose command line
#   contained the string "grep" would be silently dropped (m2). A false negative
#   on a liveness check is the worse direction. Fixing this properly needs a
#   reading that cannot contain the query — `/proc/<pid>/cwd` — which is not
#   done here.
# * **Fork over-count is unproven, not disproven.** If a run forks children whose
#   cmdline also matches, every child is a leaf and N runs are reported instead
#   of one. Measured once on one run shape: zero matching children. That is one
#   observation, not a guarantee (m3).
# * `etime` is the **leaf's**, which is the run's age. Reading a wrapper line
#   instead gives the launching *shell's* age. This has not been demonstrated on
#   a run started from a long-lived shell, because every run observed so far had
#   shell and run start together (m3).
#
# ## Reporting
#
# Report as evidence, past tense, with the root: *"a process for run-root X was
# present at HH:MM:SS"*. Never "a run is live" — by the time it is read, it may
# not be. Absence of output is **not** proof of absence; see the limits above.
set -u

SNAP=$(mktemp) || exit 1
trap 'rm -f "$SNAP"' EXIT

# One snapshot, written to a real file, then read TWICE. Do not replace this
# with `awk … - -`: the first `-` consumes stdin and the second gets EOF, so
# `NR==FNR` is true for every line, `next` fires every time, and the print block
# **never executes** — measured, GNU Awk 5.2.1: pass1=2 lines, pass2=0. The
# failure is silent and empty, i.e. indistinguishable from "no run present".
ps -eo pid,ppid,etime,args --no-headers 2>/dev/null \
  | grep -E '(-m +agent_sys\.cli\.main|/agent-sys) +run\b' \
  | grep -v grep > "$SNAP"

awk '
  NR==FNR { parent[$2] = 1; next }            # pass 1: every ppid among matches
  !parent[$1] {                                # pass 2: keep leaves only
    root = "?"
    for (i = 1; i <= NF; i++) if ($i == "--demo-root") root = $(i + 1)
    printf "run pid=%s etime=%s root=%s\n", $1, $3, root
    n++
  }
  END { printf "%d run process(es) present at %s\n", n + 0, strftime("%H:%M:%S") }
' "$SNAP" "$SNAP"
