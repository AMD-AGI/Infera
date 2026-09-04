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

# ## The argv match is a candidate filter, NOT the decision
#
# **A launcher that outlives its matching child reports as a live run.** Measured
# 2026-09-04 (m3 reasoned it, this file's author constructed it): with
# `sh -c '<run>; sleep 20'`, once `<run>` exits the shell has no matching child,
# so it is a leaf, and its argv still contains the run's command verbatim —
# reported as `run pid=1347340 root=/tmp/ATTACK;`. **A run that had ended, read
# as live.** That is the reassuring direction and the one this tool exists to
# stop being wrong in.
#
# No argv rule can fix it, because the wrapper's argv legitimately *contains* the
# run's. So the decision is made on `/proc/<pid>/exe` — **the kernel's record of
# the actual binary, which a command line cannot fake.** Measured:
#
#     real run (`python -m …`)      exe = …/bin/python3.14   keep
#     console script (`agent-sys`)  exe = …/bin/python3.14   keep   (shebang)
#     `timeout 7200 …` wrapper      exe = /usr/bin/timeout   drop
#     `sh -c '…'` / `zsh -c '…'`    exe = …/sh, …/zsh        drop
#
# The console-script form survives, so m3's finding 1 stays fixed. The leaf rule
# below is kept as a second layer for the fork case, which is unproven either way.
#
# ## `exe` unreadable is a THIRD answer, not a drop
#
# `readlink … || continue` was the first version and it is this file's own
# original defect wearing a new cause (m3): **a matching process whose `exe`
# cannot be read was dropped without a word**, and a run you cannot inspect then
# reads as no run — the escalating direction. Measured: `readlink /proc/1/exe`
# exits 1 (another user's process), and so does a pid that exited between the
# `ps` snapshot and the `readlink` (the race). Neither is reachable while every
# run on this host is `yihou`-owned, so it is latent — and latent in the
# direction that matters.
#
# So there are three outcomes, and the third is **reported**: the count and the
# reason disagree visibly instead of the count quietly being short.
UNK=$(mktemp) || exit 1
trap 'rm -f "$SNAP" "$UNK"' EXIT

# ## Match the INVARIANT, not the entry point
#
# The pattern was `(-m agent_sys\.cli\.main|/agent-sys) +run\b`, which anchors
# `run` to a known entry point. **Measured 2026-09-04 19:23: it reported 0 while
# rung 2e was alive and writing**, because that run is driven by a wrapper:
#
#     python3 .../assets/lib/run_with_long_stall.py --stall-after 3600 run --package …
#
# A third launch shape after `python -m` and the `agent-sys` console script, and
# the failure is the escalating direction — **"no process" reads as "stopped"**,
# which is the state this record escalates on. That is m3's finding 1 again, one
# entry point further out, and the third time this tool has been blind to a way
# of starting a run.
#
# So match what every entry point must pass regardless of how it was invoked:
# **` run ` followed by `--package`** — the CLI's own contract. Verified to match
# the wrapper and to reject `show --package`. The `exe` and leaf filters below
# still remove shells, `timeout`, and parents.
ps -eo pid,ppid,etime,args --no-headers 2>/dev/null \
  | grep -E '[[:space:]]run[[:space:]].*--package' \
  | grep -v grep \
  | while read -r pid rest; do
      if exe=$(readlink "/proc/$pid/exe" 2>/dev/null); then
        case "$exe" in
          */python*) printf '%s %s\n' "$pid" "$rest" ;;
          *) ;;                     # a shell or `timeout` wearing the run's argv
        esac
      else
        printf '%s %s\n' "$pid" "$rest" >> "$UNK"   # matched, undecidable
      fi
    done > "$SNAP"

NOW=$(date +%H:%M:%S)

# One snapshot, written to a real file, then read TWICE. Do not replace this
# with `awk … - -`: the first `-` consumes stdin and the second gets EOF, so
# `NR==FNR` is true for every line, `next` fires every time, and the print block
# **never executes** — measured, GNU Awk 5.2.1: pass1=2 lines, pass2=0. The
# failure is silent and empty, i.e. indistinguishable from "no run present".
#
# `date` rather than awk's `strftime`, which is a GNU extension (m3): it sits in
# the END block, so where it is missing the failure takes the count line — the
# part that gets quoted.
awk -v now="$NOW" '
  NR==FNR { parent[$2] = 1; next }            # pass 1: every ppid among matches
  # Pass 2 keeps leaves. **Since `exe` filtering arrived this rule does ONE job,
  # not two** (m3): wrappers are already gone, so the only thing it can still
  # remove is a genuine python run that is the parent of another matching python
  # run — the fork case, where dropping the parent is right. It is no longer the
  # wrapper defence; do not read it as one. If the fork case is ever shown not to
  # occur, this becomes dead weight.
  !parent[$1] {
    root = "?"
    for (i = 1; i <= NF; i++) {
      if ($i == "--demo-root") root = $(i + 1)          # space-separated form
      else if ($i ~ /^--demo-root=/) { root = $i; sub(/^--demo-root=/, "", root) }
    }                                                    # `=` form (m3): silently
    printf "run pid=%s etime=%s root=%s\n", $1, $3, root # gave root=? before
    n++
  }
  END { printf "%d run process(es) present at %s\n", n + 0, now }
' "$SNAP" "$SNAP"

# The third answer, said out loud. A count that is short because something was
# undecidable must not look like a count that is short because nothing was there.
if [ -s "$UNK" ]; then
  echo "UNDECIDED: matched the run pattern but /proc/<pid>/exe was unreadable —"
  echo "  another user's process, or it exited between the snapshot and the read."
  echo "  These are NOT counted above. The count is a lower bound:"
  while read -r upid urest; do
    echo "    pid=$upid  $(printf '%s' "$urest" | cut -c1-70)"
  done < "$UNK"
fi
