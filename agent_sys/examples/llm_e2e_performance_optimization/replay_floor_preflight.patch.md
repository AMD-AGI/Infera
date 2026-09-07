# UNAPPLIED PATCH — a preflight that computes the replay-length floor and aborts before bring-up

**Written 2026-09-06 by m35. NOT APPLIED, and the reason is the point.**

## Why this is not applied

`agent-sys run` **stages the package from the working tree per task**, so an edit
to `assets/load/replay.sh` reaches **every task that starts after the edit** —
including m2 tasks in chain 6, which is live and has not reached them.

**This patch adds a new abort path.** Putting one in front of a live run at the
end of a hold is the guard-cascade shape recorded in `.claude/CLAUDE.md`
(*"read this before adding a guard"*): today's last two run deaths were both
caused by a guard added to fix the previous death, and each was correct about the
failure in front of it.

**It lands when no chain is running**, after a `--var trace_end_ms=120000` run has
shown `CAPTURE_OK` in `capture_stacks.log` — i.e. after the thing it guards has
been demonstrated to work, not before.

## Which way it falls when it cannot decide

Per the test in `CLAUDE.md`: **a guard's worth is which way it falls when it
cannot decide.**

**This one falls toward PERMIT.** If any of the four variables is unset, empty or
non-numeric, it prints what it could not read and **continues**. It aborts only
when it has all four numbers *and* the arithmetic says the load is too short —
i.e. only on a positive determination.

**Why permit is the cheaper side here:** falling toward abort would block a
legitimate configuration nobody anticipated (a site that shortens `warmup_s`, or
a future capture sequence with different steps), and the cost of permitting is
exactly today's failure — visible, diagnosable in `capture_stacks.log`, and
already producing two refusals that name it. **The miss is loud; a wrong abort
would be a run death that looks like the problem it prevents.**

## The patch

Insert into `assets/load/replay.sh`, immediately **before** the
`if [ "$CAPTURE" = "1" ]; then` block that cuts the measurement window
(currently line ~99):

```sh
# --- replay-length floor (m35, 2026-09-06) ----------------------------------
# The two captures below run IN SEQUENCE INSIDE ONE LOAD, and the second one
# requires the load to still be running (`capture.sh:130`, `load_running`).
# So the replay must outlast warmup + window + stack_window.
#
# Measured on 20260906T113811-fdb0bd with --var trace_end_ms=60000: the load
# ended 12:14:58, the measurement window ran to 12:15:19, the stack capture
# started after that and aborted with "no aiperf load in flight". Two validators
# then refused, merge_profiling_evidence never ran, and m3/m4/m5 never started.
#
# **Falls toward PERMIT.** Any value it cannot read -> say so and continue. It
# aborts only on a positive determination, because a wrong abort here is a run
# death and a miss is a loud failure two steps later.
if [ "$CAPTURE" = "1" ]; then
  _floor_ok=1
  for _v in E2E_TRACE_END_MS E2E_WARMUP_S E2E_WINDOW_S E2E_STACK_WINDOW_S; do
    eval "_val=\${$_v:-}"
    case "$_val" in
      ''|*[!0-9]*) say "replay floor: cannot read $_v (='$_val') — SKIPPING the check, not failing it"
                   _floor_ok=0 ;;
    esac
  done
  if [ "$_floor_ok" = "1" ]; then
    _need_s=$(( E2E_WARMUP_S + E2E_WINDOW_S + E2E_STACK_WINDOW_S ))
    _have_s=$(( E2E_TRACE_END_MS / 1000 ))
    if [ "$E2E_STACK_WINDOW_S" -gt 0 ] && [ "$_have_s" -le "$_need_s" ]; then
      say "ABORT: the replay is shorter than the two captures it must cover."
      say "  trace_end_ms  = ${E2E_TRACE_END_MS} ms  = ${_have_s} s   (the load)"
      say "  warmup_s      = ${E2E_WARMUP_S} s"
      say "  window_s      = ${E2E_WINDOW_S} s"
      say "  stack_window_s= ${E2E_STACK_WINDOW_S} s"
      say "  floor         = ${_need_s} s, and the load must EXCEED it plus setup"
      say ""
      say "  The stack capture would start after the load had finished and abort"
      say "  with 'no aiperf load in flight'; check_trace_coverage and"
      say "  check_kernel_table would then both refuse, and the refusal would look"
      say "  like a producer defect. See ON-ARM-REFUSAL.md section 8."
      say ""
      say "  Either raise trace_end_ms above ${_need_s}000 (the package's own"
      say "  defaults are 120000 and 180000, both clear), or set"
      say "  --var stack_window_s=0 to declare the round is taken without it —"
      say "  which costs identify resolution level 1 and pushes m3 onto Magpie."
      exit 1
    fi
    say "replay floor ok: load ${_have_s}s > warmup+window+stack ${_need_s}s"
  fi
fi
# ---------------------------------------------------------------------------
```

## What it does NOT do, stated so nobody over-reads it

- **It does not measure the load's real duration.** It compares the *requested*
  `trace_end_ms` against the *requested* window sum. If AIPerf finishes early for
  its own reasons — a short trace, an error, a `--fixed-schedule` that runs out
  of records — the load can still be gone when the stack capture starts and this
  check will have passed. **It catches the configuration error, not every way the
  load can end early.**
- **It does not account for setup time** between the two captures, which is real
  (~31 s of slack was needed on the measured run). It uses `<=` rather than `<`,
  which buys one second, not thirty. **A configuration that only just clears the
  floor may still fail**, and it will fail the same visible way.
- **It is untested.** I have not run it. `bash -n` is the only thing it has
  passed, and `bash -n` cannot catch an unset-variable or arithmetic error — the
  same limit recorded for `set -o pipefail` under `dash`.

## How to verify it before landing it

1. `bash -n assets/load/replay.sh`
2. Break/restore, both directions, with the surrounding script stubbed:
   `E2E_TRACE_END_MS=60000 E2E_WARMUP_S=60 E2E_WINDOW_S=10 E2E_STACK_WINDOW_S=3`
   → must abort with the arithmetic; `E2E_TRACE_END_MS=120000` → must print
   `replay floor ok`; `E2E_STACK_WINDOW_S=0` → must **not** abort at 60000, since
   no stack capture is scheduled; `E2E_WARMUP_S=` (empty) → must **permit** and
   say it skipped.
3. Then a real run at `trace_end_ms=120000` and `grep CAPTURE_OK
   .../capture_stacks.log`.

**Step 3 is the one that matters.** The rest only shows the guard is
self-consistent; only a `CAPTURE_OK` shows the floor was the right floor.
