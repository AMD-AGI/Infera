#!/usr/bin/env bash
# The proposed guard for `assets/serve/reset_gpus.sh`, standalone so it can be
# tested without touching the package tree.
#
# **NOT APPLIED.** `reset_gpus.sh` is staged per task, so editing it while a
# chain is running delivers an untested change to that run's later stages.
#
# ---------------------------------------------------------------------------
# What it does
#
# `reset_gpus.sh` kills every KFD-holding process whose command name looks like
# an inference engine, node-wide, with a `sudo` fallback that reaches root
# processes inside other containers. This adds one question before the kill:
# **does this pid belong to a container we created?**
#
# Ownership is the label `mix_up.sh` writes (`infera_e2e_run`, and `infera_e2e_arm`
# on an arm). **Never the name and never adjacency** — `yihou_` is a shared
# prefix and four ownership errors were made from names on 2026-09-04/05.
#
# ---------------------------------------------------------------------------
# It fails toward SPARING, and every failure path is named
#
# `/proc` unreadable, no cgroup line, not in a container, `docker inspect`
# failing or empty — all return `unknown:<why>` and the caller spares. A guard
# that kills on uncertainty is worse than no guard, because it looks like
# protection.
#
# **Why sparing is safe here, measured rather than assumed** (2026-09-06):
# `reset_gpus.sh` already spares any process whose command name it does not
# recognise and lets the **VRAM floor** decide pass/fail — its own comment says
# *"an unrecognised process holding VRAM is a reason to stop and look, not a
# reason to shoot"*. A spared foreign process therefore lands in the branch the
# script already has: the VRAM gate times out after `TIMEOUT` and `mix_up.sh`
# aborts **loudly, before any bring-up**. The guard adds no new failure mode; it
# routes one more case into an existing one.
#
# **And our own leftovers stay killable**, which was the fear. Measured: at the
# point `reset_gpus.sh` runs, `mix_up.sh:69` has already `docker rm -f`'d this
# arm's containers, and a removed container takes its processes with it (probed
# 2026-09-06: `/proc/<pid>` was gone one second after `docker rm -f`). Anything
# still holding VRAM is therefore in a container that **still exists** — a
# previous run's, another line's — and a container that exists still carries its
# label. `unknown:` is the empty case, not our case.

#: `pid` -> `ours:<name>` | `foreign:<name>` | `unknown:<why>`.
#:
#: Reads the host's own `/proc`, which sees every container's pids when this
#: runs on the node. It would NOT be reliable from inside a pid namespace — if
#: this script is ever invoked through a transport that gives it a private
#: `/proc`, every lookup returns `unknown:` and the guard spares everything,
#: which is the correct direction for that failure too.
owner_of() {
  pid="$1"
  # **Bash builtins only, up to the one `docker` call — and the reason is
  # diagnostic accuracy, not safety.**
  #
  # I first wrote this rewrite because control C7 reported the `cat`/`sed`/`head`
  # version returning an EMPTY verdict under a broken `PATH`. **That diagnosis
  # was wrong: C7 itself was broken** — `PATH=/nonexistent bash -c …` means the
  # shell cannot find `bash`, so nothing ran at all, and a `2>/dev/null` in the
  # test hid the `command not found`. Measured afterwards with an absolute
  # `/bin/bash` and `PATH` broken *inside*, **both versions return `unknown:`**
  # and both therefore spare. The guard was never unsafe.
  #
  # The rewrite is kept for what the corrected control does show:
  #
  #     original (cat/sed/head)  ->  unknown:empty-cgroup     <- MISLEADING; the
  #                                  cgroup is fine, `cat` was missing
  #     builtins                 ->  unknown:inspect-failed   <- accurate
  #
  # Same safe direction, honest reason versus a reason that would send the next
  # reader to look at `/proc`. `read` and `[[ =~ ]]` cannot be missing.
  [ -r "/proc/$pid/cgroup" ] || { printf 'unknown:no-proc-entry'; return 0; }
  cg=""
  while IFS= read -r line; do cg="$cg$line"$'\n'; done < "/proc/$pid/cgroup" 2>/dev/null
  [ -n "$cg" ] || { printf 'unknown:empty-cgroup'; return 0; }
  if [[ "$cg" =~ docker[-/]([0-9a-f]{12,64}) ]]; then
    cid="${BASH_REMATCH[1]}"
  else
    printf 'unknown:not-in-a-container'; return 0
  fi
  name=$(docker inspect -f '{{.Name}}' "$cid" 2>/dev/null) || {
    printf 'unknown:inspect-failed'; return 0; }
  [ -n "$name" ] || { printf 'unknown:inspect-empty'; return 0; }
  lbl=$(docker inspect -f '{{index .Config.Labels "infera_e2e_run"}}' "$cid" 2>/dev/null)
  arm=$(docker inspect -f '{{index .Config.Labels "infera_e2e_arm"}}' "$cid" 2>/dev/null)
  case "${lbl}|${arm}" in
    # `index` on an absent label renders the literal `<no value>`; treat that
    # and the empty string alike. Both mean "no ownership claim".
    "|"|"<no value>|<no value>"|"|<no value>"|"<no value>|")
        printf 'foreign:%s' "${name#/}" ;;
    *)  printf 'ours:%s' "${name#/}" ;;
  esac
}

#: The two lines to insert into `reset_gpus.sh`'s loop, immediately after the
#: `KILLABLE_RE` test and immediately before `echo "[reset] killing …"`:
#:
#:     own="$(owner_of "$pid")"
#:     case "$own" in
#:       ours:*) ;;
#:       *) echo "[reset] SPARED, not ours ($own): $pid ($comm)"
#:          skipped=$((skipped + 1)); continue ;;
#:     esac
#:
#: `skipped` is the script's existing counter, so a spared process is reported
#: in the line it already prints at the end.
