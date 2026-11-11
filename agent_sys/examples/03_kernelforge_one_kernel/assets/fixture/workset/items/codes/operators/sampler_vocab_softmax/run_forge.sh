#!/bin/sh
# Hyperloom embedded Forge launcher for sampler_vocab_softmax.
# Generated from the base workset; edit its definition/workload, not this file.
# Runtime: the Hyperloom revision selected by this session's AKLR setup, [forge].
# --workspace selects the isolated engine checkout; --dry-run prints a quoted
# command before GPU/agent work. AKLR_REPO references the external knowledge
# checkout. --resume keeps immutable campaign inputs and repeats KB references.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SPEC="$HERE"/invocation_spec.json
DRIVER="$HERE"/scripts/forge_driver.py
KERNEL='srt/layers/sampler.py'
BACKEND='triton'
FORGE_PYTHON="${FORGE_PYTHON:-python3}"

# An undetermined backend refuses before the campaign.
# Upstream would accept an unknown name and substitute flydsl; a campaign
# that completes under a backend nobody chose is worse than one that does not
# start. The generator writes an empty value when the Definition's tags name no
# KernelForge backend, or name more than one.
if [ -z "$BACKEND" ]; then
  echo "run_forge.sh: cannot name a Hyperloom kernel backend for sampler_vocab_softmax" >&2
  echo "  The Definition's tags name no backend forge registers, or name more" >&2
  echo "  than one. Known backends: assembly ck flydsl triton gluon aiter hip hipblaslt fusion" >&2
  echo "  Passing an unknown name is NOT the fix: forge warns and silently" >&2
  echo "  substitutes flydsl, and the campaign completes under it." >&2
  echo "  FIX: correct the Definition's tags; or rebuild with" >&2
  echo "       --var forge_backend=<backend> (legacy alias: forge_fellow)." >&2
  echo "       The tags are the per-operator answer -- one workset can hold a" >&2
  echo "       Triton operator and a HIP one, so a launch-wide value is a" >&2
  echo "       fallback for the undecidable case, not the mechanism." >&2
  exit 2
fi
if [ -z "$KERNEL" ]; then
  echo "run_forge.sh: the workset records no edit_target.source_file for sampler_vocab_softmax," >&2
  echo "  so --kernel cannot be formed. FIX: pass --kernel <file inside --workspace>" >&2
  exit 2
fi

DRY=""
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
RESUME=""
for a in "$@"; do [ "$a" != "--resume" ] || RESUME=1; done
EXPECT_BACKEND=""
for a in "$@"; do
  if [ -n "$RESUME" ]; then
    case "$a" in
      --kernel|--kernel=*|--driver|--driver=*|--kernel-backend|--kernel-backend=*|--program-md-file|--program-md-file=*|--git-branch|--git-branch=*|--source-files|--source-files=*|--invocation-spec-file|--invocation-spec-file=*)
        echo "run_forge.sh: resume keeps the original kernel, driver, backend and brief; omit $a" >&2; exit 2 ;;
    esac
  fi
  if [ -n "$EXPECT_BACKEND" ]; then
    [ "$a" = "$BACKEND" ] || { echo "run_forge.sh: backend override $a conflicts with workset backend $BACKEND" >&2; exit 2; }
    EXPECT_BACKEND=""
  fi
  case "$a" in
    --kernel-backend) EXPECT_BACKEND=1 ;;
    --kernel-backend=*)
      [ "${a#*=}" = "$BACKEND" ] || { echo "run_forge.sh: $a conflicts with workset backend $BACKEND" >&2; exit 2; } ;;
  esac
done
[ -z "$EXPECT_BACKEND" ] || { echo "run_forge.sh: --kernel-backend needs a value" >&2; exit 2; }
if [ -z "$RESUME" ]; then
  set -- --invocation-spec-file "$SPEC" --driver "$DRIVER"          --kernel-backend "$BACKEND" --kernel "$KERNEL" "$@"
fi
# --skill-dir applies only to this invocation: pass it again on resume.
# Reference AKLR's checkout in place; never vendor its knowledge pages.
KB="${AKLR_KB_ROOT:-}"
if [ -z "$KB" ]; then
  AKLR="${AKLR_REPO:-${KFO_AKLR_REPO:-}}"
  [ -z "$AKLR" ] || KB="$AKLR/harness/skills/aklr-kb"
fi
if [ -n "$KB" ]; then
  [ -f "$KB/SKILL.md" ] || { echo "missing AKLR KB: $KB/SKILL.md" >&2; exit 2; }
  set -- "$@" --skill-dir "$KB"
  LANGUAGE="$BACKEND"
  [ "$LANGUAGE" != "gluon" ] || LANGUAGE=triton
  [ ! -f "$KB/$LANGUAGE/SKILL.md" ] || set -- "$@" --skill-dir "$KB/$LANGUAGE"
elif [ -z "$DRY" ]; then
  echo "run_forge.sh: use the session resolver's AKLR_KB_ROOT (KB is referenced, not copied)" >&2
  exit 2
fi

# `--workspace=<dir>` and `--workspace <dir>` both have to count, so this
# matches the flag name rather than a spaced token.
HAS_WS=""
for a in "$@"; do case "$a" in --workspace|--workspace=*) HAS_WS=1 ;; esac; done

if [ -n "$DRY" ]; then
  # **Prints even without `--workspace`, and says so in the command it prints.**
  # Dry-run's job is to make the invocation reviewable; refusing here would
  # break that and STEP 6's acceptance with it. The placeholder is deliberately
  # not a valid path, so the printed line cannot be pasted blind.
  [ -n "$HAS_WS" ] || set -- "$@" --workspace '<REQUIRED: the git checkout to edit>'
  [ -n "$KB" ] || set -- "$@" --skill-dir '<REQUIRED: AKLR_REPO/harness/skills/aklr-kb>'
  "$FORGE_PYTHON" -c 'import shlex, sys; print(shlex.join([sys.executable, "-m", "kernelforge.cli", "forge-loop", *sys.argv[1:]]))' "$@"
  exit 0
fi

if [ -z "$HAS_WS" ]; then
  echo "run_forge.sh: --workspace is required by python -m kernelforge.cli forge-loop" >&2
  echo "  (cli.py:725) and names the git checkout it edits in place. This" >&2
  echo "  wrapper cannot know it: the workset describes an operator, not a" >&2
  echo "  checkout." >&2
  echo "  FIX: $0 --workspace <checkout> [other forge-loop flags]" >&2
  exit 2
fi

# **Is the recorded kernel path in the checkout's frame?** Checked here so the
# answer is a message naming both frames rather than forge's
# `kernel must be inside workspace` / `is not a file` from inside a campaign
# that has already booked the node. Only checked when the caller did not
# override `--kernel`, and only when the workspace is a plain `--workspace
# <dir>` pair -- the `=` form is left to forge, because splitting it here would
# be a second parser to keep in step with click's.
WS=""
prev=""
for a in "$@"; do
  [ "$prev" = "--workspace" ] && WS="$a"
  prev="$a"
done
if [ -z "$RESUME" ] && [ -n "$WS" ] && [ ! -f "$WS/$KERNEL" ] && [ ! -f "$KERNEL" ]; then
  echo "run_forge.sh: --kernel does not resolve inside the workspace" >&2
  echo "  workspace: $WS" >&2
  echo "  kernel:    $KERNEL   (edit_target.source_file, re-framed onto its repo_root_var)" >&2
  echo "  tried:     $WS/$KERNEL" >&2
  # A suffix that IS present is reported as a suggestion and never substituted:
  # the frames differ by a prefix, and picking one automatically is how the
  # wrong file gets edited without anybody choosing it.
  suffix="$KERNEL"
  while [ "${suffix#*/}" != "$suffix" ]; do
    suffix="${suffix#*/}"
    if [ -f "$WS/$suffix" ]; then
      echo "  present:   $WS/$suffix" >&2
      echo "             -- a candidate, NOT applied. Pass it explicitly if it is the one." >&2
      break
    fi
  done
  echo "  FIX: $0 --workspace $WS --kernel <path inside the workspace>" >&2
  exit 2
fi
exec "$FORGE_PYTHON" -m kernelforge.cli forge-loop "$@"
