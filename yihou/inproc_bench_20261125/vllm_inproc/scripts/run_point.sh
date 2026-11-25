#!/bin/bash
# Run one vllm bench point in the kit container with provenance. Refuses to overwrite an existing point.
# Usage: run_point.sh NAME TOOL [tool args...]
#   TOOL: engine_step_bench | forward_bench | infer_llm
# Env: CONTAINER (default yihou-inproc-vllm), WORK_DIR (host side of /out, for the printout),
#      ARGS_FILES ("presets/x.args ..." prepended to the tool args), MODEL (default /models/DeepSeek-V4.1-Flash,
#      passed as --model unless the args already carry it), TORCHRUN_NPROC (launch with torchrun), DRY_RUN=1.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../common/lib.sh"
[ $# -ge 2 ] || kit_die "usage: run_point.sh NAME TOOL [args...]"
NAME=$1 TOOL=$2; shift 2
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || kit_die "NAME must match [A-Za-z0-9._-]+"
CONTAINER=${CONTAINER:-yihou-inproc-vllm}
PRE=()
if [ -n "${ARGS_FILES:-}" ]; then
  read -r -a AF <<< "$ARGS_FILES"
  mapfile -t PRE < <(kit_expand_args "${AF[@]}")
fi
MODEL_ARGS=()
case " $* ${PRE[*]:-} " in *" --model "*) ;; *) MODEL_ARGS=(--model "${MODEL:-/models/DeepSeek-V4.1-Flash}");; esac
# Subcommand tools (infer_*) take engine flags before the mode word: keep the user's order, prepend presets.
kit_run_point vllm "$CONTAINER" "$NAME" "$TOOL" "${MODEL_ARGS[@]}" "${PRE[@]}" "$@"
