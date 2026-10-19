#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
# One-shot runner for the infera test suite; `tests/run_tests.sh -h` for usage.
# Tier logic lives in tests/lib/.

set -uo pipefail

SUITE="${1:-}"
SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
REPO="$(dirname "$(dirname "$SCRIPT")")"

. "$REPO/tests/lib/env.sh"
. "$REPO/tests/lib/slurm.sh"
. "$REPO/tests/lib/disagg.sh"
. "$REPO/tests/lib/tiers.sh"

usage() {
  cat <<EOF
usage: $0 unit | engine | all | e2e [sglang|vllm|atom|all] [mixed|disag]

  unit     pure-Python logic suite
  engine   vllm/sglang engine suites (GPU)
  e2e      PD-mixed and/or PD-disaggregated suites (GPU; default: all engines, both)
  all      unit + engine + e2e

GPU tiers run in place on a host with docker + >=8 AMD GPUs, else on one SLURM
node; PD-disagg holds two. Common knobs:
  INFERA_E2E_SITE                     cluster profile, tests/sites/<name>.env
  INFERA_E2E_MODEL_DIR                pre-staged models, mounted read-only
  INFERA_E2E_SLURM_PARTITION          partition (default: the cluster's default)
  INFERA_E2E_SLURM_ACCOUNT_QOS_PAIRS  account:qos,... ladder, tried in order
  INFERA_E2E_GFX_ARCH                 gfx950 (default) | gfx942
  INFERA_E2E_K                        pytest -k over the e2e case ids
EOF
  exit "${1:-2}"
}

report() {
  local rc="$1"
  if [ "$rc" -ne 0 ]; then
    echo ""
    echo "===================== FAILED TEST SUMMARY ====================="
    if [ -s "$SCRATCH/failures.txt" ]; then
      sort -u "$SCRATCH/failures.txt" | sed 's/^/  /'
    else
      echo "  (a tier failed but no per-test detail was captured — likely an image"
      echo "   build error or a native crash before pytest ran; scan above.)"
    fi
    echo "==============================================================="
  fi
  log_dir_banner
  ls -1 "$E2E_LOG_DIR"/*.log 2>/dev/null | sed 's|^|  |'
  if [ "$rc" -ne 0 ]; then
    echo "RESULT: FAIL"
  elif [ -n "$_SKIPPED_TIERS" ]; then
    echo "RESULT: PASS (SKIPPED: $_SKIPPED_TIERS)"
  else
    echo "RESULT: PASS"
  fi
}

main() {
  case "$SUITE" in
    unit | engine | e2e | all) ;;
    -h | --help) usage 0 ;;
    *) usage ;;
  esac
  load_site_profile
  resolve_arch
  init_job_names
  init_scratch
  init_log_dir
  trap '_release_hold; _cleanup_scratch' EXIT
  trap '_wipe_disag_nodes; _release_hold; _cancel_dispatched; exit 130' INT TERM
  log_dir_banner
  init_e2e_flags
  init_slurm_config
  prepare_node

  local rc=0
  case "$SUITE" in
    unit) unit_tier || rc=1 ;;
    engine) engine_tier || rc=1 ;;
    e2e) run_e2e "${2:-}" "${3:-}" || rc=1 ;;
    all)
      unit_tier || rc=1
      engine_tier || rc=1
      run_e2e || rc=1 ;;
  esac
  report "$rc"
  exit "$rc"
}

main "$@"
