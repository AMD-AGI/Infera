#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
# In-container pytest; failures go to /scratch/failures.txt under <tag>.
# Usage: container_pytest.sh [--per-file] <tag> <pytest args... | test dir>
set -uo pipefail

FAILS=/scratch/failures.txt
OUT=/scratch/.pytest.out
PYTEST=(python3 -m pytest -p no:cacheprovider -o addopts= -rfE)
[ -n "${INFERA_E2E_K:-}" ] && PYTEST+=(-k "$INFERA_E2E_K")

pip install -q pytest pytest-asyncio nats-py 2>/dev/null || true

run_once() {
  local tag="$1" rc
  shift
  "${PYTEST[@]}" "$@" 2>&1 | stdbuf -oL tee "$OUT"
  rc=${PIPESTATUS[0]}
  grep -aE '^(FAILED|ERROR) ' "$OUT" 2>/dev/null | sed "s|^|$tag |" >> "$FAILS"
  return "$rc"
}

# One process per file, so a ROCm/HIP native crash costs one file, not the run.
per_file() {
  local tag="$1" scope="$2" files f code out line fails rc=0
  # A scope with no tests must fail, or a rename would pass having tested nothing.
  if ! files=$(find "$scope" -name "test_*.py") || [ -z "$files" ]; then
    echo "$tag FATAL: scope unreadable or holds no test_*.py" >&2
    echo "$tag scope unreadable or holds no test_*.py" >> "$FAILS"
    return 1
  fi
  for f in $(printf '%s\n' "$files" | sort); do
    echo "----- pytest $f -----"
    "${PYTEST[@]}" -q "$f" 2>&1 | stdbuf -oL tee "$OUT"
    code=${PIPESTATUS[0]}
    out=$(cat "$OUT")
    case $code in
      0)
        line=$(printf '%s' "$out" | grep -E "passed|failed|skipped|no tests ran" | tail -1) ;;
      134 | 137 | 139)
        line="CRASH(exit=$code)"
        echo "$tag CRASH(exit=$code) $f" >> "$FAILS" ;;
      5)
        # Guarded files importorskip a module their image ships: nothing collected = broken image.
        line="FAIL: no tests collected (exit=5)"
        echo "$tag $f (exit=5, no tests collected)" >> "$FAILS" ;;
      *)
        line=$(printf '%s' "$out" | grep -E "passed|failed|error|skipped" | tail -1)
        line="${line:-(exit=$code)}"
        fails=$(printf '%s\n' "$out" | grep -aE "^(FAILED|ERROR) ")
        if [ -n "$fails" ]; then
          printf '%s\n' "$fails" | sed "s|^|$tag |" >> "$FAILS"
        else
          echo "$tag $f (exit=$code)" >> "$FAILS"
        fi ;;
    esac
    [ "$code" -eq 0 ] || rc=1
    printf "  %-56s %s\n" "$f" "$line"
  done
  return "$rc"
}

if [ "${1:-}" = "--per-file" ]; then
  shift
  per_file "$@"
else
  run_once "$@"
fi
