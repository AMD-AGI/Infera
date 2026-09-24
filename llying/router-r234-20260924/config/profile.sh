#!/usr/bin/env bash
# Source this file with a profile. It sets Router-only controls, starts nothing.
case "${1:-legacy}" in
  legacy|r1|r2-shadow|r2|r3-shadow|r3-gpu|r3-host|r4-shadow|r4|combined) ;;
  *) printf 'Unknown Router profile: %s\n' "$1" >&2; return 2 2>/dev/null || exit 2 ;;
esac
export INFERA_PD_PREFILL_GUARD_RELEASE=decode
export INFERA_R2_DECODE_DEMAND=off
export INFERA_R3_CACHE_TIERS=off
export INFERA_R3_HOST_WEIGHT=0
export INFERA_R4_PREFILL_WORK=off

case "${1:-legacy}" in
  legacy) ;;
  r1) export INFERA_PD_PREFILL_GUARD_RELEASE=completion ;;
  r2-shadow|r2)
    export INFERA_PD_PREFILL_GUARD_RELEASE=completion
    export INFERA_R2_DECODE_DEMAND=shadow
    [[ "$1" == r2 ]] && export INFERA_R2_DECODE_DEMAND=on
    ;;
  r3-shadow|r3-gpu|r3-host)
    export INFERA_PD_PREFILL_GUARD_RELEASE=completion
    export INFERA_R3_CACHE_TIERS=on
    [[ "$1" == r3-shadow ]] && export INFERA_R3_CACHE_TIERS=shadow
    [[ "$1" != r3-gpu ]] && export INFERA_R3_HOST_WEIGHT=0.5
    ;;
  r4-shadow|r4)
    export INFERA_PD_PREFILL_GUARD_RELEASE=completion
    export INFERA_R4_PREFILL_WORK=shadow
    [[ "$1" == r4 ]] && export INFERA_R4_PREFILL_WORK=on
    ;;
  combined)
    export INFERA_PD_PREFILL_GUARD_RELEASE=completion
    export INFERA_R2_DECODE_DEMAND=on
    export INFERA_R3_CACHE_TIERS=on
    export INFERA_R3_HOST_WEIGHT=0.5
    export INFERA_R4_PREFILL_WORK=on
    ;;
esac
# Do not propagate the status of the last conditional into a caller using -e.
true
