#!/usr/bin/env bash
# The whole verification (README "验证计划", REPRODUCE.zh-CN.md), one step after another on the prefill node:
#   preflight  image present on both nodes; the 11 base files of the image equal patches/BASE_MD5SUMS (the files every
#              patch here replaces); stops otherwise (ALLOW_BASE_MISMATCH=1 to go on anyway: the results then say
#              nothing about this fix on that image)
#   unit       scripts/test_fix.py in the image with the fix mounted
#   B1 N1a A1 B2 A2 B3 C0 C1 C2 N0 N1 Bprod Cprod   deployments (scripts/verify_suite.sh, arms in config/)
#   scoreB scoreC   AgentX C104, 3600 s (scripts/pd_perf_chain.sh), summary in $LOG_ROOT/score-*.txt
# STEPS="..." runs a subset in the given order (default: all). Stops at the first step that fails to run.
# Summary: $LOG_ROOT/verify.out (pass it to scripts/judge.py).
# Usage (prefill node): [STEPS="..."] run_verify.sh >> $LOG_ROOT/verify.out 2>&1
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PKG="${PKG:-$(cd "$here/.." && pwd)}"
C=$PKG/config
set -a
# shellcheck source=/dev/null
. "$C/site.local.env"
set +a
mkdir -p "$LOG_ROOT"
ALL="preflight unit B1 N1a A1 B2 A2 B3 C0 C1 C2 N0 N1 Bprod Cprod scoreB scoreC"
say() { echo "$(date -u +%T) $*"; }
on() { if [[ "$1" == localhost ]]; then bash -c "$2"; else ssh -o BatchMode=yes "$1" "$2"; fi; }
suite() {  # ARM [VAR=VAL ...]
    local arm="$C/arm-$1.env"; shift
    env "$@" bash "$here/verify_suite.sh" "$arm" || { say "$(basename "$arm" .env) failed: stopping"; exit 1; }
}
gate() { suite "$1" PROBE_N=200 "LONGCTX_SPECS=16000:128 16000:128"; }
prod() { suite "$1" PROBE_N=1319 "LONGCTX_SPECS=16000:128 100000:32" EVAL="${PROD_EVAL:-1}"; }  # PROD_EVAL=0: no GitHub / PyPI
score() {  # X (B or C): arm-X-prod.env at C104
    local lab="vfix-$1-c104" out="$LOG_ROOT/score-vfix-$1.txt"
    bash "$here/pd_perf_chain.sh" "$PKG/kit" "$C/arm-$1-prod.env" "$LOG_ROOT/perf/$lab" 104
    bash "$here/score_summary.sh" "$LOG_ROOT/perf/$lab/c104.log" > "$out" 2>&1
    say "score $1: $(grep -E '^\| C104' "$out")"
}
for step in ${STEPS:-$ALL}; do
    case "$step" in
    preflight)
        for n in localhost "$DECODE_SSH"; do
            id="$(on "$n" "docker image inspect -f '{{.Id}}' $IMAGE" 2>/dev/null)"
            [[ -n "$id" ]] || { say "preflight: $IMAGE missing on $n: stopping"; exit 1; }
            sums="$(on "$n" "docker run --rm --network none --entrypoint sh $IMAGE -c 'cd /sgl-workspace/sglang/python/sglang/srt && md5sum $(awk '{print $2}' "$PKG/patches/BASE_MD5SUMS" | tr '\n' ' ')'")"
            bad="$(diff <(sort -k2 <<<"$sums") <(sort -k2 "$PKG/patches/BASE_MD5SUMS") | grep '^[<>]' | awk '{print $3}' | sort -u | tr '\n' ' ')"
            say "preflight $n: image $id; base files differing: ${bad:-none}"
            if [[ -n "$bad" && "${ALLOW_BASE_MISMATCH:-0}" != 1 ]]; then say "preflight: base mismatch on $n: stopping"; exit 1; fi
        done ;;
    unit)
        bash "$here/run_unit.sh" "${PREFILL0_GPUS%%,*}" > "$LOG_ROOT/unit.log" 2>&1; rc=$?
        say "unit: $(tail -1 "$LOG_ROOT/unit.log")"
        [[ $rc == 0 ]] || { say "unit checks failed: stopping"; exit 1; } ;;
    B1 | B2 | B3) gate B-gate ;;
    A1 | A2) gate A-gate ;;
    C1 | C2) gate C-gate ;;
    C0) suite C0-e11 PROBE_N=200 LONGCTX_SPECS= ;;
    N0) suite N0-nospec PROBE_N=200 LONGCTX_SPECS= ;;
    N1) suite N1-nospec-fix PROBE_N=400 "LONGCTX_SPECS=16000:32" ;;
    N1a) suite N1a-nospec-audit PROBE_N=400 "LONGCTX_SPECS=16000:32" ;;
    Bprod) prod B-prod ;;
    Cprod) prod C-prod ;;
    scoreB) score B ;;
    scoreC) score C ;;
    *) say "unknown step $step: stopping"; exit 1 ;;
    esac
done
say "run_verify done"
python3 "$here/judge.py" "$LOG_ROOT/verify.out"
