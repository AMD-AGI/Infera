#!/usr/bin/env bash
# Purpose: T2f -- the ungated AgentX concurrency sweep against the ONE live t2f
#   stack (index_share=false AND --disable-custom-all-reduce). Identical to the
#   fixed sweep.yihou.sh, INCLUDING its 20 s late-result guard, EXCEPT it points
#   the bench at config.yihou.full.dcar.sh so the run's config provenance matches
#   the stack that is actually up. NO coherence gate: under simulated acceptance
#   (3.61) garbled text is expected by construction (spec_utils.py:401-434), so a
#   coherence gate would test a property the simulation is designed to violate.
#
# Usage: OUT_PREFIX=t2f ./sweep.yihou.dcar.sh [CONC ...]   (default 40 56 72 96 128)
# Artifacts: <workspace>/<prefix>-agentx-c<N>/ per point, plus per-point .log.
#
# DO NOT "DE-DUPLICATE" THIS AGAINST sweep.yihou.sh. The ONLY difference is the
# CONFIG path: sweep.yihou.sh uses config.yihou.full.sh (custom all-reduce ON,
# the t2e cell); this uses config.yihou.full.dcar.sh (--disable-custom-all-reduce,
# the t2f cell). Merging the two and keeping the wrong CONFIG would make the
# recorded config disagree with the stack that produced the numbers -- the exact
# provenance trap this file exists to avoid. Keep both; they measure different
# cells.
set -uo pipefail

W="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$W/../.." && pwd)"
CONCS=("${@:-}")
[[ -n "${CONCS[0]:-}" ]] || CONCS=(40 56 72 96 128)
SETTLE="${SETTLE:-180}"
PREFIX="${OUT_PREFIX:-t2f}"

echo "sweep start $(date -u +%FT%TZ) prefix=$PREFIX points: ${CONCS[*]}"
for conc in "${CONCS[@]}"; do
    out="$W/$PREFIX-agentx-c$conc"
    echo "=== CONC=$conc start $(date -u +%FT%TZ) -> $out"
    if "$BENCH/agentx_bench.sh" CONC="$conc" \
        CONFIG="$W/config.yihou.full.dcar.sh" \
        TOPOLOGY="$W/topology.yihou.tsv" \
        OUT_DIR="$out" >"$W/$PREFIX-agentx-c$conc.log" 2>&1; then
        echo "=== CONC=$conc PASS $(date -u +%FT%TZ)"
    elif rc=$?; sleep 20; [[ -s "$out/agentx_conc$conc.json" ]]; then
        # agentx_bench.sh:105 tests the result file the instant the client exits,
        # but the container's EXIT trap chowns the tree, so a large artifact set
        # can still be settling. Believe a non-zero exit only if the file is
        # still absent 20 s later. (Same guard as the fixed sweep.yihou.sh.)
        echo "=== CONC=$conc PASS (late result, rc=$rc ignored) $(date -u +%FT%TZ)"
    else
        echo "=== CONC=$conc FAIL rc=$rc $(date -u +%FT%TZ) -- stopping sweep"
        tail -20 "$W/$PREFIX-agentx-c$conc.log" | sed 's/^/    /'
        break
    fi
    echo "--- settling ${SETTLE}s for HiCache release"
    sleep "$SETTLE"
done
echo "sweep end $(date -u +%FT%TZ)"
