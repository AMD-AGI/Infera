#!/usr/bin/env bash
# One-off (10-27 10:25): patch v1 (receive side only) did not fix PP comm overlap. Once reorder2_1027.sh has
# taken clean-k5 + async + v1 down, stop it (process group 511263), clean up, run patch v2 (send and receive
# side) for PP comm overlap, and for async too unless v1 already fixed it; then the remaining queue.
set -uo pipefail
L=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/llr/bisect
P=/home/liyingli/bench_agentx/baseline/Infera/llying/prefill-opt-20261027
C=$P/config
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
G=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/2p1d/work/runs/correct-v2/gsm8k/samples_gsm8k_2026-10-20T16-13-32.558577.jsonl
W=$(ls "$L"/waiter9-*.out)
until grep -aq "site-bis5-clean-k5-async-fix-.* down" "$W"; do sleep 10; done
kill -- -511263 2>/dev/null
sleep 5
echo "$(date -u +%T) reorder2_1027 stopped"
(set -a; . "$C/site-bis6-clean-k3-ppcomm-fix2.env"; set +a; cd "$KIT" && bash engine/down.sh)
cd "$KIT" || exit 1
needle=$(grep -a "site-bis5-clean-k5-async-fix-.* longctx" "$W" | grep -o 'needle: n=16 correct=[0-9]*' | grep -o '[0-9]*$')
sites=("$C/site-bis6-clean-k3-ppcomm-fix2.env")
if [[ "${needle:-0}" -lt 14 ]]; then sites+=("$C/site-bis6-clean-k5-async-fix2.env"); fi
echo "$(date -u +%T) async + v1 needle=${needle:-?}/16; running ${#sites[@]} v2 site(s)"
LONGCTX_SPECS=16000:32 bash "$P/scripts/quick_correct.sh" "$KIT" "$L" "${sites[@]}"
PROBE_N=1319 PROBE_PROMPTS=$G LONGCTX_SPECS=16000:32 bash "$P/scripts/quick_correct.sh" "$KIT" "$L" \
    "$C/site-bis4-clean-k3-car.env" "$C/site-bis-clean-k3.env"
PROBE_N=64 LONGCTX_CONC=8 LONGCTX_SPECS="2000:16 4000:16 8000:16" bash "$P/scripts/quick_correct.sh" "$KIT" "$L" \
    "$C/site-bis3-clean-k5-async.env"
