#!/usr/bin/env bash
# One-off (10-27 10:00): reorder_1027.sh skipped clean-k3 + PP comm overlap (up.sh rejected the then-unset
# PREFILL_MOUNT_EXTRA). Stop it (process group 453025), clean up, and run the same queue again in order.
set -uo pipefail
L=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/llr/bisect
P=/home/liyingli/bench_agentx/baseline/Infera/llying/prefill-opt-20261027
C=$P/config
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
G=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/2p1d/work/runs/correct-v2/gsm8k/samples_gsm8k_2026-10-20T16-13-32.558577.jsonl
kill -- -453025 2>/dev/null
sleep 5
echo "$(date -u +%T) reorder_1027 stopped"
(set -a; . "$C/site-bis5-clean-k3-ppcomm-fix.env"; set +a; cd "$KIT" && bash engine/down.sh)
cd "$KIT" || exit 1
LONGCTX_SPECS=16000:32 bash "$P/scripts/quick_correct.sh" "$KIT" "$L" \
    "$C/site-bis4-clean-k3-ppcomm.env" "$C/site-bis5-clean-k3-ppcomm-fix.env" "$C/site-bis5-clean-k5-async-fix.env"
PROBE_N=1319 PROBE_PROMPTS=$G LONGCTX_SPECS=16000:32 bash "$P/scripts/quick_correct.sh" "$KIT" "$L" \
    "$C/site-bis4-clean-k3-car.env" "$C/site-bis-clean-k3.env"
PROBE_N=64 LONGCTX_CONC=8 LONGCTX_SPECS="2000:16 4000:16 8000:16" bash "$P/scripts/quick_correct.sh" "$KIT" "$L" \
    "$C/site-bis3-clean-k5-async.env"
