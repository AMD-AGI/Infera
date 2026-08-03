#!/bin/bash
# THE CONTROL ARM: Case A with MTP OFF, everything else identical to arm A.
#
# This is the measurement that turns "MTP is the leading suspect" into a result.
# It was specified but NOT run before this kit was written.
#
# SINGLE VARIABLE. Same image, same rate (0.15), same ctx (262144), same
# kvd/kvaware wiring, and -- critically -- the same --disable-custom-all-reduce.
# That last one is why CUSTOM_AR is pinned below rather than left to default:
# it used to FOLLOW MTP, which would silently re-enable the aiter kernel that
# deadlocks on gfx950 and make this a two-variable comparison. See REPRODUCE.md §7.
#
# Usage: run_armB_mtpoff.sh
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
: "${PJOB:?export PJOB=<prefill job>}"; : "${PIP:?export PIP=<prefill ip>}"
: "${DJOB:?export DJOB=<decode job>}";  : "${DIP:?export DIP=<decode ip>}"

echo "===== 1. boot BOTH legs (never just one -- it orphans the survivor) ====="
bash "$D/boot.sh" prefill 262144 1 0 armB     # kvd=1 mtp=0
bash "$D/boot.sh" decode  262144 0 0 armB     # kvd=0 mtp=0   <-- the ONLY delta vs arm A
bash "$D/wait_ready.sh" 1800
bash "$D/router.sh" 8190

echo "===== 2. assert the single variable actually held ====="
L=/shared_nfs/yihou_agbench_mtp/logs/armB_decode.log
S=$(mktemp); strings "$L" > "$S"
SPEC=$(grep -oE "speculative_algorithm=[^,]*" "$S" | head -1)
CAR=$(grep -oE "disable_custom_all_reduce=[A-Za-z]+" "$S" | head -1)
echo "  $SPEC     (want: speculative_algorithm=None)"
echo "  $CAR      (want: disable_custom_all_reduce=True)"
rm -f "$S"
case "$SPEC" in *None*) ;; *) echo "ABORT: MTP is not off"; exit 1;; esac
case "$CAR"  in *True*) ;; *) echo "ABORT: custom all-reduce re-enabled -> two-variable comparison"; exit 1;; esac

echo "===== 3. the identical workload, unchanged ====="
bash "$D/run_bench.sh" full caseA_armB_mtpoff

echo "===== 4. read the outcome ====="
echo "  completes 4000s      -> the hang IS MTP-attributable (and you have Case A data)"
echo "  hangs the same way   -> MTP is NOT the variable; suspect mooncake send_metadata"
echo "  hangs differently    -> record both signatures; the shared factor is the lead"
