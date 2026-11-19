#!/usr/bin/env bash
# P4D4 end-to-end with the PR code instead of the main line's overlays, one 3600 s AgentX C104 round (single run):
#   prefill  U26 branch files (exact-size pinned CPU tier, 1.9 TB) + GEM 0.88 + renderer 4   (as m10 / m14)
#   decode   U12 + U13 branch files (gluon logits, AITER AG/RS in FULL capture) + DP engine LB (INFERA_VLLM_DP_ENGINE_LB=1)
#            + util 0.95 + renderer 4                                                          (as m14)
#   router   Rust, serial vLLM Mooncake legs (U16), with the U24 pool idle timeout (bin/infera-router-vllmserial-u24,
#            the same patch as pr/router-pool-idle), and NO keep-alive workaround on the engines (uvicorn's 5 s)
# Compare: m10 48,572 (3600 s, no LB, keep-alive 120, overlays), m14 49,183 (1800 s, LB), f1 48,173 (LB, util 0.93).
# Validates: U24 (router 5xx / "error sending request" without the workaround), DP LB at 3600 s, U12 / U13 / U26 PR code.
# key: for b in u26-rocm-exact-pin u12-gfx950-gluon-mqa-logits u13-aiter-agrs-full-capture; do git -C /home/liyingli/bench_agentx/upstream-20261119/wt/$b rev-parse HEAD; done
# key: sha256sum /home/liyingli/bench_agentx/upstream-20261119/p4d4/scripts/p4d4_run.sh /home/liyingli/bench_agentx/upstream-20261119/p4d4/bin/infera-router-vllmserial-u24
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
# The PR code must have passed its unit tests (J1) first; a P4D4 round is too expensive to spend on a known failure.
for s in v034-u26b v034-u12 v034-u13b; do
    [[ "$(cat "$U/results/$s/rc" 2>/dev/null)" == 0 ]] || { echo "unit step $s has not passed; not starting the round"; exit 3; }
done
need_ready p4d4 || exit 1
export R=$U/p4d4
cd "$R" || exit 1
tag=${P4D4_TAG:-pr-u12u13u26-dplb-u24-nokeepalive-3600}
name=${tag%%-*}-$JOB
mounts() {  # src:dst:ro for every vllm/ file a v0.34.0 branch adds or changes
    local w f
    w=$(wt "$1")
    git -C "$w" diff --name-only --diff-filter=AM v0.34.0 HEAD -- vllm/ | while read -r f; do printf '%s:%s:ro ' "$w/$f" "$SP/$f"; done
}
P_MOUNT_X=$(mounts u26-rocm-exact-pin)
D_MOUNT_X="$(mounts u12-gfx950-gluon-mqa-logits)$(mounts u13-aiter-agrs-full-capture)"
echo "prefill mounts: $P_MOUNT_X"; echo "decode mounts: $D_MOUNT_X"
( bash scripts/memguard.sh "p4d4-$tag" 120 ) &
D_MODE=stock P_GEM=1 P_UTIL=0.88 D_UTIL=0.95 RENDER_TIMEOUT=600 P_RENDER=4 D_RENDER=4 DURATION=3600 USAGE_LOG=1 \
    CPU_BYTES=1900000000000 ROUTER_BIN=infera-router-vllmserial-u24 KEEPALIVE_ENV="" \
    D_ENV_X="INFERA_VLLM_DP_ENGINE_LB=1 VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1" \
    P_MOUNT_X="$P_MOUNT_X" D_MOUNT_X="$D_MOUNT_X" bash scripts/snap.sh p4d4_run.sh "$tag"
echo $? > "logs/once-$JOB.rc"
bash scripts/finish_round.sh "logs/once-$JOB.rc" "p4d4-$tag" "$name"
d=runs/p4d4-$tag
O=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/upstream-vllm-20261117/results/$name
{
    echo "router 'error sending request': $(grep -a -c 'error sending request' "$d/vpd-router.stream.log" 2>/dev/null)"
    echo "router 'unreachable': $(grep -a -c 'unreachable' "$d/vpd-router.stream.log" 2>/dev/null)"
    echo "router 5xx lines: $(grep -a -c -E ' 50[0-9] |status=50[0-9]' "$d/vpd-router.stream.log" 2>/dev/null)"
    echo "decode 'gfx950 gluon': $(grep -a -c 'gluon paged-MQA logits kernel' "$d/vpd-decode.stream.log" 2>/dev/null)"
    echo "decode 'AITER custom AG/RS': $(grep -a -c 'Using AITER custom AG/RS' "$d/vpd-decode.stream.log" 2>/dev/null)"
    echo "prefill exact pinned: $(grep -a 'at their exact size in pinned host memory' "$d/vpd-prefill.stream.log" 2>/dev/null | head -4)"
    grep -a "p4d4-$tag" logs/summary.out | tail -2
} | tee "$OUT/pr-checks.txt"
mkdir -p "$O" && cp "$OUT/pr-checks.txt" "$O/"
grep -q "score p4d4-$tag .* rc=0" logs/summary.out
