#!/usr/bin/env bash
# #48047 RDMA A/B on the user's holder (11-12): run inside it with
#   srun --jobid=<holder> --overlap -N1 -n1 -w <node> bash e1ab_holder_chain.sh <MIN_EVICT>...
# Builds e1ab-sglang:1112 (nightly + host libionic entrypoint) if missing, then one config-B gsp point per value.
set -uo pipefail
E=/perf_apps/liyingli/bench_agentx/e1ab
n="$(hostname -s)"
say() { echo "$(date -u +%FT%T) e1ab-holder $SLURM_JOB_ID $n $*" | tee -a "$E/logs/e1ab.out"; }
say "start: tag ${TAG:-br} PREFILL_EXTRA_ARGS=[${PREFILL_EXTRA_ARGS:-}] values $*, amdgpu $(cat /sys/module/amdgpu/version)"
case "$(cat /sys/module/amdgpu/version)" in 6.19.*) say "amdgpu 6.19 node, not running"; exit 3 ;; esac
if ! docker image inspect e1ab-sglang:1112 >/dev/null 2>&1; then
    export DOCKER_CONFIG="$(mktemp -d /tmp/llying-docker.XXXX)"
    (cd "$E/image" && docker build --pull -t e1ab-sglang:1112 .) > "$E/logs/build-$n.log" 2>&1 || { say "image build failed"; exit 1; }
    say "image built"
fi
for me in "$@"; do
    EST_MIN=25 MIN_EVICT=$me GSP_TAG=${TAG:-br} GSP_G=1024 GSP_S=11000 GSP_P=1 GSP_Q=8192 GSP_O=64 \
        bash "$E/e1ab_item.sh" "gsp${TAG:-br}-me$me" gsp 1node > "$E/logs/gsp${TAG:-br}-me$me-$n.item.log" 2>&1
    say "me=$me exit $?"
done
say "done"
