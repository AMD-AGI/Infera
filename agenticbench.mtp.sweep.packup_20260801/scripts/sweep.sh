#!/bin/bash
# sglang's own bench_serving against the kv-aware router: 8 runs, ONE server.
#
# THE MATRIX (operator decision: paired percentiles, P99 dropped)
#
#     P50 point : ISL  74,000  OSL   320
#     P90 point : ISL 155,000  OSL 3,300
#     x concurrency {1, 32, 64, 128}
#
# Pairing ISL with its OWN percentile's OSL keeps each point a real Case A
# request shape. A full 3x3 cross would spend most of its wall clock on shapes
# the workload never produces (e.g. 235K in / 320 out).
#
# ONE SERVER FOR ALL EIGHT, sized for the largest point -- no per-workload
# retuning, so every number is comparable and no result is a re-tuned special
# case. Headroom was computed, not assumed: the KV pool is 3,260,992 tokens per
# DP rank, so at the P90 point (158,300 tok/req) a rank holds ~20 requests and
# the 8-rank node ~164 -- conc=128 fits.
#
# WHY A DISTINCT SEED PER CONCURRENCY. bench_serving generates prompts
# deterministically, so with one fixed seed a larger N's prompt set is a
# SUPERSET of a smaller N's. Running 128 then 64 would leave 64's prompts
# already cached and its "cold" run would silently be warm. A distinct seed per
# N keeps the sets disjoint.
#
# ACCEPTANCE: bench_serving reads `avg_spec_accept_length` from the server's
# /server_info (benchmark/serving.py:1525) and prints `Accept length`. This is
# where the MTP acceptance number comes from -- the AgenticBench driver has no
# such path.
#
# Usage: sweep.sh [tag]
set -u
TAG="${1:-sweep1}"
W=/shared_nfs/yihou_agbench_mtp
PJOB="${PJOB:-24300}"; PIP="${PIP:-10.245.157.89}"
CTR=agbench_mtp
ROUTER="http://$PIP:8190"
MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
BENCH=/sgl-workspace/sglang/python/sglang/bench_serving.py
OUT=$W/bench/$TAG

spur exec "$PJOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg; docker exec $CTR mkdir -p $OUT" 2>&1 | grep -v libtinfow

run() {  # label isl osl conc seed nprompts
  local L=$1 ISL=$2 OSL=$3 C=$4 SEED=$5 NP=$6
  local F="$OUT/${TAG}_${L}_isl${ISL}_osl${OSL}_conc${C}.json"
  echo
  echo "##### $L  ISL=$ISL OSL=$OSL conc=$C prompts=$NP seed=$SEED  $(date -u +%T)"
  spur exec "$PJOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
    docker exec $CTR python3 $BENCH \
      --backend sglang-oai --base-url $ROUTER \
      --model glm5.2-mxfp4 --tokenizer $MODEL \
      --dataset-name random \
      --random-input-len $ISL --random-output-len $OSL --random-range-ratio 1.0 \
      --num-prompts $NP --max-concurrency $C --request-rate inf \
      --warmup-requests 0 --seed $SEED \
      --output-file $F 2>&1 | grep -vE 'it/s\]|^Namespace|Warning|^ *$' | tail -32" 2>&1 | grep -v libtinfow
  # kvd + acceptance snapshot after each point, so a per-point story is possible
  spur exec "$PJOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
    docker exec $CTR bash -c 'python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock > $OUT/kvd_${L}_conc${C}.json 2>&1;
      curl -sf -m10 http://10.245.146.87:30001/server_info 2>/dev/null > $OUT/serverinfo_${L}_conc${C}.json || true'" 2>&1 | grep -v libtinfow
}

echo "############ bench_serving sweep tag=$TAG  start $(date -u +%FT%TZ)"
for C in 1 32 64 128; do
  # num_prompts = 2x concurrency: two full waves, enough to measure without
  # letting a 155K-token conc=128 point run for hours. conc=1 gets 4 so its
  # percentiles are not a single sample.
  NP=$(( C * 2 )); [ "$C" = "1" ] && NP=4
  run p50  74000  320 "$C" $(( 1000 + C )) "$NP"
done
for C in 1 32 64 128; do
  NP=$(( C * 2 )); [ "$C" = "1" ] && NP=4
  run p90 155000 3300 "$C" $(( 2000 + C )) "$NP"
done
echo
echo "############ done $(date -u +%FT%TZ)"
spur exec "$PJOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg; docker exec $CTR ls -la $OUT" 2>&1 | grep -v libtinfow
