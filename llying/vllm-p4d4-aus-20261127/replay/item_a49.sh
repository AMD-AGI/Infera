#!/usr/bin/env bash
# Holder item: the crusoe a49 round (pair12, side B) on this AUS node, through replay/R (assemble.sh).
# Env (all optional):
#   SE      score-env name (aus-a49); the round's results go to replay/R/results/$SE/B
#   GATE    1 = InferenceX eval (real acceptance, kit/eval.sh) first, then the score; 0 = score only (default 1)
#   FAST    1 = InferenceX fast mode (1200 s, 1 warmup per lane): bring-up only, not a score
#   CONC    client concurrency (152)
#   SCORE   0 = eval only
set -uo pipefail
. /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/site/env.sh
R=$W/replay/R
bash "$W/replay/assemble.sh" || exit 1
docker image inspect "$IMAGE_V0340" > /dev/null 2>&1 || { echo "no $IMAGE_V0340 on $NODE (queue build/build_v0340.sh first)"; exit 1; }
P=$W/results/probe-$NODE
for i in $(seq 120); do [ -e "$P/done" ] && break; sleep 10; done   # a probe re-run may still be going
[ -e "$P/done" ] || { echo "no probe for $NODE"; exit 1; }
gem=$(cat "$P/gem"); lib=$(cat "$P/libionic")
export GPU_NICS=$(awk '{print $2}' "$P/gpu-nic.txt" | paste -sd, -)
export HOST_LIBIONIC=$lib P_GEM=$gem D_GEM=$gem MC_DISABLE_DMABUF=$((1 - gem)) FAST=${FAST:-0}
SE=${SE:-aus-a49}
{ sed "s/^S_TAG=a49B-/S_TAG=${SE}B-/" "$R/once/score-a49-B.env"
  echo "S_GATE=${GATE:-1}"; echo "S_SCORE=${SCORE:-1}"; echo "S_CONC=${CONC:-152}"; } > "$R/once/score-$SE-B.env"
echo "$(ts) $NODE: $SE gem=$gem libionic_host=$lib nics=$GPU_NICS fast=$FAST gate=${GATE:-1} conc=${CONC:-152}"
cd "$R" && JOB=${SLURM_JOB_ID:-0} NODE0=$NODE SCORE_ENV=$SE PAIR_SIDES=B bash once/p4d4-pair12.sh
rc=$?
tail -5 "$R/logs/summary.out"
exit $rc
