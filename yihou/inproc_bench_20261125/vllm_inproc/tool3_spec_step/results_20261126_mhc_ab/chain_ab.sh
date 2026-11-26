#!/bin/bash
# mHC overlay A/B (A = stock yihou-inproc-v3a, B = kf round-4 overlay yihou-inproc-v3b), interleaved, GPUs 6,7.
cd /data/yihou/inproc_bench_20261125
export KIT_DIR=$PWD/kit_v3 WORK_DIR=$PWD/work_v3 PINNED_DIGEST=sha256:57ab3dc0e10f873d117f6c7d581737d3f80c5b6062d25ef38c167163f79d9326
export ARGS_FILES=presets/dsv41_llying_decode.args
S=$KIT_DIR/vllm_inproc/tool3_spec_step/scripts
ARGS=(--tp-size 1 --dp-size 2 --batch-size 448 --input-len 173000 --output-len 1024 --warmup-steps 10 --seed 1234)
for run in A1 B1 A2 B2 A3 B3; do
  arm=${run:0:1}; c=yihou-inproc-v3a; [ "$arm" = B ] && c=yihou-inproc-v3b
  echo "$(date -u +%FT%TZ) start $run on $c"
  CONTAINER=$c bash $S/run_decode.sh v3ab_${run}_u173k "${ARGS[@]}" > logs_v3/v3ab_${run}.log 2>&1
  echo "$(date -u +%FT%TZ) end $run rc=$?"
  [ "$arm" = B ] && ls -d cache_v3B/aiter_jit/module_mega_mhc_kfr4_* 2>&1 | sed "s/^/  jit: /"
done
for arm in A B; do
  c=yihou-inproc-v3a; [ "$arm" = B ] && c=yihou-inproc-v3b
  CONTAINER=$c MAX_STEPS=40 PROFILE_START=20 PROFILE_NUM=5 PROFILE_RANKS=all bash $S/run_profile.sh p_v3ab_${arm}_u173k > logs_v3/p_v3ab_${arm}.log 2>&1
  echo "$(date -u +%FT%TZ) profile $arm rc=$?"
done
echo chain_ab done
