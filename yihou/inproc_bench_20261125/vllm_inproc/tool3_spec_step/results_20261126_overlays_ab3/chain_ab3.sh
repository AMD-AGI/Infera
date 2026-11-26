#!/bin/bash
# kf overlays A/B3 (A stock v3xa, B3 indexer_4707282 + sparse_mla_aa2a608 v3xe), ROCR 6,7; order A-B3, B3-A, A-B3, B3-A; then 1 profile per arm.
cd /data/yihou/inproc_bench_20261125
export KIT_DIR=$PWD/kit_v3 WORK_DIR=$PWD/work_v3 PINNED_DIGEST=sha256:57ab3dc0e10f873d117f6c7d581737d3f80c5b6062d25ef38c167163f79d9326
export ARGS_FILES=presets/dsv41_llying_decode.args
S=$KIT_DIR/vllm_inproc/tool3_spec_step/scripts
ARGS=(--tp-size 1 --dp-size 2 --batch-size 448 --input-len 173000 --output-len 1024 --warmup-steps 10 --seed 1234)
snap() { for a in A E; do echo "  cache $a: triton=$(find cache_v3x$a/triton -type f 2>/dev/null | wc -l) flydsl=$(find cache_v3x$a/flydsl -type f 2>/dev/null | wc -l)"; done; }
run() { local arm=$1 rep=$2 c=yihou-inproc-v3xa; [ $arm = B3 ] && c=yihou-inproc-v3xe
  echo "$(date -u +%FT%TZ) start ${arm}r${rep} on $c"
  CONTAINER=$c bash $S/run_decode.sh v3z_${arm}r${rep}_u173k "${ARGS[@]}" > logs_v3/v3z_${arm}r${rep}.log 2>&1
  echo "$(date -u +%FT%TZ) end ${arm}r${rep} rc=$?"; }
snap
run A 1; run B3 1; snap
run B3 2; run A 2
run A 3; run B3 3
run B3 4; run A 4
for arm in A B3; do c=yihou-inproc-v3xa; [ $arm = B3 ] && c=yihou-inproc-v3xe
  CONTAINER=$c MAX_STEPS=40 PROFILE_START=20 PROFILE_NUM=5 PROFILE_RANKS=all bash $S/run_profile.sh p_v3z_${arm}_u173k > logs_v3/p_v3z_${arm}.log 2>&1
  echo "$(date -u +%FT%TZ) profile $arm rc=$?"; done
snap
echo chain_ab3 done
