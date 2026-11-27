#!/bin/bash
# kf overlays B3 vs B5 (B3 = indexer_4707282 + sparse_mla_aa2a608 in v3xe; B5 = sparse_mla_aa2a608 + indexer_0445826 in v3xg), ROCR 6,7;
# 6 rounds alternating B3-B5 / B5-B3, then 1 profile per arm. Run id stepE-overlay-ab5-20261126.
cd /data/yihou/inproc_bench_20261125
export KIT_DIR=$PWD/kit_v3 WORK_DIR=$PWD/work_v3 PINNED_DIGEST=sha256:57ab3dc0e10f873d117f6c7d581737d3f80c5b6062d25ef38c167163f79d9326
export ARGS_FILES=presets/dsv41_llying_decode.args
S=$KIT_DIR/vllm_inproc/tool3_spec_step/scripts
ARGS=(--tp-size 1 --dp-size 2 --batch-size 448 --input-len 173000 --output-len 1024 --warmup-steps 10 --seed 1234)
snap() { for a in E G; do echo "  cache $a: triton=$(find cache_v3x$a/triton -type f 2>/dev/null | wc -l) flydsl=$(find cache_v3x$a/flydsl -type f 2>/dev/null | wc -l)"; done; }
late() { [ $(date -u +%H%M) -ge 1825 ]; }
run() { if late; then echo "$(date -u +%FT%TZ) SKIP $1r$2 (past 18:25Z)"; return; fi; local arm=$1 rep=$2 c=yihou-inproc-v3xe; [ $arm = B5 ] && c=yihou-inproc-v3xg
  echo "$(date -u +%FT%TZ) start ${arm}r${rep} on $c"
  CONTAINER=$c bash $S/run_decode.sh v3v_${arm}r${rep}_u173k "${ARGS[@]}" > logs_v3/v3v_${arm}r${rep}.log 2>&1
  echo "$(date -u +%FT%TZ) end ${arm}r${rep} rc=$?"; }
snap
run B3 1; run B5 1; snap
run B5 2; run B3 2
run B3 3; run B5 3
run B5 4; run B3 4
run B3 5; run B5 5
run B5 6; run B3 6
for arm in B3 B5; do if late; then echo "$(date -u +%FT%TZ) SKIP profile $arm (past 18:25Z)"; continue; fi; c=yihou-inproc-v3xe; [ $arm = B5 ] && c=yihou-inproc-v3xg
  CONTAINER=$c MAX_STEPS=40 PROFILE_START=20 PROFILE_NUM=5 PROFILE_RANKS=all bash $S/run_profile.sh p_v3v_${arm}_u173k > logs_v3/p_v3v_${arm}.log 2>&1
  echo "$(date -u +%FT%TZ) profile $arm rc=$?"; done
snap
echo chain_ab5 done
