#!/bin/bash
# kf combined overlays A/B/C (A stock v3xa, B indexer+sparse_mla v3xb, C B+moe v3xc), ROCR 6,7, order A B C x4, then 1 profile per arm.
cd /data/yihou/inproc_bench_20261125
export KIT_DIR=$PWD/kit_v3 WORK_DIR=$PWD/work_v3 PINNED_DIGEST=sha256:57ab3dc0e10f873d117f6c7d581737d3f80c5b6062d25ef38c167163f79d9326
export ARGS_FILES=presets/dsv41_llying_decode.args
S=$KIT_DIR/vllm_inproc/tool3_spec_step/scripts
ARGS=(--tp-size 1 --dp-size 2 --batch-size 448 --input-len 173000 --output-len 1024 --warmup-steps 10 --seed 1234)
snap() { for a in A B C; do echo "  cache $a: triton=$(find cache_v3x$a/triton -type f 2>/dev/null | wc -l) flydsl=$(find cache_v3x$a/flydsl -type f 2>/dev/null | wc -l)"; done; }
snap
for rep in 1 2 3 4; do for arm in A B C; do
  c=yihou-inproc-v3x$(echo $arm | tr ABC abc)
  echo "$(date -u +%FT%TZ) start ${arm}${rep} on $c"
  CONTAINER=$c bash $S/run_decode.sh v3x_${arm}${rep}_u173k "${ARGS[@]}" > logs_v3/v3x_${arm}${rep}.log 2>&1
  echo "$(date -u +%FT%TZ) end ${arm}${rep} rc=$?"
  [ $rep = 1 ] && snap
done; done
for arm in A B C; do
  c=yihou-inproc-v3x$(echo $arm | tr ABC abc)
  CONTAINER=$c MAX_STEPS=40 PROFILE_START=20 PROFILE_NUM=5 PROFILE_RANKS=all bash $S/run_profile.sh p_v3x_${arm}_u173k > logs_v3/p_v3x_${arm}.log 2>&1
  echo "$(date -u +%FT%TZ) profile $arm rc=$?"
done
snap
echo chain_abc done
