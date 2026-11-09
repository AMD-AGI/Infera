#!/usr/bin/env bash
# EP8 MoE (33 experts, inter 2048, token key 1024) kernel sweep at M 704: for each block_m, pair the i-th gemm1 name with the
# i-th gemm2 name of the same tile M (gemm1 and gemm2 times are read separately, so one run measures both). One process per pair.
D=/perf_apps/liyingli/bench_agentx/kopt-20261109/tune/ep33
T=/perf_apps/liyingli/bench_agentx/kopt-20261109/configs/glm5_fp4_tuned_fmoe.csv
C=/sgl-workspace/aiter/aiter/configs
L=$C/tuned_fmoe.csv; for f in $C/model_configs/*tuned_fmoe*.csv; do case $f in *untuned*) ;; *) L=$L:$f;; esac; done
OUT=$D/sweep_results.txt
mkdir -p $D/sweep; : > $OUT
hdr=$(head -1 $D/cand_from1024.csv); base=$(tail -1 $D/cand_from1024.csv)
cd /perf_apps/liyingli/bench_agentx/kopt-20261109/bench
for bm in 32 64 16; do
  if [ $bm = 16 ]; then p1="flydsl_mxmoe_g1_a4w4_16x"; else p1="flydsl_moe1_afp4_wfp4_bf16_t${bm}x"; fi
  mapfile -t G1 < <(awk -F, 'NR>1{print $18}' $T | grep "^$p1" | sort -u)
  mapfile -t G2 < <(awk -F, 'NR>1{print $21}' $T | grep -E "^flydsl_moe2.*_t${bm}x.*(sbm${bm}|bnt2)$" | sort -u)
  n=${#G1[@]}; [ ${#G2[@]} -gt $n ] && n=${#G2[@]}
  for ((i = 0; i < n; i++)); do
    k1=${G1[$((i % ${#G1[@]}))]}; k2=${G2[$((i % ${#G2[@]}))]}
    c=$D/sweep/bm${bm}_$i.csv
    { echo "$hdr"; echo "$base" | awk -F, -v OFS=, -v bm=$bm -v k1=$k1 -v k2=$k2 '{$15=bm; $18=k1; $21=k2; print}'; } > $c
    r=$(AITER_CONFIG_FMOE=$L:$c timeout 300 python3 ep_sweep_one.py 704 2>&1 | grep -E "^RESULT|Error|error" | tail -1 | cut -c1-260)
    echo "bm$bm $k1 + $k2 -> $r" >> $OUT
  done
done
echo "SWEEP DONE $(date -u +%T)" >> $OUT
