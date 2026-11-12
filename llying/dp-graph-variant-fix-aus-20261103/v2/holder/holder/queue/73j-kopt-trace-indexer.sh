#!/usr/bin/env bash
# kopt 73j (trace analysis only, no GPU): per-call time of the indexer kernels in the TP8 PP8 MegaMoeTP decode traces --
# low concurrency (rows <= 48: KF fused indexer) vs c160 (rows > 48: SGLang's standard indexer), to size "run the fused
# indexer in chunks of 48 rows" for busy batches.
A=/perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/ana
T=/perf_apps/liyingli/vfix-logs/opt/trace
O=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench/out/kopt73j.log
echo "$(date -u +%FT%TZ) start on $(hostname)" > $O
P='verify  |dual_gemv|qk_rope_hadamard|logits_hist|topk_transform|paged_mqa|_topk_|kn_entry|hadamard_transform|indexer_k_quant|act_quant|Layernorm2d|mul_unsqueeze|hgemm_bf16_t(64x32x128x4|16x32x128x[67])'
for d in opt-megatp-mt4q2-lowbs-strace/traces/decode-c8-204925 opt-megatp-mt4q2-lowbs-strace/traces/decode-c24-205156 opt-megatp-mt4q2-lowbs-strace/traces/decode-c40-205427 opt-megatp-mt4q2-lowbs-strace/traces/decode-c64-205658 opt-megatp-mt4q2-pp8-strace/traces/decode-c160-193241; do
    echo "== $d" >> $O
    timeout 900 python3 $A/step_cats.py --top 80 $T/$d/*.trace.json.gz 2>&1 | grep -E "$P" >> $O
done
echo "$(date -u +%FT%TZ) done" >> $O
