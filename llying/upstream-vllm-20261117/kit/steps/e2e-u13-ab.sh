#!/usr/bin/env bash
# U13 end to end on vLLM main 8828834fb (nightly 9c1d45c2 + synced Python code): two DP4 TP1 GLM-5.2 servers side by
# side (no EP, AITER fused shared experts), GPUs 0-3 = main + U13, GPUs 4-7 = main; correctness gate on both, then the
# track A decode bench (104 requests x 64K in / 4K out, decode step p50 in the all-decoding window) twice on both at
# once. Checks that only the U13 server logs "Using AITER custom AG/RS". Whole-node step (plan GPUS all).
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u13-aiter-agrs-full-capture-main rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/main2 rev-parse HEAD
# key: sha256sum /home/liyingli/bench_agentx/upstream-20261119/e2e/agg_up_main.sh /home/liyingli/bench_agentx/upstream-20261119/e2e/gate.sh /home/liyingli/bench_agentx/upstream-20261119/e2e/decode_bench.sh
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
need_ready nightly model || exit 1
E=$U/e2e
pr=$(wt u13-aiter-agrs-full-capture-main) base=$(wt main2)
for t in "$pr" "$base"; do
    compiled=$(git -C "$t" diff --name-only "$NIGHTLY_COMMIT" HEAD -- csrc/ cmake/ CMakeLists.txt setup.py \
        | grep -v -E '^csrc/cpu/|^cmake/cpu_extension\.cmake|^cmake/external_projects/vllm_flash_attn\.cmake|^csrc/fused_qknorm_rope_kernel\.cu$')
    [[ -z "$compiled" ]] || { echo "$t changes compiled code vs the image: $compiled"; exit 2; }
done
trap 'docker rm -f up-e2e-pr up-e2e-base > /dev/null 2>&1' EXIT
bash "$E/agg_up_main.sh" up-e2e-pr 0,1,2,3 28100 29610 "$pr" "$OUT/logs" > "$OUT/up-pr.txt" 2>&1 &
pp=$!
bash "$E/agg_up_main.sh" up-e2e-base 4,5,6,7 28200 29710 "$base" "$OUT/logs" > "$OUT/up-base.txt" 2>&1 &
pb=$!
wait $pp; rp=$?
wait $pb; rb=$?
cat "$OUT/up-pr.txt" "$OUT/up-base.txt"
(( rp == 0 && rb == 0 )) || { echo "a server did not start (pr rc=$rp, base rc=$rb)"; exit 3; }
bash "$E/gate.sh" pr 28100 "$OUT" > "$OUT/gate-pr.txt" 2>&1 &
g1=$!
bash "$E/gate.sh" base 28200 "$OUT" > "$OUT/gate-base.txt" 2>&1 &
g2=$!
wait $g1 $g2
cat "$OUT/gate-pr.txt" "$OUT/gate-base.txt"
for r in 1 2; do
    bash "$E/decode_bench.sh" 28100 "$OUT/bench-pr-$r" > /dev/null 2>&1 &
    b1=$!
    bash "$E/decode_bench.sh" 28200 "$OUT/bench-base-$r" > /dev/null 2>&1 &
    b2=$!
    wait $b1 $b2
done
{
    echo "AITER AG/RS log lines: main + U13 $(grep -a -c 'Using AITER custom AG/RS' "$OUT/logs/up-e2e-pr.log"), main $(grep -a -c 'Using AITER custom AG/RS' "$OUT/logs/up-e2e-base.log")"
    for s in pr base; do
        for r in 1 2; do
            echo "$s run $r: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("step_p50_ms", d.get("step_p50_ms"), "mean", d.get("step_mean_ms"), "p99", d.get("step_p99_ms"), "n_itl", d.get("n_itl"), "window_s", d.get("window_s"))' "$OUT/bench-$s-$r/window.json" 2>&1)"
        done
    done
} | tee "$OUT/verdict.txt"
grep -q "main + U13 [1-9]" "$OUT/verdict.txt" && [[ -s "$OUT/bench-pr-1/window.json" && -s "$OUT/bench-base-1/window.json" ]]
