#!/usr/bin/env bash
# U17 end to end on vLLM main 8828834fb (nightly 9c1d45c2 + synced Python code): two DP4 TP1 GLM-5.2 servers side by
# side with --enable-expert-parallel and AITER fused shared experts (the default), GPUs 0-3 = main + U17, GPUs 4-7 =
# main, then the correctness gate on both at once. Expected: main shows the bug (GSM8K far below 190 / 200, as on
# v0.34.0: 1 / 200), main + U17 passes (>= 190). Whole-node step (plan GPUS all).
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u17-aiter-fused-shared-ep-main rev-parse HEAD
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/main2 rev-parse HEAD
# key: sha256sum /home/liyingli/bench_agentx/upstream-20261119/e2e/agg_up_main.sh /home/liyingli/bench_agentx/upstream-20261119/e2e/gate.sh
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
need_ready nightly model || exit 1
E=$U/e2e
fix=$(wt u17-aiter-fused-shared-ep-main) base=$(wt main2)
for t in "$fix" "$base"; do
    compiled=$(git -C "$t" diff --name-only "$NIGHTLY_COMMIT" HEAD -- csrc/ cmake/ CMakeLists.txt setup.py \
        | grep -v -E '^csrc/cpu/|^cmake/cpu_extension\.cmake|^cmake/external_projects/vllm_flash_attn\.cmake|^csrc/fused_qknorm_rope_kernel\.cu$')
    [[ -z "$compiled" ]] || { echo "$t changes compiled code vs the image: $compiled"; exit 2; }
done
trap 'docker rm -f up-e2e-fix up-e2e-base > /dev/null 2>&1' EXIT
bash "$E/agg_up_main.sh" up-e2e-fix 0,1,2,3 28100 29610 "$fix" "$OUT/logs" --enable-expert-parallel > "$OUT/up-fix.txt" 2>&1 &
pf=$!
bash "$E/agg_up_main.sh" up-e2e-base 4,5,6,7 28200 29710 "$base" "$OUT/logs" --enable-expert-parallel > "$OUT/up-base.txt" 2>&1 &
pb=$!
wait $pf; rf=$?
wait $pb; rb=$?
cat "$OUT/up-fix.txt" "$OUT/up-base.txt"
(( rf == 0 && rb == 0 )) || { echo "a server did not start (fix rc=$rf, base rc=$rb)"; exit 3; }
grep -a -m2 -E "Expert parallelism is enabled|Local/global" "$OUT/logs/up-e2e-fix.log" | cut -c1-200
bash "$E/gate.sh" fix 28100 "$OUT" > "$OUT/gate-fix.txt" 2>&1 &
gf=$!
bash "$E/gate.sh" base 28200 "$OUT" > "$OUT/gate-base.txt" 2>&1 &
gb=$!
wait $gf $gb
cat "$OUT/gate-fix.txt" "$OUT/gate-base.txt"
python3 - "$OUT" <<'EOF' | tee "$OUT/verdict.txt"
import re, sys
out = sys.argv[1]
def correct(label):
    try:
        s = open(f"{out}/gate-{label}/summary.txt").read()
    except OSError:
        return None
    m = re.search(r"correct=(\d+)/(\d+)", s)
    return int(m.group(1)) if m else None
fix, base = correct("fix"), correct("base")
print(f"GSM8K correct: main + U17 {fix}/200, main {base}/200 (DP4 + EP + fused shared experts)")
ok = fix is not None and base is not None and fix >= 190 and base < 150
print("verdict:", "PASS (bug on main, fixed by U17)" if ok else "CHECK (see the gate outputs)")
sys.exit(0 if ok else 1)
EOF
exit "${PIPESTATUS[0]}"
