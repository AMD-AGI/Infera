#!/usr/bin/env bash
# Copy the AUS verification / optimisation scripts, configs and results into the Infera checkout
# (llying/dp-graph-variant-fix-aus-20261103/); raw probe outputs gzipped. Commit with COMMIT="message" (and push).
set -uo pipefail
A=/perf_apps/liyingli/bench_agentx/vfix-aus-20261103
O=/perf_apps/liyingli/bench_agentx/opt-aus-20261103
PKG=/perf_apps/liyingli/bench_agentx/vfix-20261103
L=/perf_apps/liyingli/vfix-logs
REPO=/home/liyingli/code/bench_agentx/Infera
D=$REPO/llying/dp-graph-variant-fix-aus-20261103
mkdir -p "$D"/{scripts,config,opt/scripts,opt/items,opt/config,results/verify,results/ref,results/opt,upstream}
cp "$A"/scripts/*.sh "$D/scripts/"
cp "$PKG/config/site.local.env" "$D/config/site.local.env.aus"
cp "$O"/scripts/*.sh "$D/opt/scripts/"; cp "$O"/items/*.sh "$D/opt/items/"
for f in "$O"/pkg/config/arm-*.env; do [[ -e "$PKG/config/$(basename "$f")" ]] || cp "$f" "$D/opt/config/"; done
mkdir -p "$D/opt/scripts/ana" && cp "$O"/scripts/ana/*.py "$D/opt/scripts/ana/"
cp "$O/pkg/scripts/synth_decode_load.py" "$D/opt/scripts/"
mkdir -p "$D/opt/kopt-bench" "$D/opt/overlay-kopt/kopt_megatp"
cp /perf_apps/liyingli/bench_agentx/kopt-20261109/bench/*.py /perf_apps/liyingli/bench_agentx/kopt-20261109/bench/*.sh "$D/opt/kopt-bench/"
cp "$O/pkg/kit/overlay-kopt/sitecustomize.py" "$D/opt/overlay-kopt/"
cp "$O/pkg/kit/overlay-kopt/kopt_megatp/__init__.py" "$D/opt/overlay-kopt/kopt_megatp/"
mkdir -p "$D/opt/overlay-gemmmerge/kopt_gemm_merge" "$D/opt/overlay-variants"
cp "$O/pkg/kit/overlay-gemmmerge/sitecustomize.py" "$D/opt/overlay-gemmmerge/"
cp "$O/pkg/kit/overlay-gemmmerge/kopt_gemm_merge/__init__.py" "$D/opt/overlay-gemmmerge/kopt_gemm_merge/"
cp "$O"/pkg/kit/overlay-variants/*.py "$D/opt/overlay-variants/"
mkdir -p "$D/opt/aiter_csv" && cp "$O"/pkg/kit/aiter_csv/kopt_*.csv "$D/opt/aiter_csv/"
cp "$O/router-decode-request-weight.diff" "$D/opt/"
mkdir -p "$D/opt/dense-fp8" && cp "$O"/dense-fp8/*.diff "$O/dense-fp8/patched/layers/quantization/quark/quark.py" "$D/opt/dense-fp8/" 2>/dev/null
(cd /perf_apps/liyingli/bench_agentx && diff -ru vfix-20261103/kit opt-aus-20261103/pkg/kit) > "$D/opt/kit-router-hooks.diff"
cp "$L/verify.out" "$L/unit.log" "$D/results/" 2>/dev/null
python3 "$PKG/scripts/judge.py" "$L/verify.out" > "$D/results/judge.txt" 2>&1
for f in "$L"/verify/*.review.txt "$L"/ref/*.review.txt "$L"/opt/verify/*.review.txt; do
    [[ -e "$f" ]] || continue
    case "$f" in */ref/*) t=ref ;; */opt/*) t=opt ;; *) t=verify ;; esac
    cp "$f" "$D/results/$t/"
done
for f in "$L"/verify/*.jsonl "$L"/ref/*.jsonl "$L"/opt/verify/*.jsonl; do
    [[ -e "$f" ]] || continue
    case "$f" in */ref/*) t=ref ;; */opt/*) t=opt ;; *) t=verify ;; esac
    g="$D/results/$t/$(basename "$f").gz"
    [[ -e "$g" && "$g" -nt "$f" ]] || gzip -9c "$f" > "$g"
done
cp "$L/ref/ref.out" "$D/results/ref/" 2>/dev/null
cp "$L"/score-*.txt "$D/results/" 2>/dev/null
cp "$L/opt/opt.out" "$L"/opt/score-*.txt "$D/results/opt/" 2>/dev/null
mkdir -p "$D/v2/scripts" "$D/v2/holder" "$D/results/v2/sweeps"
cp /perf_apps/liyingli/bench_agentx/v2-aus/*.sh /apps/liyingli/jobs/holder.sh /apps/liyingli/jobs/holder1.sh "$D/v2/scripts/"
for h in holder holder1; do
    for s in queue running parked; do
        mkdir -p "$D/v2/holder/$h/$s" && rm -f "$D/v2/holder/$h/$s"/*.sh && cp /apps/liyingli/$h/$s/*.sh "$D/v2/holder/$h/$s/" 2>/dev/null
    done
done
cp "$L/v2/v2.out" "$L"/v2/*.review.txt "$D/results/v2/" 2>/dev/null
for s in "$L"/v2/sweeps/*/summary.md; do [[ -e "$s" ]] && cp "$s" "$D/results/v2/sweeps/$(basename "$(dirname "$s")").md"; done
cp "$L/patrol/patrol.log" "$D/results/" 2>/dev/null
if [[ -n "${COMMIT:-}" ]]; then
    cd "$REPO" && git add -A llying/dp-graph-variant-fix-aus-20261103 &&
        git commit -q -m "$COMMIT" && git pull -q --rebase origin dev/pd_opt/glm_5.2_agentx &&
        git push -q origin HEAD 2>&1 | tail -2
    git -C "$REPO" log --oneline -1
fi
du -sh "$D"
