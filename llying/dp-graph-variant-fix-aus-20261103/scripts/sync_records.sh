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
cp "$O/router-decode-request-weight.diff" "$D/opt/"
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
cp "$L/patrol/patrol.log" "$D/results/" 2>/dev/null
if [[ -n "${COMMIT:-}" ]]; then
    cd "$REPO" && git add -A llying/dp-graph-variant-fix-aus-20261103 &&
        git commit -q -m "$COMMIT" && git pull -q --rebase origin dev/pd_opt/glm_5.2_agentx &&
        git push -q origin HEAD 2>&1 | tail -2
    git -C "$REPO" log --oneline -1
fi
du -sh "$D"
