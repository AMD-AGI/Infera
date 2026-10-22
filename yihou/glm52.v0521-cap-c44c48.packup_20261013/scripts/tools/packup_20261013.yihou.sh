#!/usr/bin/env bash
# Assemble the task-v0521cap pack-up (copy only; originals untouched).
set -uo pipefail
P=/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-cap-c44c48.packup_20261013
DOC=/home/yihou/dev/git/infera.glm52.view/glm52.v0521-cap-c44c48-23k_20261012
WS=/shared_nfs/yihou_glm52/v0521dbg_20261010; R=$WS/runs; H=$WS/harness; CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
mkdir -p $P/{spec,analysis/research,scripts/harness,scripts/tools,patches/runtime-mounts,results,logs}
cp -a $DOC/mission.md $DOC/PLAN.md $P/spec/; cp -a /home/yihou/dev/git/infera.glm52.view/CLAUDE.md $P/spec/CLAUDE.task.md
cp -a $DOC/debug/working_process.md $DOC/debug/poll.md $P/analysis/; cp -a $DOC/research/*.md $P/analysis/research/
cp -a /home/yihou/dev/git/infera.glm52.view/decode.kvcache.report.md $P/analysis/
# harness verbatim (3.2 MB), no pycache
cp -a $H/. $P/scripts/harness/; find $P/scripts/harness -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null
cp -a $CAP/tools/*.py $CAP/tools/*.sh $P/scripts/tools/ 2>/dev/null
# patches: runshare v1/v2 (full file + diff vs image decode.py md5 bd1e5fe7)
for f in decode.yihou-runshare.py decode.yihou-runshare.diff decode.yihou-runshare2.py decode.yihou-runshare2.diff; do cp -a $H/mounts-v0521/src/$f $P/patches/runtime-mounts/; done
cp -a $H/mounts.yihou.decode-runshare.tsv $H/mounts.yihou.decode-runshare2.tsv $H/config.yihou.runshare.sh $H/config.yihou.runshare2.sh $H/config.yihou.mc4340n.sh $H/config.yihou.mc4340prio.sh $P/patches/runtime-mounts/
# results: per run JSON + benchmark.log + command/env (no aiperf_artifacts)
res(){ local src=$1 dst=$2; mkdir -p $P/results/$dst; for f in agentx_conc*.json benchmark.log benchmark_command.txt runtime.env; do [ -e $src/$f ] && cp -a $src/$f $P/results/$dst/; done; [ -e $P/results/$dst/benchmark.log ] && gzip -f $P/results/$dst/benchmark.log; }
res $R/e2j/agentx-c44-R3 X1-e2j-C44-R3
res $R/k1/agentx-c44 k1-A1-C44; res $R/k1/agentx-c44-A2 k1-A2-C44
res $R/q1/agentx-c44 q1-A1-C44; res $R/q1/agentx-c44-A2 q1-A2-C44; res $R/q1/agentx-c48-A3 q1-A3-C48; res $R/q1/agentx-c52-A4 q1-A4-C52
res $R/pc40c/agentx-c40 pc40c-A1-C40-runshare1; res $R/pc40c/agentx-c40-A2 pc40c-A2-C40-runshare1
res $R/pc40d/agentx-c40 pc40d-A1-C40-runshare2
for r in k1 q1 pc40c pc40d; do mkdir -p $P/results/round-$r; for f in census.txt audit.txt status.tsv burst.txt burst2.txt dryrun; do [ -e $R/$r/$f ] && cp -a $R/$r/$f $P/results/round-$r/; done; done
cp -a $R/pc40c.smoke2.log $R/pc40c.chain.log $R/pc40d.chain.log $R/k1.chain.log $R/q1.chain.log $R/q1.A3c48.log $R/q1.A4c52.log $R/e2j.R3.log $P/results/ 2>/dev/null
cp -a $H/dryrun.{k1-102,q1-102,q1-137,q2-137,q3-102,q4-137,pc40-174,pc40c-102,pc40d-102,e2g-102}.yihou.txt $P/results/ 2>/dev/null
# /metrics TSVs gzipped
mkdir -p $P/results/metrics; for d in x1-e2j44 k1 q1 pc40c pc40d; do for f in $CAP/runs/$d/*.tsv; do [ -e "$f" ] && gzip -c $f > $P/results/metrics/$d.$(basename $f).gz; done; done
# server logs: gzip; q1 logs live on 137 local disk
for r in e2j k1 pc40c pc40d q1; do for f in $R/$r/server-logs/*.log; do [ -e "$f" ] && gzip -c $f > $P/logs/$r.$(basename $f).gz; done; done
ls -la $P/logs $P/results/metrics; du -sh $P
