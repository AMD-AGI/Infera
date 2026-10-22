#!/usr/bin/env bash
# Copy-only delta for glm52.v0521-offloaderv2-highconc.packup_20261015 (events 11:40 -> 13:20 UTC 10-15). Sources read-only.
set -uo pipefail
P=/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-offloaderv2-highconc.packup_20261015
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness; R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
N=/shared_nfs/yihou_glm52/v0521cap2_20261013; L=/home/yihou/dev/git/infera.glm52.view/glm52.v0521-cap-c44c48-23k_20261012
LIM=4000000
c(){ [ -e "$1" ] && cp -p "$1" "$2" || echo "MISSING $1"; }
gz(){ [ -e "$1" ] || { echo "MISSING $1"; return; }; local n; n=$(gzip -c "$1" | wc -c)
  if [ "$n" -lt "$LIM" ]; then gzip -c "$1" > "$2"; else echo "TOOBIG-GZ $1 ($n)"; return 1; fi; }
excerpt(){ { echo "### EXCERPT of $1 ($(stat -c %s "$1") bytes): head 2 MB / key lines / tail 1 MB"; head -c 2000000 "$1"; echo; echo "### ---- KEY LINES ----"
    grep -anE 'Traceback|Error|OOM|out of memory|max_total_num_tokens|Memory pool|avail mem|offloader|warmup|Engine startup|SIGTERM|Terminated' "$1" | head -c 1000000
    echo; echo "### ---- TAIL ----"; tail -c 1000000 "$1"; } | gzip -c > "$2"; }
for f in run.v3s.yihou.sh pipeline.v3s.yihou.sh chain_cap2ls.yihou.sh run136_ofg4h2.yihou.sh env.ofg4h2-136.yihou.sh; do c $H/$f $P/scripts/harness/; done
c $N/mk_streamcopies.yihou.py $P/scripts/tools/; c $N/packup_delta2.yihou.sh $P/scripts/tools/; c $N/dryrun.ofg4h2-136.yihou.txt $P/results/dryruns/
for r in ofg3b ofg2h3 ofg4h2; do d=$P/results/rounds/$r; mkdir -p $d
  for f in status.tsv census.txt audit.txt stop.seq.log; do [ -e $R/$r/$f ] && cp -p $R/$r/$f $d/; done
  for f in $R/$r.chain.log $R/$r.pipeline.log $R/$r.cont.log $R/$r.run136.log; do [ -e $f ] && cp -p $f $d/; done
  for f in $R/$r/agentx-*.log; do [ -e $f ] && gzip -c $f > $d/$(basename $f).gz; done; done
for b in ofg3b/agentx-c56-A3 ofg2h3/agentx-c56-A1; do s=$R/$b; d=$P/results/${b%%/*}-${b#*agentx-}; mkdir -p $d
  for f in benchmark_command.txt runtime.env client.container-exit-code client.shell-exit-code; do [ -e $s/$f ] && cp -p $s/$f $d/; done
  for f in benchmark.log runner.log; do [ -e $s/$f ] && gzip -c $s/$f > $d/$f.gz; done; done
for r in ofg3b ofg2h3; do mkdir -p $P/logs/server/$r; for f in $R/$r/server-logs/*.log; do [ -e $f ] || continue; b=$(basename $f)
  gz $f $P/logs/server/$r/$b.gz || excerpt $f $P/logs/server/$r/$b.excerpt.txt.gz; done; done
for r in ofg3b ofg2h3; do for f in $N/runs/$r/metrics.A*.tsv; do [ -e $f ] && { gz $f $P/results/metrics/$r.$(basename $f).gz || true; }; done; done
python3 -B $N/results_csv_v15b.yihou.py highconc > $P/results/results.csv
c $L/debug/poll.md $P/analysis/ledger/; c $L/debug/working_process.md $P/analysis/ledger/
echo DELTA-DONE
