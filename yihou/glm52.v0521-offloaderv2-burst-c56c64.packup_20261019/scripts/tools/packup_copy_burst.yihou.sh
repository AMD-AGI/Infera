#!/usr/bin/env bash
# Copy-only pack-up builder for glm52.v0521-offloaderv2-burst-c56c64.packup_20261019 (run on a spur login host).
# Never deletes or moves anything; sources are read-only. Writes only inside $P.
set -uo pipefail
P=/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-offloaderv2-burst-c56c64.packup_20261019
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
L=/home/yihou/dev/git/infera.glm52.view/glm52.v0521-cap-c44c48-23k_20261012
LIM=4000000
c(){ [ -e "$1" ] && cp -p "$1" "$2" || echo "MISSING $1"; }
gz(){ [ -e "$1" ] || { echo "MISSING $1"; return 1; }
  local n; n=$(gzip -c "$1" | wc -c)
  if [ "$n" -lt "$LIM" ]; then gzip -c "$1" > "$2"; else echo "TOOBIG-GZ $1 ($n)"; return 1; fi; }
excerpt(){ { echo "### EXCERPT of $1 ($(stat -c %s "$1") bytes): head 1 MB / key lines (<=1 MB) / tail 1 MB"
    echo "### ---- HEAD ----"; head -c 1000000 "$1"
    echo; echo "### ---- KEY LINES ----"
    grep -anE 'Traceback|Error|Exception|OOM|out of memory|max_total_num_tokens|Memory pool|avail mem|Load weight end|offloader|warmup|Engine startup|SIGTERM|Terminated|retract' "$1" | head -c 1000000
    echo; echo "### ---- TAIL ----"; tail -c 1000000 "$1"; } | gzip -c > "$2"; }
SCORED="ofg2h5 ofg3h2 ofg2h17 ofg2h21 ofg3h14 ofg2h22 ofg4h4 ofg4h5"
LOST="ofg2h4 ofg2h7 ofg2h8 ofg2h9 ofg2h10 ofg2h11 ofg2h12 ofg2h13 ofg2h14 ofg2h15 ofg2h16 ofg2h18 ofg2h19 ofg2h20 ofg3h ofg3h3 ofg3h4 ofg3h5 ofg3h6 ofg3h7 ofg3h8 ofg3h9 ofg3h10 ofg3h11 ofg3h12 ofg3h13"
mkdir -p $P/scripts/harness $P/scripts/session/shimbin $P/scripts/tools $P/patches $P/results/rounds $P/results/metrics \
         $P/results/dryruns $P/results/pyspy $P/results/unofficial $P/env $P/logs/server $P/analysis/ledger $P/spec
# ---- harness (verbatim)
cp -pr $H/bench-harness $H/tools $H/mounts-v0521 $P/scripts/harness/
for f in config.yihou.v0521-base.sh config.yihou.E2-target.sh config.yihou.offload.sh config.yihou.tokw2.sh \
  mounts.yihou.offload-attrs.tsv mounts.yihou.v2-regbase.tsv mounts.yihou.v2.tsv \
  chain_cap2.yihou.sh chain_cap2l.yihou.sh chain_cap2ls.yihou.sh cont_bench.yihou.sh warmup_pyspy.yihou.sh \
  run.v3.yihou.sh run.v3s.yihou.sh pipeline.v3.yihou.sh pipeline.v3s.yihou.sh tokdiff.yihou.sh run136_ofg4h3.yihou.sh env.ofg4h3-136.yihou.sh; do c $H/$f $P/scripts/harness/; done
for f in $H/config.yihou.mc4340n*.sh $H/env.ofg2h*-*.yihou.sh $H/env.ofg3h*-*.yihou.sh $H/env.ofg4h4-*.yihou.sh $H/env.ofg4h5-*.yihou.sh; do c $f $P/scripts/harness/; done
# ---- session tools
for f in tools_hits.yihou.py collect_env.yihou.sh prep_node.yihou.sh results_csv_burst.yihou.py mk_streamcopies.yihou.py; do c $N/$f $P/scripts/session/; done
c $N/shimbin/ssh $P/scripts/session/shimbin/
c $CAP/tools/metrics_sampler.yihou.py $P/scripts/tools/
c $N/packup_copy_burst.yihou.sh $P/scripts/tools/
# ---- patch (unchanged since the first offload kit; copied for self-containment)
c $N/offload/offloader.py.orig $P/patches/; c $N/offload/offloader.yihou-attrs.py $P/patches/; c $N/offload/build_offpatch.yihou.py $P/patches/
diff -u $N/offload/offloader.py.orig $N/offload/offloader.yihou-attrs.py > $P/patches/0001-offloader-carry-is_shuffled.diff
# ---- env: collect_env of earlier nodes + prep logs of every burst node used
for f in $N/env/collect_env.*.txt; do c $f $P/env/; done
for k in 001 006 007 008 010 011 012 019 025 027 031 033 062 179 200 219 228 232 241 250 274; do [ -e $N/logs/prep.$k.yihou.log ] && cp -p $N/logs/prep.$k.yihou.log $P/env/; done
# ---- dryruns + tokdiffs
for r in $SCORED $LOST; do for f in $N/dryrun.$r-*.yihou.txt; do [ -e $f ] && cp -p $f $P/results/dryruns/; done; done
c $N/dryrun.ofg2h3-v3026.yihou.txt $P/results/dryruns/
td(){ ( cd $H && bash tokdiff.yihou.sh $N/dryrun.$1.yihou.txt $N/dryrun.$2.yihou.txt ) > $P/results/dryruns/tokdiff.$1-vs-$2.txt; }
td ofg2h3-v3026 ofg2h5-v3008; td ofg2h5-v3008 ofg2h21-v3006; td ofg2h21-v3006 ofg2h22-v3006; td ofg2h22-v3006 ofg4h4-v3006
td ofg4h4-v3006 ofg4h5-v3006; td ofg3h2-v3010 ofg3h14-v3006; td ofg3h13-250 ofg3h14-v3006
# ---- rounds (status / chain / pipeline / bench console logs), scored and lost
for r in $SCORED $LOST; do
  d=$P/results/rounds/$r; mkdir -p $d
  for f in status.tsv audit.txt census.txt smoke-response.json spec_accept.after-bench.txt workers.json engine-prefill.log engine-decode.a1.log stop.log; do
    [ -e $R/$r/$f ] && cp -p $R/$r/$f $d/; done
  for f in $R/$r.chain.log $R/$r.pipeline.log $R/$r.cont.log $R/$r.pyspy.log $R/$r.stop.log $R/$r.stop2.log; do [ -e $f ] && cp -p $f $d/; done
  for f in $R/$r/agentx-*.log; do [ -e $f ] && gzip -c $f > $d/$(basename $f).gz; done
done
# ---- py-spy dumps (warm-up watcher), any round that produced one
for r in $SCORED $LOST; do [ -d $R/$r/pyspy ] && [ -n "$(ls -A $R/$r/pyspy 2>/dev/null)" ] && { mkdir -p $P/results/pyspy/$r; cp -p $R/$r/pyspy/* $P/results/pyspy/$r/ 2>/dev/null; }; done
# ---- bench dirs (complete or partial); profile_export.jsonl / server_metrics_export.json are NOT copied (>4 MB) - paths in README
for s in $R/ofg2h5/agentx-c56-A* $R/ofg3h2/agentx-c56-A* $R/ofg3h2/agentx-c64-A3 $R/ofg2h17/agentx-c56-A* $R/ofg2h21/agentx-c56-A* \
         $R/ofg3h14/agentx-c64-A* $R/ofg2h22/agentx-c64-A* $R/ofg4h4/agentx-c64-A* $R/ofg4h5/agentx-c64-A* $R/ofg2h9/agentx-c56-A1.preempt1 \
         $R/ofg2h19/agentx-c56-A1 $R/ofg2h20/agentx-c56-A1; do
  [ -d $s ] || continue; r=$(basename $(dirname $s)); d=$P/results/$r-${s##*agentx-}; mkdir -p $d
  for f in agentx_conc*.json benchmark_command.txt client.container-exit-code client.shell-exit-code metrics_plots.png runtime.env runtime.node.env \
           workload_distribution_plots.png workload_distribution_summary.txt; do for g in $s/$f; do [ -e $g ] && cp -p $g $d/; done; done
  [ -d $s/service ] && { mkdir -p $d/service; cp -p $s/service/hardware.json $s/service/workers.json $d/service/ 2>/dev/null
    for g in $s/service/*server-info.json; do [ -e $g ] && gzip -c $g > $d/service/$(basename $g).gz; done; }
  for f in benchmark.log runner.log; do [ -e $s/$f ] && gzip -c $s/$f > $d/$f.gz; done
  a=$s/aiperf_artifacts; if [ -d $a ]; then mkdir -p $d/aiperf
    for f in profile_export_aiperf.csv profile_export_aiperf.json profile_export_console.txt; do [ -e $a/$f ] && cp -p $a/$f $d/aiperf/; done
    [ -e $a/server_metrics_export.csv ] && { gz $a/server_metrics_export.csv $d/aiperf/server_metrics_export.csv.gz || true; }; fi
done
# ---- unofficial rescoring of ofg4h4 (NUL line dropped); copy the JSON outputs only (inputs are >4 MB, paths in README)
for t in A1 A2; do u=$N/unofficial/ofg4h4-$t-nulline-dropped; d=$P/results/unofficial/ofg4h4-$t-nulline-dropped; mkdir -p $d; c $u/agentx_conc64.json $d/; done
# ---- /metrics samplers
for r in $SCORED; do for f in $N/runs/$r/metrics.A*.tsv; do [ -e $f ] && { gz $f $P/results/metrics/$r.$(basename $f).gz || echo "metrics too big: $f"; }; done; done
# ---- server logs (gzip < 4 MB, else excerpt), scored rounds + the two silent-loss rounds
for r in $SCORED ofg2h19 ofg2h20; do mkdir -p $P/logs/server/$r
  for f in $R/$r/server-logs/*.log; do [ -e $f ] || continue; b=$(basename $f)
    gz $f $P/logs/server/$r/$b.gz 2>/dev/null || excerpt $f $P/logs/server/$r/$b.excerpt.txt.gz; done; done
# ---- results.csv + ledgers + spec
python3 -B $N/results_csv_burst.yihou.py > $P/results/results.csv
c $L/debug/poll.md $P/analysis/ledger/; c $L/debug/working_process.md $P/analysis/ledger/; c $L/mission.md $P/spec/
echo COPY-DONE
