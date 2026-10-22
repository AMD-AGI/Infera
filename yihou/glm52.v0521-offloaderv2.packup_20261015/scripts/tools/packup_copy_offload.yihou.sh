#!/usr/bin/env bash
# Copy-only pack-up builder for glm52.v0521-offloaderv2.packup_20261015 (run on a spur login host).
# Never deletes or moves anything; sources are read-only. Writes only inside $P.
set -uo pipefail
P=/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-offloaderv2.packup_20261015
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
L=/home/yihou/dev/git/infera.glm52.view/glm52.v0521-cap-c44c48-23k_20261012
LIM=4000000
c(){ [ -e "$1" ] && cp -p "$1" "$2" || echo "MISSING $1"; }
gz(){ [ -e "$1" ] || { echo "MISSING $1"; return; }
  local n; n=$(gzip -c "$1" | wc -c)
  if [ "$n" -lt "$LIM" ]; then gzip -c "$1" > "$2"; else echo "TOOBIG-GZ $1 ($n)"; return 1; fi; }
excerpt(){ { echo "### EXCERPT of $1 ($(stat -c %s "$1") bytes, mtime $(stat -c %y "$1")): head 2 MB / key lines (<=1 MB) / tail 1 MB"
    echo "### ---- HEAD ----"; head -c 2000000 "$1"
    echo; echo "### ---- KEY LINES ----"
    grep -anE 'Traceback|Error|error|Exception|OOM|out of memory|abort|Abort|max_total_num_tokens|KV Cache is allocated|Memory pool|avail mem|Load weight end|offloader|warmup|Engine startup|retract|flush' "$1" | head -c 1000000
    echo; echo "### ---- TAIL ----"; tail -c 1000000 "$1"; } | gzip -c > "$2"; }
mkdir -p $P/scripts/harness $P/scripts/session/shimbin $P/scripts/tools $P/patches $P/results/rounds $P/results/metrics \
         $P/results/dryruns $P/results/correctness $P/results/h2d $P/env $P/logs/server $P/analysis/ledger $P/analysis/research $P/spec
# ---- harness (verbatim)
cp -pr $H/bench-harness $H/tools $H/mounts-v0521 $P/scripts/harness/
for f in config.yihou.v0521-base.sh config.yihou.E2-target.sh config.yihou.mc4340n.sh config.yihou.mc4340n4.sh config.yihou.mc4340n5.sh \
  config.yihou.offload.sh mounts.yihou.offload-attrs.tsv mounts.yihou.v2-regbase.tsv mounts.yihou.v2-regbase-attndp.tsv mounts.yihou.v2.tsv \
  env.pf90-089.yihou.sh env.cx90-089.yihou.sh env.ofs0-136.yihou.sh env.ofs0c-136.yihou.sh env.ofs0-089.yihou.sh env.ofs0c-089.yihou.sh \
  env.ofg4-252.yihou.sh env.ofg4c-252.yihou.sh env.ofg3-266.yihou.sh env.ofg3c-266.yihou.sh env.ofg2-288.yihou.sh env.ofg2c-288.yihou.sh \
  topology.yihou.089.tsv topology.yihou.136.tsv topology.yihou.252.tsv topology.yihou.266.tsv topology.yihou.288.tsv \
  chain_cap2.yihou.sh chain_cap2l.yihou.sh chain_cx.yihou.sh chain_cxl.yihou.sh after_wts64_s0.yihou.sh after_tw90_s0.yihou.sh ofs_seq.yihou.sh \
  run.v3.yihou.sh pipeline.v3.yihou.sh tokdiff.yihou.sh; do c $H/$f $P/scripts/harness/; done
# ---- session tools
for f in tools_hits.yihou.py tools_summ.yihou.py tools_cmp.yihou.py collect_env.yihou.sh vram_peak.yihou.sh correctness_tests.yihou.py \
  h2d_bench.yihou.py summary.v0521cap2.yihou.md; do c $N/$f $P/scripts/session/; done
c $N/shimbin/ssh $P/scripts/session/shimbin/
c $CAP/tools/metrics_sampler.yihou.py $P/scripts/tools/
c $N/packup_copy_offload.yihou.sh $P/scripts/tools/
# ---- patch (YIHOU-OFFLOAD-ATTRS)
c $N/offload/offloader.py.orig $P/patches/
c $N/offload/offloader.yihou-attrs.py $P/patches/
c $N/offload/build_offpatch.yihou.py $P/patches/
diff -u $N/offload/offloader.py.orig $N/offload/offloader.yihou-attrs.py > $P/patches/0001-offloader-carry-is_shuffled.diff
# ---- H2D bandwidth bench
c $N/offload/h2d_bench.089.json $P/results/h2d/
# ---- env captures + dryruns
for f in $N/env/collect_env.089.20261014.txt $N/env/collect_env.136.20261014.txt $N/env/collect_env.252.20261015.txt $N/env/collect_env.288.20261015.txt; do c $f $P/env/; done
for f in dryrun.pf90-089 dryrun.cx90-089 dryrun.ofs0-136 dryrun.ofs0c-136 dryrun.ofs0-089 dryrun.ofs0c-089 dryrun.ofg4-252 dryrun.ofg4c-252 \
         dryrun.ofg3-266 dryrun.ofg3c-266 dryrun.ofg2-288 dryrun.ofg2c-288; do c $N/$f.yihou.txt $P/results/dryruns/; done
( cd $H && bash tokdiff.yihou.sh $N/dryrun.pf90-089.yihou.txt $N/dryrun.ofs0-089.yihou.txt ) > $P/results/dryruns/tokdiff.pf90-vs-ofs0.txt
( cd $H && bash tokdiff.yihou.sh $N/dryrun.cx90-089.yihou.txt $N/dryrun.ofs0c-089.yihou.txt ) > $P/results/dryruns/tokdiff.cx90-vs-ofs0c.txt
for r in ofg4-252 ofg3-266 ofg2-288; do ( cd $H && bash tokdiff.yihou.sh $N/dryrun.ofs0-089.yihou.txt $N/dryrun.$r.yihou.txt ) > $P/results/dryruns/tokdiff.ofs0-vs-$r.txt; done
for r in ofg4c-252 ofg3c-266 ofg2c-288; do ( cd $H && bash tokdiff.yihou.sh $N/dryrun.ofs0c-089.yihou.txt $N/dryrun.$r.yihou.txt ) > $P/results/dryruns/tokdiff.ofs0c-vs-$r.txt; done
# ---- correctness results
for r in cx90 ofs0c ofg4c ofg3c ofg2c; do d=$P/results/correctness/$r; mkdir -p $d
  for f in results.jsonl correctness.log; do [ -e $N/runs/$r/$f ] && { gz $N/runs/$r/$f $d/$f.gz || true; }; done; done
# ---- rounds: status, driver logs, audit, census
for r in ofs0c ofs0 ofg4c ofg3c ofg2c ofg3 ofg2 ofg4c.hang1; do
  d=$P/results/rounds/$r; mkdir -p $d
  for f in status.tsv audit.txt census.txt smoke-response.json spec_accept.after-bench.txt workers.json engine-prefill.log engine-decode.a1.log stop.log stop.release266.log; do
    [ -e $R/$r/$f ] && cp -p $R/$r/$f $d/; done
  [ -d $R/$r/audit ] && { mkdir -p $d/audit; for f in $R/$r/audit/*; do gzip -c $f > $d/audit/$(basename $f).gz; done; }
  for f in $R/$r.chain.log $R/$r.pipeline.log $R/$r.seq.log $R/$r.handover089.log; do [ -e $f ] && cp -p $f $d/; done
  for f in $R/$r/agentx-*.log; do [ -e $f ] && gzip -c $f > $d/$(basename $f).gz; done
done
for f in $R/ofg4c.hang1/ofg4c.chain.log $R/ofg4c.hang1/ofg4.seq.log; do c $f $P/results/rounds/ofg4c.hang1/; done
for f in $R/ofs0.handover089.log $R/ofg4.seq.log $R/ofg3.seq.log $R/ofg2.seq.log; do [ -e $f ] && cp -p $f $P/results/rounds/; done
# ---- per-bench result dirs
for b in ofs0/agentx-c48-A1 ofs0/agentx-c48-A2 pf90/agentx-c48-A1 pf90/agentx-c48-A2; do
  s=$R/$b; d=$P/results/${b%%/*}-${b##*agentx-}; mkdir -p $d
  for f in agentx_conc*.json benchmark_command.txt client.container-exit-code client.shell-exit-code metrics_plots.png runtime.env runtime.node.env \
           workload_distribution_plots.png workload_distribution_summary.txt; do for g in $s/$f; do [ -e $g ] && cp -p $g $d/; done; done
  [ -d $s/service ] && { mkdir -p $d/service; cp -p $s/service/hardware.json $s/service/workers.json $d/service/ 2>/dev/null
    for g in $s/service/*server-info.json; do [ -e $g ] && gzip -c $g > $d/service/$(basename $g).gz; done; }
  for f in benchmark.log runner.log; do [ -e $s/$f ] && gzip -c $s/$f > $d/$f.gz; done
  a=$s/aiperf_artifacts; if [ -d $a ]; then mkdir -p $d/aiperf
    for f in profile_export_aiperf.csv profile_export_aiperf.json profile_export_console.txt; do [ -e $a/$f ] && cp -p $a/$f $d/aiperf/; done
    [ -e $a/server_metrics_export.csv ] && gzip -c $a/server_metrics_export.csv > $d/aiperf/server_metrics_export.csv.gz; fi
done
# ---- /metrics samplers
for r in ofs0; do for f in $N/runs/$r/metrics.A*.tsv; do [ -e $f ] && { gz $f $P/results/metrics/$r.$(basename $f).gz || echo "metrics too big: $f"; }; done; done
# ---- server logs (gzip < 4 MB, else excerpt); live rounds are snapshots
for r in ofs0c ofs0 ofg4c ofg3c ofg2c ofg3 ofg2 ofg4c.hang1; do mkdir -p $P/logs/server/$r
  for f in $R/$r/server-logs/*.log; do [ -e $f ] || continue; b=$(basename $f)
    gz $f $P/logs/server/$r/$b.gz || excerpt $f $P/logs/server/$r/$b.excerpt.txt.gz; done; done
# ---- ledgers + spec
c $L/debug/poll.md $P/analysis/ledger/; c $L/debug/working_process.md $P/analysis/ledger/; c $L/mission.md $P/spec/
echo COPY-DONE
