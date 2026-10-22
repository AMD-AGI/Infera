#!/usr/bin/env bash
# Copy-only pack-up builder for glm52.v0521-prefill-mem-hicache-policy.packup_20261014 (run on spur006).
# Never deletes or moves anything; sources are read-only. Writes only inside $P.
set -uo pipefail
P=/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-prefill-mem-hicache-policy.packup_20261014
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
L=/home/yihou/dev/git/infera.glm52.view/glm52.v0521-cap-c44c48-23k_20261012
LIM=4000000
c(){ [ -e "$1" ] && cp -p "$1" "$2" || echo "MISSING $1"; }
gz(){ # gz <src> <dst.gz>  (only if result < LIM)
  [ -e "$1" ] || { echo "MISSING $1"; return; }
  local n; n=$(gzip -c "$1" | wc -c)
  if [ "$n" -lt "$LIM" ]; then gzip -c "$1" > "$2"; else echo "TOOBIG-GZ $1 ($n)"; return 1; fi; }
excerpt(){ # excerpt <src> <dst.excerpt.txt.gz>
  { echo "### EXCERPT of $1 ($(stat -c %s "$1") bytes, mtime $(stat -c %y "$1")): head 2 MB / key lines (<=1 MB) / tail 1 MB"
    echo "### ---- HEAD ----"; head -c 2000000 "$1"
    echo; echo "### ---- KEY LINES (grep -nE 'Traceback|Error|error|Exception|OOM|out of memory|NODE|abort|Abort|max_total_num_tokens|KV Cache is allocated|Memory pool|avail mem|Load weight end|retract|flush') ----"
    grep -anE 'Traceback|Error|error|Exception|OOM|out of memory|abort|Abort|max_total_num_tokens|KV Cache is allocated|Memory pool|avail mem|Load weight end|retract|flush' "$1" | head -c 1000000
    echo; echo "### ---- TAIL ----"; tail -c 1000000 "$1"; } | gzip -c > "$2"; }
mkdir -p $P/scripts/harness $P/scripts/session/shimbin $P/scripts/tools $P/patches $P/results/rounds $P/results/metrics \
         $P/results/dryruns $P/results/vram $P/env $P/logs/server $P/analysis/ledger $P/spec
# ---- harness (verbatim)
cp -pr $H/bench-harness $H/tools $H/mounts-v0521 $P/scripts/harness/
for f in config.yihou.v0521-base.sh config.yihou.E2-target.sh config.yihou.mc4340n.sh config.yihou.mc4340n2.sh config.yihou.mc4340n3.sh \
  config.yihou.mc4340n4.sh config.yihou.pcomplete.sh config.yihou.pcomplete2.sh \
  mounts.yihou.v2-regbase-attndp.tsv mounts.yihou.pcomplete-attndp.tsv mounts.yihou.v2-regbase.tsv mounts.yihou.v2.tsv \
  env.pf90-089.yihou.sh env.pf92-085.yihou.sh env.pf92-206.yihou.sh env.pf88-274.yihou.sh env.pf88b-085.yihou.sh env.wts48-136.yihou.sh \
  env.pc64-089.yihou.sh env.cx90-089.yihou.sh env.pf93-136.yihou.sh env.pf94-136.yihou.sh env.q1-137.yihou.sh env.pc48b-274.yihou.sh \
  topology.yihou.089.tsv topology.yihou.085.tsv topology.yihou.274.tsv topology.yihou.136.tsv topology.yihou.206.tsv \
  chain_cap2.yihou.sh chain_cap2l.yihou.sh cont_bench.yihou.sh cont_benchl.yihou.sh cont_benchl2.yihou.sh stop_then_chain.yihou.sh \
  after_a2_then.yihou.sh wait_then.yihou.sh after_wts64_pf94.yihou.sh chain_cx.yihou.sh run.v3.yihou.sh pipeline.v3.yihou.sh \
  tokdiff.yihou.sh dryrun.q1-137.yihou.txt; do c $H/$f $P/scripts/harness/; done
# ---- session tools
for f in tools_hits.yihou.py tools_summ.yihou.py tools_cmp.yihou.py collect_env.yihou.sh vram_peak.yihou.sh correctness_tests.yihou.py \
  env.wts48-NODE.template env.wts48dry.yihou.sh summary.v0521cap2.yihou.md; do c $N/$f $P/scripts/session/; done
c $N/shimbin/ssh $P/scripts/session/shimbin/
c $CAP/tools/metrics_sampler.yihou.py $P/scripts/tools/
# ---- patches (pc64 attndp hook)
c $N/mounts/pcomplete/pd_disaggregation_hook.py.orig $P/patches/
c $N/mounts/pcomplete/pd_disaggregation_hook.yihou-attndp.py $P/patches/
diff -u $N/mounts/pcomplete/pd_disaggregation_hook.py.orig $N/mounts/pcomplete/pd_disaggregation_hook.yihou-attndp.py > $P/patches/0001-pcomplete-attndp-validation.diff
# ---- env captures + dryruns
for f in $N/env/collect_env.*.20261014.txt; do c $f $P/env/; done
for f in dryrun.pf90-089 dryrun.pf92-206 dryrun.pf88-274 dryrun.pf88b-085 dryrun.wts48dry dryrun.wts48-136 dryrun.pc64-089 dryrun.cx90-089 dryrun.pf94-136; do
  c $N/$f.yihou.txt $P/results/dryruns/; done
# ---- rounds: status, driver logs, audit, census
for r in pf90 pf92 pf88 wts48 pc64 cx90; do
  d=$P/results/rounds/$r; mkdir -p $d
  for f in status.tsv audit.txt census.txt smoke-response.json spec_accept.after-bench.txt workers.json engine-prefill.log engine-decode.a1.log; do
    [ -e $R/$r/$f ] && cp -p $R/$r/$f $d/; done
  [ -d $R/$r/audit ] && { mkdir -p $d/audit; for f in $R/$r/audit/*; do gzip -c $f > $d/audit/$(basename $f).gz; done; }
  for f in $R/$r.chain.log $R/$r.pipeline.log $R/$r.handover.log $R/$r.cont.log; do [ -e $f ] && cp -p $f $d/; done
  for f in $R/$r/agentx-*.log; do [ -e $f ] && gzip -c $f > $d/$(basename $f).gz; done
done
for f in $R/pf88b.handover.log $R/pf94.handover.log $R/pf93.handover.log; do [ -e $f ] && cp -p $f $P/results/rounds/; done
# ---- per-bench result dirs
for b in pf90/agentx-c48-A1 pf90/agentx-c48-A2 pf92/agentx-c48-A1 pf92/agentx-c48-A2 pf88/agentx-c48-A1 wts48/agentx-c48-A1 wts48/agentx-c48-A2 \
         wts48/agentx-c64-A2 pc64/agentx-c64-A1 pc64/agentx-c64-A2; do
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
# ---- /metrics samplers + VRAM
for r in pf90 pf92 pf88 wts48 pc64; do for f in $N/runs/$r/metrics.A*.tsv; do [ -e $f ] && { gz $f $P/results/metrics/$r.$(basename $f).gz || echo "metrics too big: $f"; }; done; done
c $N/runs/pc64/vram.089.tsv $P/results/vram/
# ---- server logs (gzip < 4 MB, else excerpt)
for r in pf90 pf92 pf88 wts48 pc64; do mkdir -p $P/logs/server/$r
  for f in $R/$r/server-logs/*.log; do b=$(basename $f)
    gz $f $P/logs/server/$r/$b.gz || excerpt $f $P/logs/server/$r/$b.excerpt.txt.gz; done; done
# ---- ledgers + spec sources
c $L/debug/poll.md $P/analysis/ledger/; c $L/debug/working_process.md $P/analysis/ledger/; c $L/mission.md $P/spec/; c $L/PLAN.md $P/spec/
echo COPY-DONE
