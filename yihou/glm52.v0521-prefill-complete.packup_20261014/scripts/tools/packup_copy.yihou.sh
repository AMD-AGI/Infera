#!/usr/bin/env bash
# Packup C (prefill_complete) copy step, run on login host spur006: COPY ONLY (cp -p), no rm/mv; refuses to overwrite.
set -euo pipefail
P=/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-prefill-complete.packup_20261014
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
L=/home/yihou/dev/git/infera.glm52.view/glm52.v0521-cap-c44c48-23k_20261012
c(){ local src=$1 dst=$2; mkdir -p "$(dirname "$dst")"; [[ -e "$dst" ]] && { echo "exists, skip: $dst"; return 0; }; cp -p "$src" "$dst"; }
cz(){ local src=$1 dst=$2; mkdir -p "$(dirname "$dst")"; [[ -e "$dst" ]] && { echo "exists, skip: $dst"; return 0; }; gzip -c "$src" > "$dst"; touch -r "$src" "$dst"; }
# --- harness (verbatim)
S=$P/scripts/harness
for f in $H/bench-harness/*; do [[ -f $f ]] && c $f $S/bench-harness/$(basename $f); done
for d in $H/bench-harness/*/; do [[ -d $d ]] || continue; for f in $d*; do [[ -f $f ]] && c $f $S/bench-harness/$(basename $d)/$(basename $f); done; done
for f in config.yihou.pcomplete.sh config.yihou.mc4340n.sh config.yihou.mc4340n2.sh config.yihou.mc4340n3.sh config.yihou.mc4340n4.sh \
         config.yihou.E2-target.sh config.yihou.v0521-base.sh config.yihou.E1-relaxed.sh \
         mounts.yihou.pcomplete-attndp.tsv mounts.yihou.v2-regbase.tsv mounts.yihou.v2.tsv \
         env.pc48-274.yihou.sh env.pc48-206.yihou.sh env.pc48b-274.yihou.sh env.pc48e-274.yihou.sh env.q1-137.yihou.sh \
         chain_cap2.yihou.sh chain_cap2l.yihou.sh stop_then_chain.yihou.sh wait_then.yihou.sh after_x8p_pc48.yihou.sh cont_bench.yihou.sh \
         run.v3.yihou.sh pipeline.v3.yihou.sh tokdiff.yihou.sh dryrun.q1-137.yihou.txt; do
  [[ -f $H/$f ]] && c $H/$f $S/$f || echo "MISSING $H/$f"; done
c $H/tools/audit_live.v3.yihou.py $S/tools/audit_live.v3.yihou.py
c $H/tools/pidcensus.yihou.sh $S/tools/pidcensus.yihou.sh
for f in kv_events_publisher.yihou-noblock.py forward_mla_rocm.yihou-noabsorbcfg.py conn.yihou-regbase.py; do c $H/mounts-v0521/src/$f $S/mounts-v0521/src/$f; done
# --- session workspace files
c $N/shimbin/ssh $P/scripts/session/shimbin/ssh
for f in env.pc48-NODE.template env.pc48dry.yihou.sh tools_cmp.yihou.py tools_summ.yihou.py collect_env.yihou.sh summary.v0521cap2.yihou.md; do c $N/$f $P/scripts/session/$f; done
c $CAP/tools/metrics_sampler.yihou.py $P/scripts/tools/metrics_sampler.yihou.py
# --- patch
c $N/mounts/pcomplete/pd_disaggregation_hook.py.orig $P/patches/pd_disaggregation_hook.py.orig
c $N/mounts/pcomplete/pd_disaggregation_hook.yihou-attndp.py $P/patches/pd_disaggregation_hook.yihou-attndp.py
[[ -e $P/patches/0001-pcomplete-attndp-validation.diff ]] || ( cd $N/mounts/pcomplete && diff -u pd_disaggregation_hook.py.orig pd_disaggregation_hook.yihou-attndp.py > $P/patches/0001-pcomplete-attndp-validation.diff || true )
# --- dryruns
for f in dryrun.pc48dry.yihou.txt dryrun.pc48-274.yihou.txt dryrun.pc48-206.yihou.txt dryrun.pc48b-274.yihou.txt dryrun.pc48e-274.yihou.txt; do c $N/$f $P/results/dryruns/$f; done
# --- environment captures
for f in $N/env/collect_env.*.txt; do c $f $P/results/env/$(basename $f); done
for f in $N/logs/pc48.stop.yihou.log $N/logs/imgload.crsuse2-m2m-274.yihou.log $N/logs/seed.274.yihou.log; do c $f $P/results/env/$(basename $f); done
# --- round files
for r in pc48 pc48b pc48e; do
  for f in $R/$r.chain.log $R/$r.pipeline.log $R/$r.handover.log; do [[ -f $f ]] && c $f $P/results/rounds/$r/$(basename $f); done
  for f in status.tsv audit.txt census.txt smoke-response.json spec_accept.after-bench.txt workers.json engine-prefill.log engine-decode.a1.log; do
    [[ -f $R/$r/$f ]] && c $R/$r/$f $P/results/rounds/$r/$f; done
  if [[ -d $R/$r/audit ]]; then for f in $R/$r/audit/*.json; do cz $f $P/results/rounds/$r/audit/$(basename $f).gz; done; fi
done
c $R/pc48/server-logs/prefill-0.log $P/results/rounds/pc48/prefill-0.startup-ValueError.log
# --- per-bench artifacts
for r in pc48b pc48e; do for a in A1 A2; do
  B=$R/$r/agentx-c48-$a; D=$P/results/$r-$a-C48
  for f in agentx_conc48.json benchmark_command.txt runtime.env workload_distribution_summary.txt metrics_plots.png workload_distribution_plots.png client.container-exit-code client.shell-exit-code; do c $B/$f $D/$f; done
  for f in profile_export_console.txt profile_export_aiperf.csv profile_export_aiperf.json; do c $B/aiperf_artifacts/$f $D/aiperf/$f; done
  for f in workers.json hardware.json; do c $B/service/$f $D/service/$f; done
  cz $B/benchmark.log $D/benchmark.log.gz
  cz $B/aiperf_artifacts/server_metrics_export.csv $D/aiperf/server_metrics_export.csv.gz
  cz $N/runs/$r/metrics.$a.tsv $P/results/metrics/$r.metrics.$a.tsv.gz
done; done
# --- ledgers + spec
c $L/mission.md $P/spec/mission.md
c $L/debug/working_process.md $P/analysis/ledger/working_process.md
c $L/debug/poll.md $P/analysis/ledger/poll.md
echo COPY DONE
