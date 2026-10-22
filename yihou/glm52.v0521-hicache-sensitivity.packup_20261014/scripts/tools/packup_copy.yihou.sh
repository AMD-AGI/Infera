#!/usr/bin/env bash
# Pack-up copy step (COPY only, into the packup dir; no deletes anywhere).
set -uo pipefail
P=/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-hicache-sensitivity.packup_20261014
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
V=/home/yihou/dev/git/infera.glm52.view/glm52.v0521-cap-c44c48-23k_20261012
B=/home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd/results/yihou-hicache-ceiling-sweep/glm52.hicache-ceiling.packup_20260924/results
cd "$P" || exit 1

# ---- results: hc50 round files
mkdir -p results/hc50/round/audit
for f in status.tsv audit.txt census.txt workers.json smoke-response.json spec_accept.after-bench.txt engine-decode.a1.log engine-prefill.log; do
  cp -p "$R/hc50/$f" results/hc50/round/; done
cp -p "$R"/hc50/audit/*.json results/hc50/round/audit/
cp -p "$R/hc50.chain.log" "$R/hc50.pipeline.log" results/hc50/round/
# ---- results: hc50 A1/A2 bench dirs
for t in A1 A2; do
  s="$R/hc50/agentx-c40-$t"; d="results/hc50/$t-C40"; mkdir -p "$d/service" "$d/aiperf"
  cp -p "$s"/{agentx_conc40.json,benchmark_command.txt,runtime.env,runtime.node.env,workload_distribution_summary.txt,metrics_plots.png,workload_distribution_plots.png,client.container,client.container-exit-code,client.shell-exit-code} "$d/"
  cp -p "$s"/service/*.json "$d/service/"
  gzip -c "$s/benchmark.log" > "$d/benchmark.log.gz"
  gzip -c "$s/runner.log" > "$d/runner.log.gz"
  cp -p "$s"/aiperf_artifacts/{profile_export_aiperf.json,profile_export_aiperf.csv,profile_export_console.txt} "$d/aiperf/"
  gzip -c "$s/aiperf_artifacts/server_metrics_export.csv" > "$d/aiperf/server_metrics_export.csv.gz"
  gzip -c "$s/aiperf_artifacts/logs/aiperf.log" > "$d/aiperf/aiperf.log.gz"
done
gzip -c "$R/hc50/agentx-c40-A1.log" > results/hc50/round/agentx-c40-A1.log.gz
gzip -c "$R/hc50/agentx-c40-A2.log" > results/hc50/round/agentx-c40-A2.log.gz
# ---- /metrics TSVs (gz, only if < 4 MB)
mkdir -p results/metrics
for t in A1 A2; do gzip -c "$N/runs/hc50/metrics.$t.tsv" > "results/metrics/hc50.metrics.$t.tsv.gz"; done
# ---- reference e2g (102)
for t in A1:agentx-c40 A2:agentx-c40-A2; do
  tag=${t%%:*}; src=$R/e2g/${t#*:}; d=results/reference-e2g/$tag-C40; mkdir -p "$d"
  cp -p "$src"/{agentx_conc40.json,benchmark_command.txt,runtime.env,workload_distribution_summary.txt} "$d/" 2>/dev/null
done
cp -p "$R"/e2g/{status.tsv,audit.txt,census.txt} results/reference-e2g/
cp -p "$R/e2g.chain.log" "$R/e2g.pipeline.log" "$R/e2g.A2.log" results/reference-e2g/
# ---- other HiCache data points (JSON only)
mkdir -p results/reference-other
cp -p "$R/k1/agentx-c44/agentx_conc44.json" results/reference-other/k1-A1-C44.agentx_conc44.json
cp -p "$R/k1/agentx-c44-A2/agentx_conc44.json" results/reference-other/k1-A2-C44.agentx_conc44.json
cp -p "$R/e1c/agentx-c40-r2/agentx_conc40.json" results/reference-other/e1c-r2-C40.agentx_conc40.json
cp -p "$B/armA-reference-S154/armA-136-agentx_conc56.json" results/reference-other/8expert-armA-S154-C56.agentx_conc56.json
cp -p "$B/r001-hc274/agentx_conc56.json" results/reference-other/8expert-r001-hc274-C56.agentx_conc56.json
cp -p "$B/r001-hc450chunk512/agentx_conc56.json" results/reference-other/8expert-r001-hc450chunk512-C56.agentx_conc56.json
cp -p "$B/r001-hc274/params.txt" results/reference-other/8expert-r001-hc274.params.txt
cp -p "$B/r001-hc450chunk512/params.txt" results/reference-other/8expert-r001-hc450chunk512.params.txt
# ---- dryruns + diff
mkdir -p results/dryrun
cp -p "$N/dryrun.hc50-136.yihou.txt" "$H/dryrun.e2g-102.yihou.txt" results/dryrun/
diff "$H/dryrun.e2g-102.yihou.txt" "$N/dryrun.hc50-136.yihou.txt" > results/dryrun/diff.e2g-102_vs_hc50-136.txt
# ---- summary table of the session
cp -p "$N/summary.v0521cap2.yihou.md" results/

# ---- scripts (verbatim)
mkdir -p scripts/harness/tools scripts/harness/shimbin scripts/tools scripts/harness/bench-harness
for f in chain_cap2l.yihou.sh chain_cap2.yihou.sh stop_then_chain.yihou.sh wait_then.yihou.sh after_x8p_pc48.yihou.sh cont_bench.yihou.sh \
         run.v3.yihou.sh run.v2.yihou.sh pipeline.v3.yihou.sh tokdiff.yihou.sh stage_image.yihou.sh \
         env.hc50-136.yihou.sh env.e2g-102.yihou.sh env.pf88-274.yihou.sh env.pf90-089.yihou.sh env.pf92-085.yihou.sh env.pf92-206.yihou.sh \
         config.yihou.mc4340.sh config.yihou.mc4340n.sh config.yihou.mc4340n2.sh config.yihou.mc4340n3.sh config.yihou.mc4340n4.sh \
         config.yihou.E2-target.sh config.yihou.v0521-base.sh config.yihou.E1-relaxed.sh \
         mounts.yihou.v2-regbase.tsv topology.yihou.136.tsv topology.yihou.102.tsv RUNBOOK.v2.yihou.md; do
  cp -p "$H/$f" scripts/harness/; done
cp -p "$H/tools/audit_live.v3.yihou.py" scripts/harness/tools/
cp -a "$H/bench-harness/." scripts/harness/bench-harness/
cp -p "$N/shimbin/ssh" scripts/harness/shimbin/ssh
cp -p "$CAP/tools/metrics_sampler.yihou.py" "$N/tools_cmp.yihou.py" "$N/tools_summ.yihou.py" "$N/collect_env.yihou.sh" scripts/tools/

# ---- patches (runtime mounts listed by mounts.yihou.v2-regbase.tsv)
mkdir -p patches/mounts-v0521/src
for f in kv_events_publisher.yihou-noblock.py forward_mla_rocm.yihou-noabsorbcfg.py conn.yihou-regbase.py; do
  cp -p "$H/mounts-v0521/src/$f" patches/mounts-v0521/src/; done

# ---- env captures
mkdir -p env
cp -p "$N"/env/collect_env.136.20261014.txt env/

# ---- spec + ledgers
mkdir -p spec analysis/ledger
cp -p "$V/mission.md" "$V/PLAN.md" spec/
cp -p "$V/debug/working_process.md" "$V/debug/poll.md" analysis/ledger/

# ---- log index data
{ echo "## sizes at pack-up time ($(date -u +%FT%TZ))"
  ls -l "$R"/hc50/server-logs/ "$R"/hc50/agentx-c40-A*/aiperf_artifacts/ "$R"/e2g/server-logs/ "$N"/runs/hc50/
} > logs/.ls_raw.txt 2>&1
echo COPY DONE
