#!/usr/bin/env bash
# gather (task g4tune packup 2026-10-22): copy the small evidence files of the 10-21 rounds into a staging dir
# under the g4tune workspace (a yihou dir) and build results/summary.tsv. Read-only on the sources.
# Usage (on a node with /shared_nfs): bash gather.yihou.sh <stage-dir>
set -uo pipefail
S=$1
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; N=/shared_nfs/yihou_glm52/v0521cap2_20261013
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness; W=/shared_nfs/yihou_glm52/g4tune_20261019
D=/shared_nfs/yihou_glm52/idxshare_dbg_20261020
[[ "$S" == *yihou* && ! -e "$S" ]] || { echo "stage must be new and contain yihou"; exit 1; }
mkdir -p "$S"/results/rounds "$S"/results/xnode "$S"/scripts/harness "$S"/scripts/tools "$S"/scripts/mounts-src \
  "$S"/scripts/envs "$S"/results/dryruns "$S"/results/launch "$S"/patches
ROUNDS="idxf1a idxf1b idxf1c g4h1 g4k1 g4p1 g4c1 g4c2 g4c3 g4x1 g4x2 g3x1"
for r in $ROUNDS; do
  o="$S/results/rounds/$r"; mkdir -p "$o"
  for f in status.tsv census.txt audit.txt workers.json spec_accept.after-bench.txt; do [ -f "$R/$r/$f" ] && cp -p "$R/$r/$f" "$o/"; done
  [ -d "$R/$r/audit" ] && { mkdir -p "$o/audit"; cp -p "$R/$r"/audit/*server-info.json "$o/audit/" 2>/dev/null; }
  for f in chain.log pipeline.log conc.log diskwatch.log stop.log pyspy.log; do [ -f "$R/$r.$f" ] && cp -p "$R/$r.$f" "$o/$r.$f"; done
  for b in "$R/$r"/agentx-c*-A*; do
    [ -d "$b" ] || continue; bn=$(basename "$b"); mkdir -p "$o/$bn"
    cp -p "$b"/agentx_conc*.json "$o/$bn/" 2>/dev/null
    f=$(find "$b" -name server_metrics_export.csv | head -n1); [ -n "$f" ] && gzip -c "$f" > "$o/$bn/server_metrics_export.csv.gz"
    [ -f "$b.log" ] && tail -n 30 "$b.log" > "$o/$bn.log.tail30"
  done
  ls "$R/$r"/hipxfer.*.log >/dev/null 2>&1 && { mkdir -p "$o/hipxfer"; for f in "$R/$r"/hipxfer.*.log; do gzip -c "$f" > "$o/hipxfer/$(basename "$f").gz"; done; }
done
# summary.tsv: one row per bench
bash "$(dirname "$0")/summary.yihou.sh" "$S"
# new harness pieces (10-21) + the env/dryrun/launch files of these rounds
for f in config.yihou.xnode.sh config.yihou.xfertrace.sh config.yihou.idlefix.sh config.yihou.mc4340n19.sh run.v3x.yihou.sh \
  pipeline.v3x.yihou.sh chain_x.yihou.sh conc_after_x.yihou.sh after_x_then.yihou.sh stop_then_chain.yihou.sh \
  after_a2_then.yihou.sh chain_cap2.yihou.sh run.v3.yihou.sh pipeline.v3.yihou.sh run.v2.yihou.sh \
  mounts.yihou.v2-xferfix2-gtt-idlefix.tsv mounts.yihou.offload-attrs-hipxfer.tsv topology.yihou.135x136.tsv; do
  [ -f "$H/$f" ] && cp -p "$H/$f" "$S/scripts/harness/" || echo "missing harness $f"; done
cp -p "$H/tools/audit_live.v3x.yihou.py" "$H/tools/xfer_summary.yihou.py" "$S/scripts/tools/"
cp -p "$N/hist.yihou.py" "$S/scripts/tools/"
cp -p "$H/mounts-v0521/src/hipxfer_trace.yihou.c" "$H/mounts-v0521/src/eager_runner.yihou-idlefix.py" "$S/scripts/mounts-src/"
cp -p "$W/xnode/rail_probe.yihou.sh" "$W/stop_round.yihou.sh" "$S/scripts/harness/"
cp -p "$D/env.idxf1b-268.yihou.sh" "$D/env.idxf1c-054.yihou.sh" "$D/env.idxf1a-097.yihou.sh" "$S/scripts/envs/"
cp -p "$D/dryrun.idxf1b-268.yihou.txt" "$D/dryrun.idxf1c-054.yihou.txt" "$S/results/dryruns/"
for e in g4h1-268 g4k1-054 g4p1-054 g4c1-268 g4c2-268 g4c3-054 g4x1-135x136 g4x2-135x136 g3x1-135x136; do
  cp -p "$W/env.$e.yihou.sh" "$S/scripts/envs/"; cp -p "$W/dryrun.$e.yihou.txt" "$S/results/dryruns/"; done
cp -p "$W"/launch.idxf1[bc].yihou.log "$W"/after.*.yihou.log "$W"/stc.*.yihou.log "$W"/stop.g4x1.yihou.log "$W"/stop.g3x1.yihou.log "$S/results/launch/" 2>/dev/null
# cross-node: rail probes (json + logs, no mooncakeperf dumps), TPOT analysis
for p in "$W"/xnode/preflight/ionic_*; do q="$S/results/xnode/preflight/$(basename "$p")"; mkdir -p "$q"; cp -p "$p"/*.json "$p"/rank-*.log "$q/"; done
cp -p "$W/xnode/tpot_analysis.20261021.md" "$S/results/xnode/"
cp -p "$N/logs/prep.135.yihou.log" "$N/logs/prep.136x.yihou.log" "$S/results/launch/"
du -sh "$S"; find "$S" -type f -size +4M -exec ls -la {} \;
echo GATHER_DONE
