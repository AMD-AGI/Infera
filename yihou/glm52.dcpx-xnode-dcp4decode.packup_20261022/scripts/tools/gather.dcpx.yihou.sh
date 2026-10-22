#!/usr/bin/env bash
# gather (task dcpx packup 2026-10-22): copy the small evidence files of the dcpx/dcps rounds (DCP4 decode leg) plus the
# g3x1 reference into a NEW staging dir (must contain "yihou"), build results/summary.tsv. Read-only on the sources.
# Usage (on a host with /shared_nfs): bash gather.dcpx.yihou.sh <stage-dir>
set -uo pipefail
S=$1
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; N=/shared_nfs/yihou_glm52/v0521cap2_20261013
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness; W=/shared_nfs/yihou_glm52/g4tune_20261019
D=/shared_nfs/yihou_glm52/dcpx_20261022
[[ "$S" == *yihou* && ! -e "$S" ]] || { echo "stage must be new and contain yihou"; exit 1; }
mkdir -p "$S"/results/rounds "$S"/results/dryruns "$S"/results/rdma "$S"/scripts/harness "$S"/scripts/tools \
  "$S"/scripts/envs "$S"/imgbuild
ROUNDS="dcpx1 dcpx2 dcpx3 dcpx4 dcpx5 dcps1 dcpx6"
for r in $ROUNDS; do
  o="$S/results/rounds/$r"; mkdir -p "$o"
  for f in status.tsv census.txt audit.txt workers.json spec_accept.after-bench.txt; do [ -f "$R/$r/$f" ] && cp -p "$R/$r/$f" "$o/"; done
  [ -d "$R/$r/audit" ] && { mkdir -p "$o/audit"; cp -p "$R/$r"/audit/*server-info.json "$o/audit/" 2>/dev/null; }
  for f in chain.log pipeline.log diskwatch.log; do [ -f "$R/$r.$f" ] && cp -p "$R/$r.$f" "$o/$r.$f"; done
  for b in "$R/$r"/agentx-c*-A*; do
    [ -d "$b" ] || continue; bn=$(basename "$b"); mkdir -p "$o/$bn"
    cp -p "$b"/agentx_conc*.json "$o/$bn/" 2>/dev/null
    f=$(find "$b" -name server_metrics_export.csv | head -n1); [ -n "$f" ] && gzip -c "$f" > "$o/$bn/server_metrics_export.csv.gz"
    [ -f "$b.log" ] && tail -n 30 "$b.log" > "$o/$bn.log.tail30"
  done
  for m in "$N/runs/$r"/metrics.A*.tsv; do [ -f "$m" ] && gzip -c "$m" > "$o/$(basename "$m").gz"; done
  # decode crash excerpts (OOR rounds): last 200 lines of the decode log
  for l in "$R/$r"/server-logs/decode-0.a*.log; do [ -f "$l" ] && grep -a -q -E "OUT_OF_RESOURCES|Traceback" "$l" && \
    tail -n 200 "$l" > "$o/$(basename "$l").tail200"; done
done
[ -f "$N/runs/g3x1/metrics.A1.tsv" ] && { mkdir -p "$S/results/rounds/g3x1-ref"; gzip -c "$N/runs/g3x1/metrics.A1.tsv" > "$S/results/rounds/g3x1-ref/metrics.A1.tsv.gz"; }
bash "$(dirname "$0")/summary.dcpx.yihou.sh" "$S"
for f in config.yihou.dcpx.sh mounts.yihou.dcpx-overlay.sglang-93034eba91.tsv config.yihou.xnode.sh config.yihou.idlefix.sh \
  run.v3x.yihou.sh pipeline.v3x.yihou.sh chain_x.yihou.sh; do
  [ -f "$H/$f" ] && cp -p "$H/$f" "$S/scripts/harness/" || echo "missing harness $f"; done
cp -p "$H/tools/audit_live.v3x.yihou.py" "$N/hist.yihou.py" "$D/pdbal.yihou.py" "$D/decode_stats.dcp.yihou.py" "$D/rdmactr.yihou.sh" \
  "$(dirname "$0")/gather.dcpx.yihou.sh" "$(dirname "$0")/summary.dcpx.yihou.sh" "$S/scripts/tools/"
for e in dcpx1-135x136 dcpx2-135x136 dcpx3-135x136 dcpx4-135x136 dcpx5-135x136 dcps1-136 dcpx6-135x136; do
  cp -p "$W/env.$e.yihou.sh" "$S/scripts/envs/"; cp -p "$W/dryrun.$e.yihou.txt" "$S/results/dryruns/"; done
cp -p "$W"/stop.dcp*.yihou.log "$S/results/rounds/" 2>/dev/null
cp -p "$D"/rdmactr.dcpx*.yihou.log "$S/results/rdma/"
cp -p "$D"/imgbuild/* "$S/imgbuild/"
du -sh "$S"; find "$S" -type f -size +4M -exec ls -la {} \;
echo GATHER_DONE
