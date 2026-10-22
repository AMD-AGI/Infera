#!/usr/bin/env bash
# One-shot, restartable PD bring-up for dcp_pd (pdrun). Run on the LOGIN host, detached:
#   setsid nohup bash launch_pd.yihou.sh NK JOB VARIANT ATTEMPT > $PD/runs/launch.<round>.log 2>&1 < /dev/null &
# NK = node key with topology.yihou.<NK>.tsv; VARIANT = A0|A|B|Be|Bperf|Beperf (tools/mk_env.yihou.sh);
# ATTEMPT = r1, r2, ... -> round pd<VARIANT>-<ATTEMPT> (run.v2 refuses a reused container prefix, so a restart
# is: `bash run.v2.yihou.sh stop` for the old round (docker stop, never rm), then a new ATTEMPT).
# Steps: image present? (docker load from NFS save) -> image ID recorded -> env -> dryrun -> up -> census ->
# audit -> smoke -> pdcheck (log greps) -> T4a client. Extra overrides: export PD_EXTRA (e.g.
# DECODE_EXTRA_ARGS_ADD=--disable-cuda-graph) before launching. Bench is a separate, explicit step (PD_CONC, AGENTX_DURATION via YIHOU_OVERRIDES).
set -uo pipefail
NK=$1; JOB=$2; V=$3; ATT=$4
PD=/shared_nfs/yihou_glm52/dcp_pd_20261021/pd; H=$PD/harness
IMG=infera-sglang:v0522-1020-yihou-dcp-mc4340; TAR=$PD/images/infera-sglang_v0522-1020-yihou-dcp-mc4340.tar
WANT_ID=sha256:5b6565ef3e4e7dcca02423d509f842a74cf7416b18ed94edbf12e2be100f7376
SPUR=(spur exec "$JOB" bash -c)
node=$(awk -F'\t' '$1=="decode"{print $2}' "$H/topology.yihou.$NK.tsv"); [[ -n "$node" ]] || exit 2
ts(){ date -u +%FT%TZ; }
echo "$(ts) launch NK=$NK node=$node job=$JOB variant=$V attempt=$ATT"
id=$("${SPUR[@]}" "docker image inspect --format '{{.Id}}' $IMG 2>/dev/null")
if [[ "$id" != sha256:* ]]; then
  echo "$(ts) docker load $TAR on $node"
  "${SPUR[@]}" "cd /tmp && docker load -i $TAR" || { echo "docker load failed"; exit 1; }
  id=$("${SPUR[@]}" "docker image inspect --format '{{.Id}}' $IMG")
fi
echo "$(ts) image id on $node: $id"
[[ "$id" == "$WANT_ID" ]] || echo "WARNING: id differs from the 054 build ($WANT_ID)"
grep -qP "^$node\t$IMG\t$id\$" "$H/image_ids.dcp.tsv" || printf '%s\t%s\t%s\n' "$node" "$IMG" "$id" >> "$H/image_ids.dcp.tsv"
round=pd$V-$ATT
bash "$PD/tools/mk_env.yihou.sh" "$NK" "$round" "$V" || exit 1
export PD_JOB=$JOB
source "$H/env.$round-$NK.yihou.sh"
cd "$H"
bash run.v2.yihou.sh dryrun > "dryrun.$round-$NK.yihou.txt" 2>&1 || { echo "dryrun failed"; exit 1; }
echo "$(ts) up"; bash run.v2.yihou.sh up || { echo "$(ts) UP FAILED (see $PD/runs/$round/status.tsv)"; exit 1; }
for m in census audit smoke; do echo "$(ts) $m"; bash run.v2.yihou.sh $m > "$PD/runs/$round/$m.txt" 2>&1; echo "$m rc=$?"; done
bash "$H/pdcheck.yihou.sh" "$round" > "$PD/runs/$round/pdcheck.txt" 2>&1
if [[ "$V" == *perf || "$V" == A* ]]; then echo "$(ts) T4a skipped: simulated acceptance ON (output garbled by design)"
else echo "$(ts) T4a (math + passkey 8K/34K/96K) client started"; bash "$H/t4.yihou.sh" a; fi
echo "$(ts) LAUNCH DONE round=$round (stack left up); see $PD/runs/$round/{status.tsv,census.txt,audit.txt,smoke.txt,pdcheck.txt,t4/}"
