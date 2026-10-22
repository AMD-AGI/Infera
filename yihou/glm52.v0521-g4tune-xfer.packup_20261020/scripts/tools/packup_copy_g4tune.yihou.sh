#!/usr/bin/env bash
# Copy-only builder for glm52.v0521-g4tune-xfer.packup_20261020 (2026-10-20). Never deletes, never moves.
# Copies named sources into $P; skips (and lists in results/not_copied.tsv) any file > 4 MB. Re-runnable.
set -uo pipefail
P=/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-g4tune-xfer.packup_20261020
case "$P" in *yihou*) ;; *) echo "refuse: P must contain yihou"; exit 1 ;; esac
W=/shared_nfs/yihou_glm52/g4tune_20261019
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
L=/home/yihou/dev/git/infera.glm52.view/glm52.v0521-cap-c44c48-23k_20261012/debug/poll.md
mkdir -p "$P"/{scripts/harness,scripts/session,scripts/tools,scripts/machine,patches/mounts-v0521,results/bench,results/rounds,results/dryruns,results/machine,results/launch,analysis/ledger,spec}
SKIP=$P/results/not_copied.tsv
printf 'bytes\tpath (external evidence, not copied)\n' > "$SKIP"
cpf(){ # cpf <src> <dstdir>
  local s=$1 d=$2 sz
  [[ -f "$s" ]] || { printf 'missing\t%s\n' "$s" >> "$SKIP"; return; }
  sz=$(stat -c %s "$s")
  if (( sz > 4194304 )); then printf '%s\t%s\n' "$sz" "$s" >> "$SKIP"; return; fi
  mkdir -p "$d"; cp -p "$s" "$d/"
}
for f in $W/env.*.yihou.sh $W/launch_after_prep.yihou.sh $W/a2_after_orphan.yihou.sh $W/xfer_sum.yihou.py $W/mk_xfertime.yihou.py; do cpf "$f" $P/scripts/session; done
for f in $W/dryrun.*.yihou.txt; do cpf "$f" $P/results/dryruns; done
for f in $W/launch.*.yihou.log; do cpf "$f" $P/results/launch; done
cpf $W/research_wf.20261019.json $P/analysis
for f in $W/machine/*.py $W/machine/*.sh; do cpf "$f" $P/scripts/machine; done
for f in $W/machine/*.log $W/machine/*.txt; do cpf "$f" $P/results/machine; done
for f in config.yihou.xferfix2.sh mounts.yihou.v2-xferfix2.tsv config.yihou.offload.sh mounts.yihou.offload-attrs.tsv config.yihou.tokw2.sh \
         config.yihou.mc4340n16.sh config.yihou.mc4340n17.sh topology.yihou.097.tsv config.yihou.E2-target.sh config.yihou.v0521-base.sh \
         mounts.yihou.v2-regbase.tsv chain_cap2.yihou.sh pipeline.v3.yihou.sh run.v3.yihou.sh warmup_pyspy.yihou.sh tokdiff.yihou.sh; do cpf $H/$f $P/scripts/harness; done
for f in conn.yihou-xferfix2.py prefill.yihou-nonfinal-event.py kv_events_publisher.yihou-noblock.py forward_mla_rocm.yihou-noabsorbcfg.py conn.yihou-xferfix.py; do cpf $H/mounts-v0521/src/$f $P/patches/mounts-v0521; done
cpf $W/conn.yihou-xfertime.py $P/patches/mounts-v0521
cpf $N/offload/offloader.yihou-attrs.py $P/patches/mounts-v0521
for f in $N/tools_hits.yihou.py $N/hist.yihou.py $N/prep_node.yihou.sh $CAP/tools/metrics_sampler.yihou.py; do cpf "$f" $P/scripts/tools; done
for r in g4xb1 g4xb2 g4xb3 g4xb4 g4xb5 g4xm1 g4xm2 g4xm3 g4xm4 g4xc1 g4xc2; do
  d=$P/results/rounds/$r
  for f in $R/$r/status.tsv $R/$r/census.txt $R/$r/audit.txt $R/$r/workers.json $R/$r/smoke-response.json $R/$r/spec_accept.after-bench.txt \
           $R/$r/engine-prefill.log $R/$r/engine-decode.a1.log $R/$r/agentx-c56-A1.log $R/$r/agentx-c56-A2.log \
           $R/$r.chain.log $R/$r.pipeline.log $R/$r.pyspy.log $R/$r.a2.log; do [[ -e $f ]] && cpf "$f" $d; done
  if [[ -d $R/$r/audit ]]; then for f in $R/$r/audit/*; do cpf "$f" $d/audit; done; fi
  for a in A1 A2; do
    b=$R/$r/agentx-c56-$a
    [[ -d $b ]] || continue
    for f in $b/agentx_conc56.json $b/aiperf_artifacts/server_metrics_export.csv; do [[ -e $f ]] && cpf "$f" $P/results/bench/$r-c56-$a; done
  done
done
cpf $R/g4xc1/server-logs/prefill-0.log $P/results/rounds/g4xc1/server-logs
cpf $R/g4xm4/server-logs/prefill-0.log $P/results/rounds/g4xm4/server-logs
for r in g4xb5 g4xm3 g4xm4 g4xc1; do for f in $R/$r/server-logs/*.log $R/$r/agentx-c56-A*/aiperf_artifacts/profile_export.jsonl $R/$r/agentx-c56-A*/aiperf_artifacts/server_metrics_export.json; do
  if [[ -f $f ]] && (( $(stat -c %s "$f") > 4194304 )); then printf '%s\t%s\n' "$(stat -c %s "$f")" "$f" >> "$SKIP"; fi; done; done
cpf $L $P/analysis/ledger
echo "copied into $P"; find "$P" -type f | wc -l; du -sh "$P"
