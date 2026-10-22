#!/usr/bin/env bash
# Copy-only pack-up builder for glm52.v0521-offloaderv2-highconc.packup_20261015 (run on a spur login host).
# Never deletes or moves anything; sources are read-only. Writes only inside $P.
set -uo pipefail
P=/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-offloaderv2-highconc.packup_20261015
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
L=/home/yihou/dev/git/infera.glm52.view/glm52.v0521-cap-c44c48-23k_20261012
N136=/mnt/m2m_nobackup/yihou_v0521dbg/runs
LIM=4000000
c(){ [ -e "$1" ] && cp -p "$1" "$2" || echo "MISSING $1"; }
gz(){ [ -e "$1" ] || { echo "MISSING $1"; return; }
  local n; n=$(gzip -c "$1" | wc -c)
  if [ "$n" -lt "$LIM" ]; then gzip -c "$1" > "$2"; else echo "TOOBIG-GZ $1 ($n)"; return 1; fi; }
excerpt(){ { echo "### EXCERPT of $1 ($(stat -c %s "$1") bytes): head 2 MB / key lines (<=1 MB) / tail 1 MB"
    echo "### ---- HEAD ----"; head -c 2000000 "$1"
    echo; echo "### ---- KEY LINES ----"
    grep -anE 'Traceback|Error|error|Exception|OOM|out of memory|abort|max_total_num_tokens|Memory pool|avail mem|Load weight end|offloader|warmup|Engine startup|baton|retract|flush' "$1" | head -c 1000000
    echo; echo "### ---- TAIL ----"; tail -c 1000000 "$1"; } | gzip -c > "$2"; }
mkdir -p $P/scripts/harness $P/scripts/session/shimbin $P/scripts/tools $P/patches $P/results/rounds $P/results/metrics \
         $P/results/dryruns $P/results/correctness $P/results/pyspy $P/env $P/logs/server $P/analysis/ledger $P/spec
# ---- harness (verbatim)
cp -pr $H/bench-harness $H/tools $H/mounts-v0521 $P/scripts/harness/
for f in config.yihou.v0521-base.sh config.yihou.E2-target.sh config.yihou.mc4340n.sh config.yihou.mc4340n5.sh config.yihou.mc4340n6.sh \
  config.yihou.offload.sh config.yihou.tokw2.sh mounts.yihou.offload-attrs.tsv mounts.yihou.v2-regbase.tsv mounts.yihou.v2.tsv \
  env.ofg4c-252.yihou.sh env.ofg4-252.yihou.sh env.ofg2-288.yihou.sh env.ofg4b-136.yihou.sh env.ofg2b-136.yihou.sh env.ofg4h-136.yihou.sh \
  env.ofg3b-v3020.yihou.sh env.pf90b-v3026.yihou.sh env.ofg2h-v3026.yihou.sh env.ofg2h2-v3026.yihou.sh env.ofg2h3-v3026.yihou.sh env.tw2dry-v3026.yihou.sh \
  env.ofs0-136.yihou.sh env.pf90-089.yihou.sh topology.yihou.136.tsv topology.yihou.252.tsv topology.yihou.288.tsv topology.yihou.020.tsv topology.yihou.026.tsv \
  chain_cap2.yihou.sh chain_cap2l.yihou.sh chain_cx.yihou.sh cont_bench.yihou.sh ofs_seq.yihou.sh ofs_seq136l.yihou.sh after_ofg4b136l.yihou.sh \
  after_ofg4b136l2.yihou.sh cont2_v3026.yihou.sh warmup_pyspy.yihou.sh run.v3.yihou.sh pipeline.v3.yihou.sh tokdiff.yihou.sh; do c $H/$f $P/scripts/harness/; done
# ---- session tools
for f in tools_hits.yihou.py collect_env.yihou.sh correctness_tests.yihou.py prep_node.yihou.sh results_csv_v15b.yihou.py summary.v0521cap2.yihou.md; do c $N/$f $P/scripts/session/; done
c $N/shimbin/ssh $P/scripts/session/shimbin/
c $CAP/tools/metrics_sampler.yihou.py $P/scripts/tools/
c $N/packup_copy_highconc.yihou.sh $P/scripts/tools/
# ---- patch (unchanged since the first offload kit; copied for self-containment)
c $N/offload/offloader.py.orig $P/patches/; c $N/offload/offloader.yihou-attrs.py $P/patches/; c $N/offload/build_offpatch.yihou.py $P/patches/
diff -u $N/offload/offloader.py.orig $N/offload/offloader.yihou-attrs.py > $P/patches/0001-offloader-carry-is_shuffled.diff
# ---- env captures, prep logs
for f in $N/env/collect_env.252.20261015.txt $N/env/collect_env.288.20261015.txt $N/env/collect_env.136.20261014.txt; do c $f $P/env/; done
for k in 252 288 020 026; do c $N/logs/prep.$k.yihou.log $P/env/; done
# ---- dryruns + tokdiffs
for f in dryrun.ofg4-252 dryrun.ofg4c-252 dryrun.ofg2-288 dryrun.ofs0-136 dryrun.ofg4b-136 dryrun.ofg2b-136 dryrun.ofg4h-136 dryrun.ofg3-266 dryrun.ofg3b-v3020 \
         dryrun.pf90-089 dryrun.pf90b-v3026 dryrun.ofg2h-v3026 dryrun.ofg2h2-v3026 dryrun.ofg2h3-v3026 dryrun.tw2dry-v3026; do c $N/$f.yihou.txt $P/results/dryruns/; done
td(){ ( cd $H && bash tokdiff.yihou.sh $N/dryrun.$1.yihou.txt $N/dryrun.$2.yihou.txt ) > $P/results/dryruns/tokdiff.$1-vs-$2.txt; }
td ofs0-136 ofg4b-136; td ofg4b-136 ofg4h-136; td ofg3-266 ofg3b-v3020; td pf90-089 pf90b-v3026; td ofg2-288 ofg2h-v3026; td ofg2h2-v3026 ofg2h3-v3026
# ---- correctness (ofg4c final)
d=$P/results/correctness/ofg4c; mkdir -p $d; for f in results.jsonl correctness.log; do gz $N/runs/ofg4c/$f $d/$f.gz || true; done
# ---- rounds
for r in ofg4c ofg4 ofg2 ofg4b ofg4h ofg3b pf90b ofg2h ofg2h.fail1 ofg2h2 ofg2h3; do
  d=$P/results/rounds/$r; mkdir -p $d
  for f in status.tsv audit.txt census.txt smoke-response.json spec_accept.after-bench.txt workers.json engine-prefill.log engine-decode.a1.log \
           stop.log stop.seq.log stop.user.log stop.release252.log stop.release288.log ofg2h.chain.log ofg2h.pipeline.log; do
    [ -e $R/$r/$f ] && cp -p $R/$r/$f $d/; done
  [ -d $R/$r/audit ] && { mkdir -p $d/audit; for f in $R/$r/audit/*; do gzip -c $f > $d/audit/$(basename $f).gz; done; }
  for f in $R/$r.chain.log $R/$r.pipeline.log $R/$r.seq.log $R/$r.cont.log $R/$r.pyspy.log; do [ -e $f ] && cp -p $f $d/; done
  for f in $R/$r/agentx-*.log; do [ -e $f ] && gzip -c $f > $d/$(basename $f).gz; done
done
for f in $R/ofs136.seq.log $R/ofs136.after.log $R/ofs136.after2.log $R/ofg4.seq.log $R/ofg2.seq.log; do [ -e $f ] && cp -p $f $P/results/rounds/; done
# ---- py-spy dumps
for r in ofg2h2 ofg2h3; do [ -d $R/$r/pyspy ] && { mkdir -p $P/results/pyspy/$r; cp -p $R/$r/pyspy/* $P/results/pyspy/$r/ 2>/dev/null; }; done
# ---- bench dirs (complete or partial)
for b in ofg2/agentx-c48-A1 ofg2/agentx-c48-A2 ofg4/agentx-c48-A1 ofg4b/agentx-c48-A1 ofg4b/agentx-c48-A2 ofg4b/agentx-c56-A3 ofg3b/agentx-c48-A1 \
         ofg3b/agentx-c48-A2 ofg2h3/agentx-c56-A1.fail1 ofg2h3/agentx-c56-A1; do
  s=$R/$b; [ -d $s ] || { echo "no bench dir $b"; continue; }; d=$P/results/${b%%/*}-${b#*agentx-}; mkdir -p $d
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
for r in ofg2 ofg4 ofg4b ofg3b ofg2h3; do for f in $N/runs/$r/metrics.A*.tsv; do [ -e $f ] && { gz $f $P/results/metrics/$r.$(basename $f).gz || echo "metrics too big: $f"; }; done; done
# ---- server logs on NFS (gzip < 4 MB, else excerpt); live rounds are snapshots
for r in ofg4c ofg4 ofg2 ofg3b pf90b ofg2h.fail1 ofg2h2 ofg2h3; do mkdir -p $P/logs/server/$r
  for f in $R/$r/server-logs/*.log; do [ -e $f ] || continue; b=$(basename $f)
    gz $f $P/logs/server/$r/$b.gz || excerpt $f $P/logs/server/$r/$b.excerpt.txt.gz; done; done
# ---- server logs node-local on 136 (read over ssh; gzip, cap check after)
mkdir -p $N/tmp_pack
for r in ofg4b ofg4h; do mkdir -p $P/logs/server/$r
  for b in prefill-0.log decode-0.a1.log; do
    ssh -o BatchMode=yes -o ConnectTimeout=15 crsuse2-m2m-136 "test -f $N136/$r/server-logs/$b && gzip -c $N136/$r/server-logs/$b" > $N/tmp_pack/$r.$b.gz
    sz=$(stat -c %s $N/tmp_pack/$r.$b.gz)
    if [ "$sz" -gt 0 ] && [ "$sz" -lt $LIM ]; then cp -p $N/tmp_pack/$r.$b.gz $P/logs/server/$r/$b.gz
    else
      ssh -o BatchMode=yes -o ConnectTimeout=15 crsuse2-m2m-136 "f=$N136/$r/server-logs/$b; test -f \$f && { head -c 2000000 \$f; echo; echo '### ---- KEY LINES ----'; grep -anE 'Traceback|Error|OOM|out of memory|max_total_num_tokens|Memory pool|avail mem|Load weight end|offloader|warmup|Engine startup' \$f | head -c 1000000; echo '### ---- TAIL ----'; tail -c 1000000 \$f; } | gzip -c" > $P/logs/server/$r/$b.excerpt.txt.gz
      echo "136 $r $b: full gzip $sz B -> excerpt"
      : > $N/tmp_pack/$r.$b.gz
    fi
  done
done
# ---- results.csv + ledgers + spec
python3 -B $N/results_csv_v15b.yihou.py highconc > $P/results/results.csv
c $L/debug/poll.md $P/analysis/ledger/; c $L/debug/working_process.md $P/analysis/ledger/; c $L/mission.md $P/spec/
echo COPY-DONE
