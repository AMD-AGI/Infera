#!/bin/bash
# Standard per-experiment analysis (run on 138): bash analyze_run.sh LABEL [CONC]
# -> throughput line, PD log bottleneck, PP stage times, prefill beats, HiCache/eviction counters.
LABEL=$1; C=${2:-104}
L=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/llr
S=/home/liyingli/bench_agentx/baseline/Infera/llying/2p1d-v0522-20261020/scripts
P=/home/liyingli/bench_agentx/baseline/Infera/llying/prefill-opt-20261027
RUN=$(grep -o 'run=[^ ]*' $L/perf/$LABEL/c$C.log | head -1 | cut -d= -f2)
[ -n "$RUN" ] || { echo "no run dir for $LABEL"; exit 1; }
B=$RUN/agentx/benchmark.log
D=$(basename $RUN | grep -o '1[0-9]\{3\}T' | head -1)
day="2026-${D:0:2}-${D:2:2}"
t0=$(grep -m1 'Phase profiling (profiling) started' $B | cut -c1-8)
t1=$(grep -m1 'Phase profiling (profiling) sending complete' $B | cut -c1-8)
echo "== $LABEL  run=$RUN  window $day $t0 .. $t1"
T=$(mktemp -d); ln -s $RUN/agentx/agentx_conc$C.json $T/; ln -s $RUN/agentx/aiperf_artifacts/profile_export_aiperf.json $T/
echo "-- summarize_points"; python3 $S/summarize_points.py $T 2>&1 | tail -3; rm -rf $T
echo "-- analyze_pd_logs"; python3 $S/analyze_pd_logs.py $RUN ${day}T$t0 ${day}T$t1 2>&1 | tail -12
for f in $RUN/logs/*prefill*.log; do
  echo "-- pp_stage_times $(basename $f)"; python3 $S/pp_stage_times.py $f ${day}T$t0 ${day}T$t1 2>&1 | tail -8
  echo "-- prefill_beats $(basename $f)"; python3 $P/scripts/prefill_beats.py $f ${day}T$t0 ${day}T$t1 2>&1 | tail -8
done
for m in $RUN/metrics/prefill*.prom; do
  echo "-- counters $(basename $m) (whole deployment, per pp_rank)"
  python3 - "$m" <<'EOF'
import re, sys, collections
v = collections.defaultdict(dict)
for line in open(sys.argv[1]):
    if line.startswith("#"): continue
    m = re.match(r'(sglang:\w+)\{([^}]*)\} (\S+)', line)
    if not m: continue
    name, lab, val = m.groups()
    pp = re.search(r'pp_rank="(\d+)"', lab); mode = re.search(r'mode="(\w+)"', lab)
    if not pp: continue
    v[int(pp.group(1))][name + ("/" + mode.group(1) if mode else "")] = float(val)
for pp in sorted(v):
    d = v[pp]; inp = d.get("sglang:prefill_effective_tokens_total/input", 0); steps = inp / 8192 or 1
    ev = d.get("sglang:eviction_duration_seconds_count", 0); evs = d.get("sglang:eviction_duration_seconds_sum", 0)
    print(f"pp{pp}: computed {inp/1e6:.1f}M (~{steps:.0f} steps)  dev_hit {d.get('sglang:prefill_effective_tokens_total/device_hit',0)/1e6:.0f}M"
          f"  host_hit {d.get('sglang:prefill_effective_tokens_total/host_hit',0)/1e6:.0f}M"
          f"  evictions {ev:.0f} ({ev/steps:.2f}/step) total {evs:.0f}s avg {1e3*evs/max(ev,1):.0f}ms"
          f"  evicted/step {d.get('sglang:evicted_tokens_total',0)/steps/1e3:.0f}K"
          f"  loadback/step {d.get('sglang:load_back_tokens_total',0)/steps/1e3:.0f}K"
          f"  backup {d.get('sglang:hicache_backup_tokens_total',0)/1e6:.0f}M tok {d.get('sglang:hicache_backup_duration_seconds_sum',0):.0f}s"
          f"  dropped {d.get('sglang:hicache_dropped_tokens_total',0):.0f}"
          f"  host_used {d.get('sglang:hicache_host_used_tokens',0)/1e6:.1f}/{d.get('sglang:hicache_host_total_tokens',0)/1e6:.1f}M")
EOF
done
