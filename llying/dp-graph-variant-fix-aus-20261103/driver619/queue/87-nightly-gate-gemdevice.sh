#!/bin/bash
# DRIVER619-20261113 formal delivery: the one-node 4+4 gate of 71 again, on e1ab-sglang:1112-hipgem2 (Mooncake main + PR 1
# with the GEM device fix of 11-16, from 81). HIP reports a GEM buffer as GPU 0's, so before the fix every rank's KV was
# located on GPU 0 and went through GPU 0's NIC (crusoe 11-16: P8D8 122 TB on ionic_0, the rest near 0). ionic counters
# on the node before and after: the KV bytes must now spread over the NICs of prefill GPUs 0-3 and decode GPUs 4-7.
OUT=$K/out/gate-gemdevice-$(date -u +%m%dT%H%M)
mkdir -p "$OUT"
snap() { srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 --exclude=$NODE616 bash $K/bench/ionic_counters.sh > "$OUT/$1.txt"; }
snap before
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 --exclude=$NODE616 env SLURM_JOB_NODELIST=$NODE619 MIN_EVICT=0 \
    E1AB_IMAGE=e1ab-sglang:1112-hipgem2 \
    KFD619_ENV_EXTRA="SGLANG_MOONCAKE_CUSTOM_MEM_POOL=AMD_GEM" PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 EST_MIN=45 \
    bash $K/nightly/nightly_item.sh nightly-gemkv-gate-gemdevice gate 1node > "$OUT/gate.log" 2>&1
snap after
awk 'NR == FNR {b[$1 " " $2] = $3; next}
     {d = $3 - b[$1 " " $2]; if (d > 0) v[$2] = v[$2] sprintf(" %s=%s", substr($1, 7), $2 ~ /bytes/ ? int(d / 1e9) "G" : d)}
     END {for (k in v) if (k ~ /ucast_bytes|retx|ack_timeout|seq_err|cnp|ecn/) print k ":" v[k]}' \
    "$OUT/before.txt" "$OUT/after.txt" | sort > "$OUT/deltas.txt"
tail -12 $K/nightly/logs/nightly.out
echo "## ionic deltas (ionic_N=value) on $NODE619"
cat "$OUT/deltas.txt"
