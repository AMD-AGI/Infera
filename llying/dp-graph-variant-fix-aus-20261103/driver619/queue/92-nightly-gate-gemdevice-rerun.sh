#!/bin/bash
# DRIVER619-20261113, 11-17: 87 again. 87 failed at start-up on n02-25: /data (docker's root, 3.5 TB) is full with other
# users' images and volumes, so docker could not populate the kit's named volumes for /root/.cache and the AITER JIT
# directory (~5 GB each). Same gate, with those four volumes created beforehand as bind volumes on the host's root file
# system (docker still copies the image's content into an empty one), removed afterwards.
VOLS=$(printf "glm52_agentx-%s-9e70a25b6602 " prefill0-cache prefill0-aiter-jit decode-cache decode-aiter-jit)
V=/tmp/llying-kfd619-vol
on() { srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 --exclude=$NODE616 "$@"; }
on bash -c "for v in $VOLS; do mkdir -p $V/\$v; docker volume inspect \$v >/dev/null 2>&1 ||
    docker volume create --driver local --opt type=none --opt o=bind --opt device=$V/\$v \$v >/dev/null; done
    docker volume ls -q | grep 9e70a25b6602"
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
on bash -c "for v in $VOLS; do docker volume rm \$v >/dev/null; done
    docker run --rm --entrypoint bash -v /tmp:/x e1ab-sglang:1112 -c 'rm -rf /x/llying-kfd619-vol'; df -h /data | tail -1"
