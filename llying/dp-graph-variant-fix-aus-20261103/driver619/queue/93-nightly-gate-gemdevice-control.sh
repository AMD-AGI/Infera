#!/bin/bash
# DRIVER619-20261113, 11-17: same-node control for 92. The gate of 92 on n02-25, with the engine from before the GEM device
# fix (70's package of 11-15: Mooncake main + the combined PR, c7d3abc), derived here as e1ab-sglang:1112-hipgem.
# Expected: every prefill rank's KV located on GPU 0, so the ionic deltas concentrate on GPU 0's NIC instead of 92's
# spread over ionic_0-3; also tells whether 92's long-context results (needle 31/32, gsm8k 30/32) are this node's noise.
# n02-25's /data is full: the kit's four named volumes go to the host's root file system as in 92, removed afterwards
# together with the control image.
IMAGE=e1ab-sglang:1112-hipgem
V=/tmp/llying-kfd619-vol
on() { srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 --exclude=$NODE616 "$@"; }
on bash -c "docker image inspect $IMAGE >/dev/null 2>&1 ||
    bash $K/mc/node_image.sh e1ab-sglang:1112 $IMAGE $K/out/nightly-build-1115T1019/main/pkg"
id=$(on docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)
[ -n "$id" ] || { echo "no $IMAGE"; exit 1; }
VOLS=$(printf "glm52_agentx-%s-$id " prefill0-cache prefill0-aiter-jit decode-cache decode-aiter-jit)
on bash -c "for v in $VOLS; do mkdir -p $V/\$v; docker volume inspect \$v >/dev/null 2>&1 ||
    docker volume create --driver local --opt type=none --opt o=bind --opt device=$V/\$v \$v >/dev/null; done
    docker volume ls -q | grep $id"
OUT=$K/out/gate-gemdevice-control-$(date -u +%m%dT%H%M)
mkdir -p "$OUT"
snap() { on bash $K/bench/ionic_counters.sh > "$OUT/$1.txt"; }
snap before
on env SLURM_JOB_NODELIST=$NODE619 MIN_EVICT=0 E1AB_IMAGE=$IMAGE \
    KFD619_ENV_EXTRA="SGLANG_MOONCAKE_CUSTOM_MEM_POOL=AMD_GEM" PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 EST_MIN=45 \
    bash $K/nightly/nightly_item.sh nightly-gemkv-gate-gemdevice-control gate 1node > "$OUT/gate.log" 2>&1
snap after
awk 'NR == FNR {b[$1 " " $2] = $3; next}
     {d = $3 - b[$1 " " $2]; if (d > 0) v[$2] = v[$2] sprintf(" %s=%s", substr($1, 7), $2 ~ /bytes/ ? int(d / 1e9) "G" : d)}
     END {for (k in v) if (k ~ /ucast_bytes|retx|ack_timeout|seq_err|cnp|ecn/) print k ":" v[k]}' \
    "$OUT/before.txt" "$OUT/after.txt" | sort > "$OUT/deltas.txt"
tail -7 $K/nightly/logs/nightly.out
echo "## ionic deltas (ionic_N=value) on $NODE619"
cat "$OUT/deltas.txt"
on bash -c "for v in $VOLS; do docker volume rm \$v >/dev/null; done; docker rmi $IMAGE >/dev/null
    docker run --rm --entrypoint bash -v /tmp:/x e1ab-sglang:1112 -c 'rm -rf /x/llying-kfd619-vol'; df -h /data | tail -1"
