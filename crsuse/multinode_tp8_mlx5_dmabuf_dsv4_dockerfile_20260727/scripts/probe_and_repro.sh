#!/usr/bin/env bash
# One polling tick: try to grab an 8-GPU node; if we get one, run the ionic dmabuf
# OOM repro (2nd confirmation on a DIFFERENT node). Idempotent + self-limiting so a
# cron every 10min never piles up jobs.
#
# State dir carries a lock + result marker so ticks don't overlap and we stop once
# confirmed. Writes everything under $STATE.
set -u
STATE=/home/yihou/dev/git/infera.yihou.dev/temp/pd_mlx5_1p1d/repro_watch
mkdir -p "$STATE"
LOCK="$STATE/tick.lock"
DONE="$STATE/CONFIRMED"           # exists once repro is conclusively done -> stop
TICKLOG="$STATE/watch.log"
IMG_TAR=/home/yihou/dev/git/infera.yihou.dev/temp/pd_mlx5_1p1d/dsv4-sgl-dmabuf.tar
KITS=/home/yihou/dev/git/infera.yihou.dev/temp/pd_mlx5_1p1d/scripts
BADNODES="crsuse2-m2m-226,crsuse2-m2m-215"   # 226 known bad; 215 = run-1 (want a NEW node)

log(){ echo "[$(date -u +%H:%M:%S)] $*" >> "$TICKLOG"; }

# stop if already confirmed
[ -f "$DONE" ] && { log "already CONFIRMED, nothing to do"; exit 0; }
# single-flight: if a previous tick is still working, skip
if ! mkdir "$LOCK" 2>/dev/null; then log "previous tick still running, skip"; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# --- FIRST: judge any in-flight repro from a previous tick ---
if [ -f "$STATE/inflight" ]; then
  read IJ INODE IRLOG < "$STATE/inflight"
  ISTATE=$(squeue -j "$IJ" -h -o "%T" 2>/dev/null)
  NODEUP=$(scontrol show node "$INODE" 2>/dev/null | grep -oE "State=[A-Z_]+" | head -1)
  # crash signatures in the server log
  CRASH=$(grep -icE "out of memory|HIP-2|hipModuleLoad|Corrupted|REM_ACCESS|Failed to register|no kernel image|Traceback" "$IRLOG" 2>/dev/null)
  READY=$(grep -c "ready to roll" "$IRLOG" 2>/dev/null)
  LASTKV=$(grep -icE "Using FP8 KV cache|Load weight end" "$IRLOG" 2>/dev/null)
  if [ -z "$ISTATE" ]; then
    # job gone. NODE_FAIL right at KV step = the repro; ready-then-gone = walltime, inconclusive
    if [ "$READY" -gt 0 ]; then
      log "inflight $IJ($INODE): reached READY then job ended (walltime?) -> ionic SURVIVED?! inconclusive, node=$NODEUP"
      echo "SURPRISE_survived $INODE" >> "$STATE/verdicts"
    elif [ "$LASTKV" -gt 0 ]; then
      log "★ inflight $IJ($INODE): job DIED at KV-registration step (last log=KV setup, node=$NODEUP). 2nd CONFIRMATION of ionic dmabuf crash."
      echo "CONFIRMED_crash_at_KV $INODE $(date -u +%FT%TZ)" >> "$STATE/verdicts"
      touch "$DONE"
    else
      log "inflight $IJ($INODE): job gone before weight-load done (early NODE_FAIL, node=$NODEUP) - inconclusive (bad node?), retry new node"
      echo "inconclusive_early $INODE" >> "$STATE/verdicts"
    fi
    rm -f "$STATE/inflight"
  elif [ "$CRASH" -gt 0 ]; then
    log "★ inflight $IJ($INODE): server CRASHED with OOM/HIP error while RUNNING. CONFIRMED ionic dmabuf failure."
    grep -iE "out of memory|HIP-2|hipModuleLoad|Corrupted|REM_ACCESS|Failed to register|no kernel image" "$IRLOG" 2>/dev/null | tail -3 >> "$TICKLOG"
    echo "CONFIRMED_oom_running $INODE $(date -u +%FT%TZ)" >> "$STATE/verdicts"
    touch "$DONE"; spur exec "$IJ" bash -c 'docker rm -f repro_ionic2 2>/dev/null'; scancel "$IJ" 2>/dev/null
    rm -f "$STATE/inflight"
  elif [ "$READY" -gt 0 ]; then
    log "inflight $IJ($INODE): reached 'ready to roll' on ionic (unexpected survival). Letting it settle; verdict=survived."
    echo "survived_ready $INODE" >> "$STATE/verdicts"
    touch "$DONE"; spur exec "$IJ" bash -c 'docker rm -f repro_ionic2 2>/dev/null'; scancel "$IJ" 2>/dev/null
    rm -f "$STATE/inflight"
  else
    log "inflight $IJ($INODE) state=$ISTATE still cold-starting; wait next tick"
    exit 0   # keep it, don't grab another
  fi
fi

# don't stack jobs: if I already hold/queue one, skip this tick
MYJOBS=$(squeue -u "$USER" -h -o "%i" 2>/dev/null | wc -l)
if [ "$MYJOBS" -gt 0 ]; then log "already have $MYJOBS job(s), skip grabbing another"; exit 0; fi

log "=== tick: trying to grab an 8-GPU node (exclude $BADNODES) ==="
JOB=$(sbatch --parsable -p amd-spur -q amd-burst-qos -N1 -G8 -t 01:30:00 --exclude=$BADNODES ~/hold_node.sh 2>/dev/null)
[ -z "$JOB" ] && { log "sbatch failed"; exit 0; }
# short wait for a real hold; if not stable in ~40s, give up this tick (cluster busy)
NODE=""
for p in $(seq 1 20); do
  N=$(squeue -j "$JOB" -h -o "%N" 2>/dev/null); [ -n "$N" ] && { NODE="$N"; break; }; sleep 2
done
if ! spur exec "$JOB" true 2>/dev/null; then
  log "job=$JOB not stable (node=${NODE:-none}, likely JobHoldMaxRequeue); cancel + wait next tick"
  scancel "$JOB" 2>/dev/null; exit 0
fi
log "GOT NODE: job=$JOB node=$NODE — running ionic repro"
echo "$JOB $NODE" > "$STATE/current_job"

CTR=repro_ionic2
RLOG="$STATE/repro_${NODE}.log"
HOST_IONIC=$(spur exec "$JOB" bash -c 'readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1' 2>/dev/null)
MY_IP=$(spur exec "$JOB" bash -c 'ip -4 -o addr show dev ens3 | awk "{print \$4}" | cut -d/ -f1' 2>/dev/null)

# load image if missing
spur exec "$JOB" bash -c "docker images -q dsv4-sgl-dmabuf:mlx5 | grep -q . || docker load -i $IMG_TAR" >/dev/null 2>&1
# start container with libionic inject
spur exec "$JOB" bash -c "docker rm -f $CTR 2>/dev/null; docker run -d --name $CTR \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband --group-add video --group-add render \
  --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --security-opt seccomp=unconfined --ipc=host --shm-size=32G \
  --network=host -v /shared_nfs:/shared_nfs -v /home/yihou:/home/yihou \
  -v $HOST_IONIC:/host-libionic/libionic.so:ro dsv4-sgl-dmabuf:mlx5 sleep infinity" >/dev/null 2>&1
# launch the ionic-forced decode leg (detached)
spur exec "$JOB" bash -c "docker exec -d $CTR bash -c \
  'MY_IP=$MY_IP CONC=128 MEMFRAC=0.90 GID=1 LOG=$RLOG bash $KITS/repro_ionic_oom.sh'" >/dev/null 2>&1
log "launched ionic decode on $NODE (MY_IP=$MY_IP); log=$RLOG. Cold start ~11-22min; will judge on next ticks."
echo "$JOB $NODE $RLOG" > "$STATE/inflight"
exit 0
