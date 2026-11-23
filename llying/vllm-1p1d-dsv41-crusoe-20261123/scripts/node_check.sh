#!/usr/bin/env bash
# Node checks at the start of a run job; sourced by run_once2.sbatch / run_opt.sbatch after env.sh.
# node_ok: 0 if this node can host a run, else prints why and returns 1. resubmit SCRIPT: excludes this node and
# submits SCRIPT again with the same name and exports (up to 4 times; RESUB counts them).
node_ok() {
    local h jobs others foreign load1 dk r memavail shmfree
    h=$(hostname -s)
    id -nG | tr ' ' '\n' | grep -qx docker || { echo "== $h: no docker group for $(id -un) ($(id -nG))"; return 1; }
    dk=$( (timeout 180 docker pull -q busybox:1.36 && timeout 60 docker run --rm --network none busybox:1.36 true && echo run-ok) 2>&1)
    case "$dk" in *run-ok*|*toomanyrequests*) dk=ok ;; *) dk=fail ;; esac
    memavail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    shmfree=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc 0-9)
    r="amdgpu=$(cat /sys/module/amdgpu/version) docker=$(curl -s -m 15 -o /dev/null -w %{http_code} --unix-socket /var/run/docker.sock http://localhost/images/json) run=$dk memavail_gb=$memavail shm_free_gb=$shmfree"
    echo "-- $r"
    [[ "$r" == *docker=200* && "$r" == *run=ok* ]] || { echo "== $h: docker check failed"; return 1; }
    # UMBP 449 GB + two engines' Engram tables (2 x 189 GiB in /dev/shm) + engine and client processes.
    (( memavail >= 1100 && shmfree >= 450 )) || { echo "== $h: not enough free host memory / shm"; return 1; }
    # Containers on a node where spur shows no other job run outside the scheduler (v3-180 on 11-23: ten agent
    # containers, load ~70). They starved AIPerf's workers at start-up and hung the replay, so refuse such nodes,
    # and any node already loaded before we start.
    jobs=$(squeue -w "$h" -h -o %i 2>/dev/null)
    others=$(grep -vcx "${SLURM_JOB_ID:-x}" <<<"$jobs")
    foreign=$(docker ps --format '{{.Names}}' | grep -vc '^lyl-dsv41')
    load1=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
    echo "-- other jobs $others, other containers $foreign, load $(cut -d' ' -f1-3 /proc/loadavg)"
    if (( (others == 0 && foreign > 0) || load1 > 32 )); then
        echo "== $h: busy outside the scheduler"; return 1
    fi
    return 0
}

resubmit() {
    local script=$1 n=${RESUB:-0}
    (( n < 4 )) || { echo "== resubmitted $n times already; giving up"; return 0; }
    grep -qx "$(hostname -s)" "$W/exclude_nodes.txt" || hostname -s >> "$W/exclude_nodes.txt"
    # Drop this job's own scheduler and GPU variables so the new job gets fresh ones (SPUR_* stays: it names the
    # controller).
    local drop=() v
    for v in $(env | grep -oE '^(SLURM_[A-Z0-9_]+|ROCR_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES|GPU_DEVICE_ORDINAL|HIP_VISIBLE_DEVICES|PREFILL_GPUS|DECODE_GPUS)='); do
        drop+=(-u "${v%=}")
    done
    ( cd "$W" && env "${drop[@]}" sbatch -J "${SLURM_JOB_NAME:-lyl-dsv41}" -x "$(paste -sd, exclude_nodes.txt)" \
        --export=ALL,RESUB=$((n + 1)) "$script" ) 2>&1 | sed 's/^/== /'
}
