# Helpers for point.sh, debug.sbatch and entry.sh. Callers set NODE_HOSTS (short hostnames, node 0 first: it runs
# etcd, the router and the AgentX client, and is the host these scripts run on).
R=/home/liyingli/bench_agentx/ix5138-crusoe-20261120
IMAGE=lmsysorg/sglang-rocm@sha256:e016f33423250be30bde9e0888d159b67b01e00944de1edc11f36dd1f3dd0269
ETCD_IMAGE=quay.io/coreos/etcd:v3.5.14
LOCAL=/mnt/m2m_nobackup/liyingli_ix5138
MODEL_SRC=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
MODEL=$LOCAL/models/GLM-5.2-MXFP4

say() { echo "== $(date -u '+%F %T') $*"; }

# The node's python3, else the image's (stdin passes through for `py -c ... <<<`).
py() {
    if command -v python3 >/dev/null; then
        python3 "$@"
    else
        docker run --rm -i --network host --user "$(id -u):$(id -g)" -v "$R:$R" -v /tmp:/tmp --entrypoint python3 \
            "$IMAGE" "$@"
    fi
}

# Run a shell command on node $1. Spur ends the whole job when a step exits non-zero, so a remote step always exits 0
# and its status comes back as the last "__rc=" line; remote steps do not inherit this environment.
on() {
    local n=$1 host out rc
    shift
    host=${NODE_HOSTS[$n]}
    if [[ "$host" == "$(hostname -s)" ]]; then
        bash -c "$*" </dev/null
        return
    fi
    out=$(srun --overlap -D /tmp -N1 -n1 -w "$host" --gres=gpu:8 /bin/bash -c "$*; rc=\$?; printf '\n__rc=%s\n' \$rc; exit 0" \
        </dev/null 2>&1 | tr -d '\r')
    rc=$(sed -n 's/^__rc=//p' <<<"$out" | tail -1)
    out=$(grep -v -e '^__rc=' -e '^srun: dispatched' <<<"$out")
    [[ -n "$out" ]] && printf '%s\n' "$out"
    return "${rc:-1}"
}

# "host,ip,ifname" of node $1: the address on the interface that carries the default route.
node_facts() {
    on "$1" 'ifc=$(ip -o -4 route show default | awk "{print \$5; exit}");
        ip=$(ip -o -4 addr show dev "$ifc" | awk "{split(\$4, a, \"/\"); print a[1]; exit}");
        echo "$(hostname -s),$ip,$ifc"'
}

# Percent of VRAM in use per GPU on node $1, space-separated.
vram_pct() {
    on "$1" 'for d in /sys/class/drm/card*/device; do [ -r $d/mem_info_vram_total ] || continue;
        echo $(( 100 * $(cat $d/mem_info_vram_used) / $(cat $d/mem_info_vram_total) )); done | tr "\n" " "'
}

# Wait until every GPU on node $1 is below 2 % VRAM and the host has 90 % of its memory available again (HiCache pins
# about 1 TB per PP4 prefill and takes minutes to give it back); $2 = timeout in seconds.
wait_idle() {
    local n=$1 deadline=$((SECONDS + ${2:-1800})) busy mem
    while ((SECONDS < deadline)); do
        busy=$(vram_pct "$n" | tr ' ' '\n' | awk '$1 >= 2' | wc -l)
        mem=$(on "$n" "awk '/MemTotal/ {t = \$2} /MemAvailable/ {a = \$2} END {print int(100 * a / t)}' /proc/meminfo")
        ((busy == 0 && mem >= 90)) && return 0
        sleep 30
    done
    say "node $n not idle after ${2:-1800} s: GPUs at or above 2 %: $busy, host memory available $mem %"
    return 1
}

# ionic RDMA counters of node $1 as "nic counter value" lines.
ionic_counters() {
    on "$1" 'for c in /sys/class/infiniband/ionic_*/ports/1/hw_counters; do n=${c#/sys/class/infiniband/}; n=${n%%/*};
        for k in tx_rdma_ucast_bytes rx_rdma_ucast_bytes tx_rdma_retx_pkts tx_rdma_ack_timeout rx_rdma_cnp_pkts; do
            [ -r $c/$k ] && echo "$n $k $(cat $c/$k)"; done; done'
}

# Copy the model to node $1's NVMe unless it is already there (stage_model.sh of the 11-16 crusoe reproduction).
stage_model() {
    on "$1" "mkdir -p $LOCAL/models $LOCAL/hf_home $LOCAL/cache && bash /home/liyingli/bench_agentx/pd-repro-20261116/scripts/stage_model.sh $MODEL_SRC $MODEL"
}

# Add this node to $R/exclude_nodes.txt and resubmit "$3 $4..." excluding the list, at most $2 times (count in $1).
resubmit_bad_node() {
    local counter=$1 max=$2 k
    shift 2
    grep -qx "$(hostname -s)" "$R/exclude_nodes.txt" || hostname -s >>"$R/exclude_nodes.txt"
    k=$(cat "$counter" 2>/dev/null || echo 0)
    if ((k < max)); then
        echo $((k + 1)) >"$counter"
        say "node $(hostname -s) unusable; resubmitted: $(cd /tmp && sbatch -x "$(paste -sd, "$R/exclude_nodes.txt")" "$@" 2>&1)"
    else
        say "node $(hostname -s) unusable; resubmit limit $max reached"
    fi
}

# Pull the image and etcd on node $1 (the stock image is 89 GB; 105-245 s on v3).
pull_images() {
    on "$1" "timeout 2400 docker pull -q $IMAGE && timeout 600 docker pull -q $ETCD_IMAGE"
}
