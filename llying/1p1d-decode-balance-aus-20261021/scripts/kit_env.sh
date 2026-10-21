# Sourced by scripts/pr210/*.sh on the login host. Drives the PR 210 kit (examples/sglang_2p1d_glm5.2 at aab4cc36,
# unmodified) on job 34439's two AUS nodes: every site value goes in through the kit's environment variables, and the
# kit's own `ssh` calls (prefill node -> decode node) go through tools/sshshim, the srun-based ssh of this job.
# Nodes and IPs come from config/topology.2p1d.aus.tsv, which scripts/set_nodes.sh rewrites on every allocation.
R=/perf_apps/liyingli/bench_agentx/2p1d-aus-v0521-20261019
export SHIM_JOB="${SHIM_JOB:-34439}" PATH="$R/tools/sshshim:$PATH"
TOPO="$R/config/topology.2p1d.aus.tsv"
P_NODE="$(awk -F'\t' '$1=="prefill"{print $2; exit}' "$TOPO")"
P_IP="$(awk -F'\t' '$1=="prefill"{print $3; exit}' "$TOPO")"
D_NODE="$(awk -F'\t' '$1=="decode"{print $2; exit}' "$TOPO")"
D_IP="$(awk -F'\t' '$1=="decode"{print $3; exit}' "$TOPO")"
KIT="$R/pr210/src-aab4cc36/examples/sglang_2p1d_glm5.2"
PR_RUNS="$R/pr210/runs"
EXP_ROUTER_BIN="$R/pr210/router-src/rust/target/release/infera-router"

# The kit's env.sh values for AUS (README "Site"). CTR keeps the glm52-llying-2p1d- prefix so that
# scripts/guard_containers.sh removes these containers when the job is preempted.
KIT_SITE=(
    "PREFILL_NODE_IP=$P_IP" "DECODE_NODE_IP=$D_IP" "DECODE_SSH=$D_NODE"
    "MODEL=/perf_apps/data/models/GLM-5.2-MXFP4" "IMAGE=infera-sglang:pr210-aab4cc36"
    "WORK_DIR=/scratch/liyingli_2p1d/pr210/work" "MOONCAKE_DISABLE_HIP_DMABUF=1"
    "PREFILL_WORKERS=1" "PREFILL0_GPUS=${PREFILL0_GPUS:-4,5,6,7}" "SIM_ACCEPT_LEN=${SIM_ACCEPT_LEN-3.61}"
    "CTR=glm52-llying-2p1d-pr210"
)
CTR=glm52-llying-2p1d-pr210
ROUTER_URL="http://$P_IP:28300"
PREFILL_URL="http://$P_IP:27201"

say() { echo "$(date -u +%FT%TZ) ${0##*/}: $*"; }
die() { say "FAIL $*"; exit 1; }
q() { local out="" i; for i in "$@"; do printf -v i '%q' "$i"; out+="${out:+ }$i"; done; echo "$out"; }

# kit_on NODE COMMAND...: run COMMAND in the kit directory on NODE with the site values and the ssh shim exported.
kit_on() {
    local node="$1"; shift
    ssh "$node" "export SHIM_JOB=$SHIM_JOB PATH=$R/tools/sshshim:\$PATH $(q "${KIT_SITE[@]}"); cd $(q "$KIT") && $*"
}

# follow NAME NODE LOGFILE: append `docker logs -f --timestamps NAME` from NODE to LOGFILE (login host, detached).
follow() {
    local name="$1" node="$2" log="$3"
    mkdir -p "$(dirname "$log")"
    setsid nohup ssh "$node" "docker logs -f --timestamps $name" >> "$log" 2>&1 < /dev/null &
    echo $! >> "$(dirname "$log")/followers.pid"
}

# follow_when_created NAME NODE LOGFILE: like follow, once the container exists (polled for up to an hour), so a
# bring-up's logs are visible while it is still starting.
follow_when_created() {
    local name="$1" node="$2" log="$3"
    mkdir -p "$(dirname "$log")"
    setsid nohup bash -c "for i in \$(seq 1 360); do ssh $node 'docker inspect $name >/dev/null 2>&1' && \
        exec ssh $node 'docker logs -f --timestamps $name'; sleep 10; done" >> "$log" 2>&1 < /dev/null &
    echo $! >> "$(dirname "$log")/followers.pid"
}
