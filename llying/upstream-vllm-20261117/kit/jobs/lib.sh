# Sourced by the plan steps and the job scripts (upstream PR / issue checks).
#   wt BRANCH                  worktree of a vLLM branch ($U/wt/<branch>, made on the login node by mk_worktrees.sh)
#   iwt BRANCH                 worktree of an Infera branch ($U/iwt/<branch>)
#   vllm_mounts WT BASE        -v args mounting every vllm/ file the branch adds or changes vs BASE over the image
#   run_vllm NAME IMG WT BASE CMD   CMD (bash -c) in IMG with the branch's vllm files, its tests/ at /work/tests,
#                              checks/ at /checks, results dir $OUT at /out; GPUs, host IPC / network
#   need_ready ITEM...         wait (up to 40 min) for prep_node.sh's marker of each item
#   pytest_in IMG WT BASE NAME ARGS   pip-install the test deps, then python -m pytest ARGS (junit xml to /out)
U=/home/liyingli/bench_agentx/upstream-20261119
IMG_V034=vllm/vllm-openai-rocm:v0.34.0@sha256:3cb773d23f063afa86291ef75773cdd486154ddea94c213300f4ed7cc42b7af1
IMG_NIGHTLY=vllm/vllm-openai-rocm:nightly-9c1d45c21836370104d4eb51d3b46eda20719c47
NIGHTLY_COMMIT=9c1d45c21836370104d4eb51d3b46eda20719c47
SP=/usr/local/lib/python3.12/dist-packages
NL=/mnt/m2m_nobackup/liyingli_up
MODEL_DIR=/mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4
PIP_TEST="pytest tblib pytest-asyncio pytest-timeout"
mkdir -p "$NL" 2> /dev/null

wt() { echo "$U/wt/$1"; }
iwt() { echo "$U/iwt/${1#pr/}"; }

vllm_mounts() {
    local w=$1 base=$2 f
    git -C "$w" diff --name-only --diff-filter=AM "$base" HEAD -- vllm/ | while read -r f; do
        printf -- '-v %s:%s:ro ' "$w/$f" "$SP/$f"
    done
}

gpu_args() {
    local g gid gids=""
    for g in video render; do
        gid=$(getent group "$g" | cut -d: -f3)
        [[ -n "$gid" ]] && gids+="--group-add $gid "
    done
    echo "--device /dev/kfd --device /dev/dri $gids--cap-add SYS_PTRACE --cap-add IPC_LOCK \
--security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ipc host --network host --shm-size 64g"
}

need_ready() {
    local item t0
    t0=$(date +%s)
    for item in "$@"; do
        until [[ -e "$NL/ready/$item" ]]; do
            (( $(date +%s) - t0 > 2400 )) && { echo "prep item $item not ready after 40 min"; return 1; }
            sleep 15
        done
    done
}

run_vllm() {
    local name=$1 img=$2 w=$3 base=$4 cmd=$5
    mkdir -p "$NL/pip" "${OUT:?OUT}"
    docker rm -f "$name" > /dev/null 2>&1
    # shellcheck disable=SC2046
    docker run --rm --name "$name" $(gpu_args) $(vllm_mounts "$w" "$base") ${EXTRA_DOCKER:-} \
        -v "$w/tests:/work/tests:ro" -v "$w/pyproject.toml:/work/pyproject.toml:ro" \
        -v "$U/checks:/checks:ro" -v "$OUT:/out" -v "$NL/pip:/root/.cache/pip" \
        -e PYTHONDONTWRITEBYTECODE=1 -e PYTEST_ADDOPTS="-p no:cacheprovider" -w /work \
        --entrypoint bash "$img" -c "$cmd"
}

pytest_in() {
    local img=$1 w=$2 base=$3 name=$4
    shift 4
    run_vllm "$name" "$img" "$w" "$base" "pip install -q $PIP_TEST ${PIP_EXTRA:-} > /out/pip-$name.log 2>&1;
python3 -c 'import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__, \"hip\", torch.version.hip)';
python3 -m pytest -q -rfEs --junitxml=/out/$name.xml $*"
}

model_dir() {
    [[ -f "$MODEL_DIR/.complete" ]] && { echo "$MODEL_DIR"; return 0; }
    return 1
}

aiter_jit_volume() {
    local id
    id=$(docker image inspect -f '{{.Id}}' "${1:-$IMG_V034}" 2> /dev/null) || return 1
    echo "up-aiter-jit-${id:7:12}"
}

# The aiter JIT modules a run compiled, kept across nodes (a fresh node otherwise recompiles them at engine start).
restore_caches() {
    local v t
    v=$(aiter_jit_volume) || return 0
    t=$U/cache/$v.tgz
    if [[ -e "$t" ]] && ! docker volume inspect "$v" > /dev/null 2>&1; then
        docker volume create "$v" > /dev/null
        docker run --rm -v "$v:/v" -v "$U/cache:/c:ro" --entrypoint sh busybox:1.36 -c "tar xzf /c/$v.tgz -C /v" \
            && echo "$(date -u +%T) restored $v"
    fi
    touch "$NL/ready/aiterjit"
}

save_caches() {
    local v
    v=$(aiter_jit_volume) || return 0
    docker volume inspect "$v" > /dev/null 2>&1 || return 0
    mkdir -p "$U/cache"
    docker run --rm -v "$v:/v:ro" -v "$U/cache:/c" --entrypoint sh busybox:1.36 -c "tar czf /c/$v.tgz.tmp -C /v . && mv /c/$v.tgz.tmp /c/$v.tgz"
}
