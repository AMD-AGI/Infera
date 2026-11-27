#!/bin/bash
#SBATCH -J lyl-vpd-fp8build
#SBATCH -p amd-spur-v2
#SBATCH -A amd-agentx
#SBATCH -q amd-burst-qos
#SBATCH -N 1
#SBATCH -c 96
#SBATCH --mem 256G
#SBATCH -t 03:00:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/vllm-pd-20261117/logs/fp8build-%j.out
# CPU-only job (no GPUs requested, so the cluster's GPU-utilisation watchdog has nothing to watch): build aiter's
# FP8 bpreshuffle tuner module (module_gemm_a8w8_bpreshuffle_tune, 159 CK instances; a single-process build on a GPU node
# ran past 35 min twice, 314276 / 314305) for gfx950 only, and copy the .so to the shared disk. The GPU tuning job
# (once/tune-fp8-bpreshuffle.sh, TUNE_PREBUILT_SO) then mounts it and goes straight to measuring.
R=/home/liyingli/bench_agentx/vllm-pd-20261117
B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123
OUT=$B/gemm-tune/fp8-build
IMAGE=infera-vllm:v0340-0c5e5e89
SP=/usr/local/lib/python3.12/dist-packages
MOD=module_gemm_a8w8_bpreshuffle_tune
mkdir -p "$OUT" && chmod 777 "$OUT"
log() { echo "$(date -u +%T) $*" | tee -a "$OUT/summary.txt"; }
: > "$OUT/summary.txt"
log "job $SLURM_JOB_ID on $(hostname -s): $(nproc) CPUs in the job, image $(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)"
cp "$0" "$OUT/build.sbatch" 2>/dev/null
# a node that never ran our jobs has no image: build it as scripts/node_prep.sh does (commit 0c5e5e89), no model staging
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    REPO=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52
    d=/mnt/m2m_nobackup/liyingli_vpd/build-0c5e5e89
    log "no $IMAGE on this node: building it"
    rm -rf "$d"; mkdir -p "$d"
    git -C "$REPO" archive 0c5e5e89 | tar -x -C "$d"
    ( cd "$d" && docker build --progress=plain -f llying/vllm-image-v0340-20261117/Dockerfile -t "$IMAGE" . > "$OUT/image-build.log" 2>&1 )
    log "image build rc=$?"
    docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "image build failed (image-build.log)"; echo "$(date -u +%m%d-%T) fp8 tune module build failed: no image" >> $R/logs/summary.out; exit 0; }
fi
t0=$(date +%s)
timeout -k 60 9600 docker run --rm --init --network none --cpus "$(nproc)" -e GPU_ARCHS=gfx950 -e AITER_ROCM_ARCH=gfx950 \
    -e MAX_JOBS="$(nproc)" -e AITER_MAX_JOBS="$(nproc)" -v "$OUT:/w" --entrypoint bash "$IMAGE" -c "
set -e
python3 - <<'PY'
import inspect, time
from aiter.jit import core
d = core.get_args_of_build('$MOD')
params = inspect.signature(core.build_module).parameters
args = {k: v for k, v in d.items() if k in params}
args['md_name'] = '$MOD'
print('build args:', sorted(args), flush=True)
t = time.time()
core.build_module(**args)
print(f'built in {time.time() - t:.0f} s', flush=True)
PY
ls -la $SP/aiter/jit/$MOD.so
cp $SP/aiter/jit/$MOD.so /w/$MOD.so
md5sum /w/$MOD.so
" > "$OUT/build.log" 2>&1
rc=$?
log "build rc=$rc in $(( $(date +%s) - t0 )) s: $(tail -2 "$OUT/build.log" | tr '\n' ' ' | cut -c1-200)"
[[ -s "$OUT/$MOD.so" ]] && echo "$(date -u +%m%d-%T) fp8 tune module built: $OUT/$MOD.so" >> $R/logs/summary.out \
    || echo "$(date -u +%m%d-%T) fp8 tune module build failed (rc $rc): $OUT/build.log" >> $R/logs/summary.out
