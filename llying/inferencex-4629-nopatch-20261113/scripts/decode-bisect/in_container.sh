#!/usr/bin/env bash
# Runs inside one pyxis container: optional SGLang patches, then the 4+4 C80 decode worker's
# server args as a plain (non-PD) server on 4 GPUs, then load.py drives a fixed long-context
# decode load, measures steady-state step time and takes a torch profile.
#   usage: in_container.sh <name> <patchset: nt4|b8|none> <gpus e.g. 0,1,2,3> <port>
set -uo pipefail
NAME=$1 PATCHSET=$2 GPUS=$3 PORT=$4
OUT=/work/out/$NAME
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
say() { echo "$(date -u +%FT%TZ) [$NAME] $*"; }
say "host $(hostname) gpus $GPUS port $PORT patchset $PATCHSET amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null)"
git -c safe.directory='*' -C /sgl-workspace/sglang log -1 --format='sglang %H %cd' 2>/dev/null
python3 - <<'PY'
import importlib.metadata as md
for p in ("torch", "triton", "amd-aiter", "aiter", "sglang", "sgl-kernel", "amd-mori"):
    try: print(f"  {p} {md.version(p)}")
    except Exception: pass
PY
pip list 2>/dev/null | grep -iE "aiter|sgl.amd|flydsl" | sed 's/^/  /'

case "$PATCHSET" in
    nt4) bash /work/patches/apply.sh || { say "patches failed"; exit 1; } ;;
    b8) python3 /work/patches/patch_mla_absorb_bmm_default_config.py || { say "B8 failed"; exit 1; } ;;
    none) ;;
    *) say "unknown patchset $PATCHSET"; exit 1 ;;
esac

export HIP_VISIBLE_DEVICES=$GPUS
# The host home is mounted at /root on a network FS; concurrent JIT builds there fail to rename.
export HOME=/tmp/home-$NAME SGLANG_JIT_CACHE_DIR=/tmp/sgljit-$NAME TRITON_CACHE_DIR=/tmp/triton-$NAME
mkdir -p "$HOME" "$SGLANG_JIT_CACHE_DIR" "$TRITON_CACHE_DIR"
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export HSA_NO_SCRATCH_RECLAIM=1 HIP_FORCE_DEV_KERNARG=1 AITER_USE_FLYDSL_MOE_SORTING=1 SAFETENSORS_FAST_GPU=1
export SGLANG_DP_USE_GATHERV=1 SGLANG_OPT_USE_TOPK_V2=true SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=0
export SGLANG_SIMULATE_ACC_LEN=3.61 SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
export SGLANG_TORCH_PROFILER_DIR="$OUT/profile"

say "launching server"
setsid python3 -m sglang.launch_server \
    --model-path /model --served-model-name amd/GLM-5.2-MXFP4 --trust-remote-code \
    --host 127.0.0.1 --port "$PORT" --nccl-port $((PORT + 500)) \
    --tp-size 4 --dp-size 4 --enable-dp-attention --ep-size 1 \
    --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton --dsa-decode-backend triton \
    --json-model-override-args '{"index_share_for_mtp_iteration":false}' \
    --enable-fused-qk-norm-rope --enable-aiter-allreduce-fusion --disable-custom-all-reduce \
    --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
    --chunked-prefill-size 32768 --mem-fraction-static 0.92 --max-running-requests 256 \
    --cuda-graph-max-bs-decode 256 --disable-radix-cache \
    --enable-metrics --watchdog-timeout 3600 \
    > "$OUT/server.log" 2>&1 &
SPID=$!
cleanup() { say "stopping server (pgid $SPID)"; kill -TERM -- -"$SPID" 2>/dev/null; sleep 20; kill -KILL -- -"$SPID" 2>/dev/null; }
trap cleanup EXIT

for _ in $(seq 1 240); do
    grep -q "fired up and ready to roll" "$OUT/server.log" 2>/dev/null && break
    kill -0 "$SPID" 2>/dev/null || { say "server exited during startup"; tail -40 "$OUT/server.log"; exit 1; }
    sleep 15
done
grep -q "fired up and ready to roll" "$OUT/server.log" || { say "server not ready after 60 min"; exit 1; }
say "server ready"

python3 /work/load.py --port "$PORT" --out "$OUT" --log "$OUT/server.log"
rc=$?
say "load.py rc=$rc"
exit $rc
