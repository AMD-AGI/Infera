#!/usr/bin/env bash
# Reference arm for "model or our stack": one aggregated SGLang server on GPUS of THIS node with none of the E10
# optimizations (no PD, no DP attention, no MTP, no KF kernels / tuned GEMM overlay, no prefill patches, no HiCache,
# custom all-reduce and QuickReduce off, default KV dtype and DSA backends), then the same probes as the verification
# (gsm8k_probe.py on the same 1,319 prompts, longctx_probe.py 16k x 128 with the same seed), then tear-down.
# The parent of MODEL is mounted: HF-style checkpoints link their shards into ../models--*/blobs.
# Private IPC namespace and bridge network: no RCCL shared-memory or port clash with the kit containers (--ipc host).
# Summary lines go to $OUT/ref.out (same format as verify.out lines, so probe_stats.py / the comparison read them).
# Usage (on the node): ref_run.sh NAME GPUS MODEL [IMAGE] [-- extra server args]
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG=/perf_apps/liyingli/bench_agentx/vfix-20261103
OUT=${OUT:-/perf_apps/liyingli/vfix-logs/ref}
name="$1" gpus="$2" model="$3" image="${4:-infera-sglang:v0522-fdbafb60}"
shift 3; [[ $# -gt 0 && "$1" != -- ]] && shift; [[ "${1:-}" == -- ]] && shift
extra=("$@")
port="${PORT:-31000}"
ctr="vfixref-$name"
mkdir -p "$OUT"
log="$OUT/$name.log"
say() { echo "$(date -u +%T) ref-$name $*" | tee -a "$OUT/ref.out"; }
prompts="${PROBE_PROMPTS:-/perf_apps/liyingli/vfix-logs/gsm8k_probe_prompts.jsonl}"
n_gpus=$(($(tr -cd , <<<"$gpus" | wc -c) + 1))
image_id="$(docker image inspect -f '{{.Id}}' "$image")"
docker rm -f "$ctr" >/dev/null 2>&1
args=(--model-path "$model" --served-model-name glm5.2-mxfp4 --trust-remote-code --tp-size "$n_gpus"
    --host 0.0.0.0 --port "$port" --reasoning-parser glm45 --disable-custom-all-reduce
    --mem-fraction-static "${MEM_FRAC:-0.85}" --max-running-requests 64 --chunked-prefill-size 16384
    --watchdog-timeout 3600 "${extra[@]}")
say "up: $(hostname -s) GPUs $gpus image $image (${image_id:7:12}) model $model args ${args[*]}"
docker run -d --name "$ctr" -p "$port:$port" --shm-size 64g \
    --device /dev/kfd --device /dev/dri --group-add video --group-add render \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
    -v "$(dirname "$model"):$(dirname "$model"):ro" -v "$ctr-cache-${image_id:7:12}:/root/.cache" -v "$ctr-aiter-jit-${image_id:7:12}:/sgl-workspace/aiter/aiter/jit" \
    -e HIP_VISIBLE_DEVICES="$gpus" -e NCCL_IB_DISABLE=1 -e ROCM_QUICK_REDUCE_QUANTIZATION=NONE -e PYTHONHASHSEED=0 \
    -e SAFETENSORS_FAST_GPU=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
    --entrypoint python3 "$image" -m sglang.launch_server "${args[@]}" > /dev/null || { say "docker run failed"; exit 1; }
url="http://127.0.0.1:$port"
t0=$SECONDS
until curl -sf -m 5 "$url/health" >/dev/null; do
    if ! docker inspect -f '{{.State.Running}}' "$ctr" 2>/dev/null | grep -q true || ((SECONDS - t0 > 3600)); then
        docker logs "$ctr" > "$log" 2>&1
        say "did not come up: $(grep -aE 'Error|error' "$log" | tail -1 | cut -c1-200)"
        docker rm -f "$ctr" >/dev/null 2>&1; exit 1
    fi
    sleep 20
done
say "ready after $((SECONDS - t0)) s; env QR=$(docker exec "$ctr" printenv ROCM_QUICK_REDUCE_QUANTIZATION) dual_graph_lines=$(docker logs "$ctr" 2>&1 | grep -ac 'DSA dual-graph enabled')"
python3 "$PKG/scripts/gsm8k_probe.py" "$prompts" "$url" glm5.2-mxfp4 "$OUT/$name.probe.jsonl" --limit "${PROBE_N:-1319}" --concurrency 64 >> "$log.client" 2>&1
say "probe: $(python3 "$PKG/scripts/probe_stats.py" "$OUT/$name.probe.jsonl" --review "$OUT/$name.review.txt")"
k=0
for spec in ${LONGCTX_SPECS-16000:128}; do
    k=$((k + 1))
    python3 "$PKG/scripts/longctx_probe.py" "$prompts" "$url" glm5.2-mxfp4 "$OUT/$name.longctx-$k-${spec%%:*}.jsonl" \
        --ctx-tokens "${spec%%:*}" --n "${spec##*:}" --concurrency 32 >> "$log.client" 2>&1
    say "longctx $spec: $(grep -aE '^(needle|gsm8k): n=' "$log.client" | tail -2 | tr '\n' ' ')"
done
docker logs "$ctr" > "$log" 2>&1
docker rm -f "$ctr" >/dev/null 2>&1
say "down"
