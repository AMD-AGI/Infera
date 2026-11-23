# Phase 0b of ../PLAN (vllm-pd-opt-plan-20261123): decode-only harness. Two DP4 decode instances with the P4D4 decode
# config (v3comm overlays, MTP3 synthetic 2.99, fp8 KV, util 0.95, no prefix caching) on GPUs 0-3 (A) and 4-7 (B), KV
# connector DecodeBenchConnector (dummy prompt KV, the last prompt token computed on the decode, like a PD decode after a
# KV pull; preempted requests recompute in full), both fed the same open-loop replay of m10's decode arrivals
# (data/m10-decode-arrivals.jsonl.gz, trace 600-2100 s). vpd_steplog in every decode process.
#   r1  A = B = baseline (harness noise; step classes under a controlled load)
#   r2  A = baseline, B = baseline + --max-num-batched-tokens 2048 (KV gain and stability of a smaller activation reserve)
# Run by run_once.sbatch (ONCE=db-0b-aa-mnbt); results in llying/vllm-pd-opt-20261123/results/b0/.
B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123
O=$B/results/b0
IMAGE=infera-vllm:v0340-0c5e5e89
MODEL=/mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4
SP=/usr/local/lib/python3.12/dist-packages
V=$SP/vllm
TRACE_START=${TRACE_START:-600} TRACE_END=${TRACE_END:-2100}
mkdir -p "$O"
cp "$0" "$O/once.sh" 2>/dev/null

gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
free=$(timeout 300 docker run --rm --network none --device /dev/kfd --device /dev/dri "${gids[@]}" --entrypoint python3 "$IMAGE" \
    -c 'import torch; print(" ".join(str(int(torch.cuda.mem_get_info(i)[0] / 2**30)) for i in range(torch.cuda.device_count())))' 2>/dev/null | tail -1)
echo "$(date -u +%T) GPU free GiB: ${free:-unknown}" | tee "$O/gpu-free-at-start.txt"
for x in $free; do
    if (( x < 250 )); then echo "$(date -u +%T) db-0b: NOT STARTED, a GPU has only $x GiB free" | tee -a logs/summary.out; exit 0; fi
done

O3=$R/overlay-v3comm/vllm
comm_mount="$O3/v1/attention/ops/rocm_aiter_mla_sparse.py:$V/v1/attention/ops/rocm_aiter_mla_sparse.py:ro $O3/v1/attention/ops/rocm_gluon_paged_mqa_logits.py:$V/v1/attention/ops/rocm_gluon_paged_mqa_logits.py:ro $O3/distributed/device_communicators/cuda_communicator.py:$V/distributed/device_communicators/cuda_communicator.py:ro"
spec='{"method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"synthetic","synthetic_acceptance_length":2.99}'
kv='{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both","kv_connector_extra_config":{"startup_fill":true}}'
base="--served-model-name glm5.2-mxfp4 --trust-remote-code --host 0.0.0.0 --kv-cache-dtype fp8_e4m3 --data-parallel-size 4 --tensor-parallel-size 1 --no-enable-prefix-caching --max-num-seqs 64 --stream-interval 30 --block-size 64 --gpu-memory-utilization 0.95 --api-server-count 1"

up() {  # NAME GPUS PORT RPC_PORT RUN_DIR EXTRA_ARGS EXTRA_ENV
    local name=$1 gpus=$2 port=$3 rpc=$4 dir=$5 xargs=$6 xenv=$7
    mkdir -p "$dir/steplog" "$dir/instr" && chmod 777 "$dir" "$dir/steplog"
    cp "$B/instr/vpd_steplog.py" "$B/instr/vpd_steplog.pth" "$dir/instr/"
    MOUNT_EXTRA="$comm_mount $dir/instr/vpd_steplog.py:$SP/vpd_steplog.py:ro $dir/instr/vpd_steplog.pth:$SP/vpd_steplog.pth:ro $dir/steplog:/steplog" \
    ENV_EXTRA="VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1 VPD_STEPLOG_DIR=/steplog PYTHONFAULTHANDLER=1 $xenv" \
    ENTRY="exec vllm serve $MODEL --port $port --data-parallel-rpc-port $rpc $base --speculative-config '$spec' --kv-transfer-config '$kv' $xargs" \
    IMAGE=$IMAGE MODEL=$MODEL R=$R bash scripts/vllm_up.sh "$name" "$gpus" "$port" > "$dir/up.log" 2>&1
}

round() {  # TAG A_ARGS A_ENV B_ARGS B_ENV
    local tag=$1 dA=$R/runs/db-$1-A dB=$R/runs/db-$1-B
    echo "$(date -u +%T) round $tag: A [$2] [$3] / B [$4] [$5]"
    docker rm -f vpd-dbA vpd-dbB > /dev/null 2>&1; sleep 20
    rm -rf "$dA" "$dB"
    up vpd-dbA 0,1,2,3 27501 29621 "$dA" "$2" "$3" &
    up vpd-dbB 4,5,6,7 27601 29721 "$dB" "$4" "$5" &
    wait
    tail -1 "$dA/up.log"; tail -1 "$dB/up.log"
    local urls=() k
    for k in A:27501 B:27601; do
        if curl -sf -m 5 "http://127.0.0.1:${k#*:}/v1/models" > /dev/null; then urls+=("${k%%:*}"); fi
    done
    echo "$(date -u +%T) round $tag: serving ${urls[*]:-none}"
    local args=()
    for k in "${urls[@]}"; do
        local port=27501 d=$dA; [[ "$k" == B ]] && port=27601 d=$dB
        args+=(--url "http://127.0.0.1:$port" --out "$d/replay.jsonl")
    done
    if (( ${#args[@]} )); then
        docker run --rm --network host -v "$B:$B" -v "$R/runs:$R/runs" --entrypoint python3 "$IMAGE" "$B/scripts/decode_replay.py" \
            --trace "$B/data/m10-decode-arrivals.jsonl.gz" --start "$TRACE_START" --end "$TRACE_END" --drain 120 "${args[@]}" \
            > "$R/runs/db-$tag-replay.log" 2>&1
        tail -2 "$R/runs/db-$tag-replay.log"
    fi
    for k in A B; do
        local d=$dA; [[ "$k" == B ]] && d=$dB
        cp "$R/logs/engines/vpd-db$k.log" "$d/engine.log" 2>/dev/null && gzip -f "$d/engine.log"
        docker inspect -f "vpd-db$k {{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}" "vpd-db$k" > "$d/container-state.txt" 2>&1
    done
    docker rm -f vpd-dbA vpd-dbB > /dev/null 2>&1
    {
        echo "== round $tag"; cat "$R/runs/db-$tag-replay.log" 2>/dev/null | tail -3
        for k in A B; do
            local d=$dA; [[ "$k" == B ]] && d=$dB
            echo; echo "==== $k: $(cat "$d/container-state.txt" 2>/dev/null)"
            zgrep -a -E "KV cache|GPU KV cache size|Maximum concurrency|vpd_steplog|HIPBLAS_STATUS|OUT_OF_RESOURCES|Traceback|Error" "$d/engine.log.gz" 2>/dev/null | sort | uniq -c | sort -rn | head -15
            [ -s "$d/replay.jsonl" ] && python3 "$B/scripts/replay_summary.py" "$d/replay.jsonl" 300
            w=$(cat "$d/replay.jsonl.window" 2>/dev/null | tr ' ' ':')
            python3 "$B/scripts/steplog_summary.py" "$d/steplog" "@${w:-0:0}" 300
        done
    } > "$O/$tag.txt" 2>&1
    gzip -f "$dA"/steplog/steps-*.csv "$dB"/steplog/steps-*.csv 2>/dev/null
    echo "$(date -u +%T) round $tag done -> $O/$tag.txt"
}

round r1-aa "--max-num-batched-tokens 8192" "" "--max-num-batched-tokens 8192" ""
round r2-mnbt2048 "--max-num-batched-tokens 8192" "" "--max-num-batched-tokens 2048" ""
echo "$(date -u +%m%d-%T) db-0b rounds done: $O" >> logs/summary.out
