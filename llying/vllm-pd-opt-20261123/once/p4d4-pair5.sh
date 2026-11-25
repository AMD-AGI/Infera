# Paired P4D4 A/B in one job on one node (node-to-node spread is 7-10%, so a change is compared with its own baseline):
# side B (gated first if its env sets S_GATE=1, then scored), then side A (score). Each side is configured by
# $R/once/score-$SCORE_ENV-A.env / -B.env, read right before that side runs. Variables (as once/p4d4-score2.sh):
#   S_TAG S_LB S_KV_W S_GEMM S_FP4 S_D_ARGS S_D_ENV S_TRACE S_DURATION, plus
#   S_TOPK  1 = decode top-k overlay (scripts/make_tritontopk_overlay.py), S_TOPK_IMPL triton (default) or
#           native (vLLM's in-tree kernel instead of aiter's one-block one)                              default 0
#           (that file is already overlaid by D_MODE=v3comm, so such a side runs D_MODE=stock with v3comm's three
#           mounts and env reproduced, its ops file replaced by the patched one built from v3comm's)
#   S_GATE  1 = gate this side before scoring (real acceptance)                                          default 0
#   S_GEMM_DIR  directory from scripts/merge_tuned_gemm2.py (mounts.txt: tables to mount); overrides S_GEMM
#   S_P_ARGS / S_P_ENV  extra prefill vLLM args / env (appended, so they override)
#   S_SCORE 0 = gate only (with S_GATE=1), e.g. PAIR_SIDES="A1 B1 A2 B2" to compare real acceptance          default 1
#   S_PPSCHED 1 = mount the PP prefill scheduling overlay (scripts/make_ppsched_overlay.py, vLLM #68614 + #68560
#           backported to v0.34.0) on the prefill; its switches go in S_P_ENV (VPD_PP_ADMIT=1,
#           VPD_PP_LONE_THRESHOLD=1, VPD_PP_MICROBATCH_MIN_TOKENS=N), all unset = v0.34.0                default 0
#   S_KVOFF 1 = mount the CPU-tier scheduler-thread overlay (scripts/make_kvoffload_overlay.py: #69540 LRU without
#           heap rebuilds, #68845 coalesced hit ranges) on the prefill                                    default 0
# PAIR_SIDES (default "B A") picks the sides and their order, e.g. PAIR_SIDES=B to add a variant next to an existing
# baseline on the same node. Results in $B/results/$SCORE_ENV/{A,B}.
# Run by run_once.sbatch: --export=ALL,ONCE=p4d4-pair5,SCORE_ENV=<name>[,PAIR_SIDES=B].
# (p4d4-pair.sh is the first version, kept as a10 ran it: its top-k side collided with v3comm's mount.)
# p4d4-pair3.sh = pair2b + prefill args / env per side (pair2 / pair2b kept unchanged while a16 / a18 ran them).
# p4d4-pair5.sh = pair3 + S_PPSCHED / S_KVOFF (pair3 kept unchanged while a31 ran it).
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}   # run_once.sbatch does not export R
cd "$R" || exit 0
B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123
KF=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-decode-trackA-20261118/scripts/kf
SE=${SCORE_ENV:?SCORE_ENV}
OO=$B/results/$SE
IMAGE=infera-vllm:v0340-0c5e5e89
SP=/usr/local/lib/python3.12/dist-packages
mkdir -p "$OO"
cp "$0" "$OO/once.sh" 2>/dev/null

gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
free=$(timeout 300 docker run --rm --network none --device /dev/kfd --device /dev/dri "${gids[@]}" --entrypoint python3 "$IMAGE" \
    -c 'import torch; print(" ".join(str(int(torch.cuda.mem_get_info(i)[0] / 2**30)) for i in range(torch.cuda.device_count())))' 2>/dev/null | tail -1)
echo "$(date -u +%T) $(hostname -s) GPU free GiB: ${free:-unknown}; disk: $(df -h / /var/lib/docker 2>/dev/null | tail -n +2 | awk '{print $6": "$4" free"}' | paste -sd' ')" | tee "$OO/gpu-free-at-start.txt"
for x in $free; do
    if (( x < 250 )); then echo "$(date -u +%T) $SE: NOT STARTED, a GPU has only $x GiB free" | tee -a logs/summary.out; exit 0; fi
done

ov=$R/runs/$SE-overlays
rm -rf "$ov"; mkdir -p "$ov" && chmod 777 "$ov"
MLA=$SP/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py
CC=$SP/vllm/v1/engine/core_client.py
NDP=$SP/vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py
RAM=$SP/vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py
OPS=$SP/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
V3=$R/overlay-v3comm/vllm
cp "$B/scripts/vpd_triton_topk_decode.py" "$ov/vpd_triton_topk_decode.py"   # tie-guarded copy of the A-line port
cp "$V3/v1/attention/ops/rocm_aiter_mla_sparse.py" "$ov/v3comm_ops_rocm_aiter_mla_sparse.py"
docker run --rm --network none --device /dev/kfd --device /dev/dri "${gids[@]}" -e HIP_VISIBLE_DEVICES=0 -v "$B:$B:ro" -v "$ov:$ov" --entrypoint bash "$IMAGE" -c \
    "md5sum $MLA $CC $NDP $RAM $OPS && python3 $B/scripts/make_syncskip_overlay.py $MLA $ov/rocm_aiter_mla_sparse.py && python3 $B/scripts/test_syncskip.py $ov/rocm_aiter_mla_sparse.py \
     && python3 $B/scripts/make_dplb_kv_overlay.py $CC $ov/core_client.py && VPD_DP_LB_KV_WEIGHT=14 python3 $B/scripts/test_dplb_kv.py $ov/core_client.py \
     && python3 $B/scripts/make_fp4dispatch_overlay.py $NDP $ov/naive_dp_ep.py $RAM $ov/rocm_aiter_moe.py && python3 $B/scripts/test_fp4dispatch.py $ov/naive_dp_ep.py $ov/rocm_aiter_moe.py \
     && mkdir -p $ov/ops && python3 $B/scripts/make_tritontopk_overlay.py $ov/v3comm_ops_rocm_aiter_mla_sparse.py $ov/ops/rocm_aiter_mla_sparse.py \
     && PYTHONPATH=$ov VPD_TRITON_TOPK=1 python3 $B/scripts/test_tritontopk.py $ov/ops/rocm_aiter_mla_sparse.py" \
    > "$OO/overlay-build-test.txt" 2>&1 || { echo "$(date -u +%T) $SE: overlay build / test failed"; exit 0; }
# PP prefill scheduling overlay: built from the image's files, then tested with it mounted over them (CPU platform,
# no GPU needed): switches off = v0.34.0, on = the upstream tests of #68614 / #68560.
mkdir -p "$ov/ppsched" && chmod 777 "$ov/ppsched"
PPS="$ov/ppsched/scheduler.py:$SP/vllm/v1/core/sched/scheduler.py:ro $ov/ppsched/interface.py:$SP/vllm/v1/core/sched/interface.py:ro"
PPS+=" $ov/ppsched/kv_cache_manager.py:$SP/vllm/v1/core/kv_cache_manager.py:ro $ov/ppsched/core.py:$SP/vllm/v1/engine/core.py:ro"
pps_v=(); for m in $PPS; do pps_v+=(-v "$m"); done
{ docker run --rm --network none -v "$B:$B:ro" -v "$ov:$ov" --entrypoint python3 "$IMAGE" \
      "$B/scripts/make_ppsched_overlay.py" "$SP" "$ov/ppsched" \
  && docker run --rm --network none -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e VLLM_TARGET_DEVICE=cpu -v "$B:$B:ro" "${pps_v[@]}" \
      --entrypoint bash "$IMAGE" -c "cd /tmp && python3 $B/scripts/test_ppsched.py off && python3 $B/scripts/test_ppsched.py on"; } \
    > "$OO/ppsched-build-test.txt" 2>&1 || { echo "$(date -u +%T) $SE: overlay build / test failed (ppsched)"; exit 0; }
# CPU-tier overlay (#69540 + #68845), built and tested the same way.
mkdir -p "$ov/kvoff" && chmod 777 "$ov/kvoff"
KVO="$ov/kvoff/lru.py:$SP/vllm/v1/kv_offload/cpu/policies/lru.py:ro"
KVO+=" $ov/kvoff/offloading_scheduler.py:$SP/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:ro"
kvo_v=(); for m in $KVO; do kvo_v+=(-v "$m"); done
{ docker run --rm --network none -v "$B:$B:ro" -v "$ov:$ov" --entrypoint python3 "$IMAGE" \
      "$B/scripts/make_kvoffload_overlay.py" "$SP" "$B/overlay-kvoffload" "$ov/kvoff" \
  && docker run --rm --network none -e HOME=/tmp -e VLLM_TARGET_DEVICE=cpu -v "$B:$B:ro" "${kvo_v[@]}" \
      --entrypoint bash "$IMAGE" -c "cd /tmp && python3 $B/scripts/test_kvoffload.py $B/overlay-kvoffload"; } \
    > "$OO/kvoff-build-test.txt" 2>&1 || { echo "$(date -u +%T) $SE: overlay build / test failed (kvoff)"; exit 0; }

OV_PIN=$R/overlay-cpupin/vllm/v1/kv_offload/cpu/gpu_worker.py:$SP/vllm/v1/kv_offload/cpu/gpu_worker.py:ro
export D_MODE=v3comm P_GEM=1 P_UTIL=0.88 D_UTIL=0.95 RENDER_TIMEOUT=600 P_RENDER=4 D_RENDER=4 P_MOUNT_X="$OV_PIN"
export D_PREFIX_CACHE=1
gemm=$R/runs/a4-gemmtune/glm52_decode_bf16_tuned_gemm.csv
[ -s "$gemm" ] || gemm=$B/gemm-tune/glm52_decode_bf16_tuned_gemm.csv

side() {   # side A|B, optionally with a repetition suffix (A1, B2, ...): env from score-$SE-<letter>.env
    local t=$1 s=${1:0:1} O=$OO/$1
    mkdir -p "$O"
    for i in $(seq 60); do
        shm=$(awk '/^Shmem:/ {print int($2 / 1048576)}' /proc/meminfo); (( shm < 100 )) && break; sleep 10
    done
    S_TAG=$SE-$s S_LB=1 S_KV_W=0 S_GEMM=0 S_GEMM_DIR="" S_FP4=0 S_TOPK=0 S_TOPK_IMPL="" S_GATE=0 S_SCORE=1 S_D_ARGS="" S_D_ENV="" S_P_ARGS="" S_P_ENV="" S_PPSCHED=0 S_KVOFF=0 S_TRACE=1 S_DURATION=3600
    [ -r "$R/once/score-$SE-$s.env" ] && . "$R/once/score-$SE-$s.env" && cp "$R/once/score-$SE-$s.env" "$O/"
    [[ "$t" != "$s" ]] && S_TAG+="-r${t:1}"
    local PM="$OV_PIN"
    [[ "$S_PPSCHED" == 1 ]] && PM+=" $PPS"
    [[ "$S_KVOFF" == 1 ]] && PM+=" $KVO"
    local OV="$ov/rocm_aiter_mla_sparse.py:$MLA:ro" denv="VPD_STEPLOG_DIR=/steplog"
    [[ "$S_LB" == 1 ]] && denv+=" INFERA_VLLM_DP_ENGINE_LB=1"
    if [[ "$S_KV_W" != 0 ]]; then OV+=" $ov/core_client.py:$CC:ro"; denv+=" VPD_DP_LB_KV_WEIGHT=$S_KV_W"; fi
    if [[ "$S_FP4" == 1 ]]; then OV+=" $ov/naive_dp_ep.py:$NDP:ro $ov/rocm_aiter_moe.py:$RAM:ro"; denv+=" VPD_FP4_DISPATCH=1"; fi
    local dmode=$D_MODE
    if [[ "$S_TOPK" == 1 ]]; then
        dmode=stock
        OV+=" $ov/ops/rocm_aiter_mla_sparse.py:$OPS:ro $ov/vpd_triton_topk_decode.py:$SP/vpd_triton_topk_decode.py:ro"
        OV+=" $V3/v1/attention/ops/rocm_gluon_paged_mqa_logits.py:$SP/vllm/v1/attention/ops/rocm_gluon_paged_mqa_logits.py:ro"
        OV+=" $V3/distributed/device_communicators/cuda_communicator.py:$SP/vllm/distributed/device_communicators/cuda_communicator.py:ro"
        denv+=" VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1 VPD_TOPK_IMPL=${S_TOPK_IMPL:-triton}"
    fi
    if [[ -n "$S_GEMM_DIR" && -s "$S_GEMM_DIR/mounts.txt" ]]; then
        while IFS=: read -r src dst; do OV+=" $S_GEMM_DIR/$src:$dst:ro"; done < "$S_GEMM_DIR/mounts.txt"
    elif [[ "$S_GEMM" == 1 && -s "$gemm" ]]; then
        cp "$gemm" "$ov/"; OV+=" $ov/glm52_decode_bf16_tuned_gemm.csv:$SP/aiter/configs/model_configs/glm52_bf16_tuned_gemm.csv:ro"
    fi
    denv+=" $S_D_ENV"
    echo "$(date -u +%T) $SE-$s: tag $S_TAG; LB $S_LB; KV $S_KV_W; FP4 $S_FP4; TOPK $S_TOPK; gate $S_GATE; gemm $([[ "$OV" == *glm52_bf16_tuned_gemm* ]] && echo on || echo off); D_MODE $dmode; decode args [$S_D_ARGS]; env [$denv]; prefill args [$S_P_ARGS]; prefill env [$S_P_ENV]; ppsched $S_PPSCHED; kvoff $S_KVOFF" | tee "$O/config.txt"
    if [[ "$S_GATE" == 1 ]]; then
        local gtag=$S_TAG-gate
        D_MODE=$dmode CPU_BYTES=200000000000 D_MOUNT_X="$OV" D_ENV_X="${denv/VPD_STEPLOG_DIR=\/steplog/}" D_ARGS_X="$S_D_ARGS" P_ARGS_X="$S_P_ARGS" P_ENV_X="$S_P_ENV" P_MOUNT_X="$PM" P4D4_STEP=gate_p4d4.sh \
            bash scripts/snap.sh p4d4_run.sh "$gtag" > "$O/gate-run.log" 2>&1
        local line; line=$(grep -a "gate p4d4-$gtag:" logs/summary.out | tail -1)
        echo "$line" > "$O/gate.txt"
        cp results/gate-p4d4-$gtag/{smoke.out,probe-stats.out,longctx.out,metrics-delta.json,probe-review.txt} "$O/" 2>/dev/null
        local ok; ok=$(python3 "$B/scripts/gate_ok.py" "$line")
        echo "$(date -u +%T) $SE-$s gate clean: $ok" | tee -a "$O/gate.txt"
        [[ "$ok" == yes ]] || return 1
        [[ "$S_SCORE" == 1 ]] || return 0
        for i in $(seq 60); do
            shm=$(awk '/^Shmem:/ {print int($2 / 1048576)}' /proc/meminfo); (( shm < 100 )) && break; sleep 10
        done
    fi
    local run=p4d4-$S_TAG D=$R/runs/p4d4-$S_TAG
    mkdir -p "$D/steplog" "$D/instr" && chmod 777 "$D/steplog"
    cp "$B/instr/vpd_steplog.py" "$B/instr/vpd_steplog.pth" "$D/instr/"
    local DM="$OV $D/instr/vpd_steplog.py:$SP/vpd_steplog.py:ro $D/instr/vpd_steplog.pth:$SP/vpd_steplog.pth:ro $D/steplog:/steplog"
    ( bash scripts/memguard.sh "$run" 120 ) &
    D_MODE=$dmode CPU_BYTES=1900000000000 DURATION=$S_DURATION USAGE_LOG=1 TRACE=$S_TRACE TRACE_AT=1500 \
        D_ARGS_X="$S_D_ARGS" D_ENV_X="$denv" D_MOUNT_X="$DM" P_ARGS_X="$S_P_ARGS" P_ENV_X="$S_P_ENV" P_MOUNT_X="$PM" \
        bash scripts/snap.sh p4d4_run.sh "$S_TAG"
    echo $? > "logs/once-$JOB.rc"
    bash scripts/finish_round.sh "logs/once-$JOB.rc" "$run" "$SE-$t"
    zcat -f "$D"/vpd-decode*.log* 2>/dev/null | grep -a -E "GPU KV cache size" | head -1 > "$O/decode-kv-size.txt"
    python3 "$B/scripts/steplog_summary.py" "$D/steplog" "$D/agentx-c104/benchmark.log" 300 > "$O/steplog_summary.txt" 2>&1
    python3 "$B/scripts/dp_imbalance.py" "$D/steplog" > "$O/dp_imbalance.txt" 2>&1
    local f1; f1=$(find "$D/trace" -name "dp1_*.json*" -type f | head -1)
    [ -n "$f1" ] && python3 scripts/trace_cat.py "$f1" --top 40 > "$O/trace-cat-dp1.txt" 2>&1
    [ -n "$f1" ] && python3 "$B/scripts/kernel_dist.py" "$f1" 'radix_topk|_topk_est|_topk_collect|_topk_select' 'allgather_vecIDF16b' 'reduce_scatter_split_first_dim' > "$O/kernel-dist-dp1.txt" 2>&1
    find "$D/trace" -name "dp*.json*" -type f | sort | head -1 | while read -r f; do
        python3 "$B/scripts/trace_gaps.py" "$f" --step-kernel _combine_sampled_and_draft_tokens_kernel --top 8
    done > "$O/trace-gaps.txt" 2>&1
    cp "$B/../vllm-p4d4-crusoe-20261117/results/$SE-$t/"{summary.txt,window_tput.txt,samples_summary.txt,decode_pressure.txt,dp_balance.txt,health.txt,score.txt,ttft_breakdown.txt} "$O/" 2>/dev/null
    gzip -k -f "$D"/steplog/steps-*.csv 2>/dev/null
    wait
}

# B first by default: its gate gives the correctness answer early; A (the baseline on this node) runs either way.
for s in ${PAIR_SIDES:-B A}; do side "$s"; done
echo "$(date -u +%m%d-%T) $SE pair analyses in $OO" >> logs/summary.out
