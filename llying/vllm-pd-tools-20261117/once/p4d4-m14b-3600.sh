# m14 config at 3600 s (decode vLLM-internal DP balance, decode util 0.95): does it beat m10 (48,572) in the
# sglang-comparable metric? f2 (same plus util 0.93) was 8% below m14 over the first 1800 s on v3-124.
# Run by run_once.sbatch (sbatch --export=ALL,ONCE=p4d4-m14b-3600 run_once.sbatch); analyses land in results/m14b/.
tag=m14b-v3comm-dplb-dutil095-hostalloc-3600
OV=$R/overlay-cpupin/vllm/v1/kv_offload/cpu/gpu_worker.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/cpu/gpu_worker.py:ro
( bash scripts/memguard.sh "p4d4-$tag" 120 ) &
D_MODE=v3comm P_GEM=1 P_UTIL=0.88 D_UTIL=0.95 RENDER_TIMEOUT=600 P_RENDER=4 D_RENDER=4 DURATION=3600 USAGE_LOG=1 \
    CPU_BYTES=1900000000000 D_ENV_X='INFERA_VLLM_DP_ENGINE_LB=1' P_MOUNT_X="$OV" bash scripts/snap.sh p4d4_run.sh "$tag"
echo $? > "logs/once-$JOB.rc"
bash scripts/finish_round.sh "logs/once-$JOB.rc" "p4d4-$tag" m14b
wait
