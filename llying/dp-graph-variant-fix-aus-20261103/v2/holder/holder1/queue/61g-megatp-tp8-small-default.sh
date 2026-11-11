#!/usr/bin/env bash
# KERNEL-PLAN-20261109 7.4 M2, TP8 small batches (lb_min 256): global tokens 128 / 192 / 256 (per-rank batch 4 / 6 / 8)
# fall below the default lb_min 256 (dynamic schedule); at TP4 lb_min 64 saved 54-64 us there (61e). mt 2 / 4, npp 2, q 2.
B=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench
O=$B/out/megatp_sweep_tp8small_default.log
C=llying-kopt-bench
cleanup() { docker rm -f $C >/dev/null 2>&1; kill "$g" 2>/dev/null; }
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
echo "$(date -u +%FT%TZ) start on $(hostname)" >> $O
srun --jobid="$SLURM_JOB_ID" --overlap -N1 -n1 -w "$(hostname -s)" --export=NONE \
    /bin/bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/guard_containers.sh llying-kopt- \
    >> /perf_apps/liyingli/vfix-logs/opt/guard.log 2>&1 &
g=$!
for i in $(seq 30); do
    rocm-smi --showmemuse 2>/dev/null | awk '/VRAM%/ {if ($NF + 0 > 50) b = 1} END {exit b}' && break
    echo "$(date -u +%T) waiting for VRAM" >> $O; sleep 60
done
docker rm -f $C >/dev/null 2>&1
docker run -d --name $C --network host --ipc host --shm-size 32g --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 -v /perf_apps:/perf_apps \
  -v llying-kopt-cache:/root/.cache -v llying-kopt-aiter-jit:/sgl-workspace/aiter/aiter/jit \
  --entrypoint sleep infera-sglang:v0522-fdbafb60 infinity >/dev/null || exit 1
docker exec -e HSA_ENABLE_IPC_MODE_LEGACY=1 -e AITER_MEGAMOE_TP_LB_MIN=256 $C bash -lc \
  "cd $B && timeout 2400 python3 -m torch.distributed.run --standalone --nproc_per_node=8 megatp_sweep.py --out out/megatp_sweep_tp8small_default.jsonl --tokens 128,192,256 --mt 2,4 --npp 2 --q 2" \
  2>&1 | grep -v "^\[aiter\]" | grep -E "^\{|tokens |Error|error|Traceback" >> $O
echo "$(date -u +%FT%TZ) done" >> $O
tail -25 $O
