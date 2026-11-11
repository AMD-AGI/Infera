#!/usr/bin/env bash
# MegaMoeTP GEMM2 column-group widths, confirmation: aiter's own q 3 ([3,3,3,2,1], env only) vs q 2 (measured twice for noise),
# coarser tapers; TP8 (mt 4, npp 2) at 256-768 tokens and TP4 (mt 2, npp 2, lb_min 64) at 128-384. Microbench only.
B=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench
O=$B/out/megatp_cgw2.log
C=llying-kopt-bench
cleanup() { docker rm -f $C >/dev/null 2>&1; kill "$g" 2>/dev/null; }
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
echo "$(date -u +%FT%TZ) start on $(hostname)" > $O
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
docker exec $C python3 $B/megatp_cgw_patch.py >> $O 2>&1
docker exec -e HSA_ENABLE_IPC_MODE_LEGACY=1 $C bash -lc \
  "cd $B && timeout 1800 python3 -m torch.distributed.run --standalone --nproc_per_node=8 megatp_cgw.py --out out/megatp_cgw2.jsonl --mt 4 --npp 2 --tokens 256,384,512,640,768 --variants 'q2_a:2:;q3_native:3:;3322:2:3,3,2,2,1,1;44211:2:4,4,2,1,1;43221:2:4,3,2,2,1;333111:2:3,3,3,1,1,1;6321:2:6,3,2,1;4431:2:4,4,3,1;q2_b:2:'" \
  2>&1 | grep -v "^\[aiter\]" >> $O
docker exec -e HSA_ENABLE_IPC_MODE_LEGACY=1 -e HIP_VISIBLE_DEVICES=0,1,2,3 $C bash -lc \
  "cd $B && timeout 1800 python3 -m torch.distributed.run --standalone --nproc_per_node=4 megatp_cgw.py --out out/megatp_cgw2.jsonl --mt 2 --npp 2 --lb-min 64 --tokens 128,192,256,384 --variants 'q2_a:2:;q3_native:3:;3322:2:3,3,2,2,1,1;422211:2:4,2,2,2,1,1;44211:2:4,4,2,1,1;q2_b:2:'" \
  2>&1 | grep -v "^\[aiter\]" >> $O
echo "$(date -u +%FT%TZ) done" >> $O
tail -30 $O
