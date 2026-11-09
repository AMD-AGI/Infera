#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109: two GPU checks in the production image, then exit (no idle lease):
#   sparse_u32.py      KF sparse MLA with unsigned 32-bit page offsets vs production at the TP4 / TP8 pool sizes (GPU 0)
#   kf_realw_check.py  KF MoE vs aiter on real GLM-5.2 layer-3 weights, TP4 slice (GPU 1); aiter run-to-run, TP8 (GPU 2)
B=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench
K=/perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit
cleanup() { docker rm -f llying-kopt-bench >/dev/null 2>&1; }
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
echo "$(date -u +%FT%TZ) start on $(hostname)"
docker rm -f llying-kopt-bench >/dev/null 2>&1
docker run -d --name llying-kopt-bench --network host --ipc host --shm-size 32g --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 -v /perf_apps:/perf_apps -v $K/overlay:/yihou_overlay:ro \
  -v llying-kopt-cache:/root/.cache -v llying-kopt-aiter-jit:/sgl-workspace/aiter/aiter/jit \
  --entrypoint sleep infera-sglang:v0522-fdbafb60 infinity >/dev/null || exit 1
run() { docker exec -e HIP_VISIBLE_DEVICES=$1 -e HIP_FORCE_DEV_KERNARG=1 -e DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 \
          llying-kopt-bench bash -lc "cd $B && timeout 1200 python3 $2" 2>&1 | grep -v '^\[aiter\]' > $B/out/$3.log; }
run 0 sparse_u32.py sparse_u32 &
run 1 "kf_realw_check.py 4" kf_realw_tp4 &
run 2 "kf_realw_check.py 8" kf_realw_tp8 &
wait
echo "$(date -u +%FT%TZ) done"; tail -n 8 $B/out/sparse_u32.log $B/out/kf_realw_tp4.log $B/out/kf_realw_tp8.log
