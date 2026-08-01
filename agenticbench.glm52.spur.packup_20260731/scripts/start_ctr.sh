#!/bin/bash
# Start one agentic-bench container on a spur node, health-gate the GPUs, apply the
# mooncake early-send wait_event patch, and (on the prefill node) bring up etcd.
#
# Fusion of two sanctioned kits:
#   * spur mechanics  -- main_converged/scripts/start_ctr_conv.sh
#                        (spur exec, /shared_nfs mount, GPUGATE, --entrypoint '')
#   * kvd/etcd wiring -- pr.final/scripts/kvaware_start_prefill.sh
#
# Deliberate delta from pr.final: NO host-libionic bind mount. That exists for the
# vultr ionic fabric; spur's KV NIC is mlx5 and the container's own provider is
# correct. Keeping the mount here would bind a path that does not exist on spur.
#
# Usage: start_ctr.sh <job> <prefill|decode> <my-ip>
set -eu
JOB="${1:?job}"
ROLE="${2:?prefill|decode}"
MY_IP="${3:?ens3 ip}"
IMG=infera/engine-sglang:kvaware-kvd
CTR=agbench
W=/shared_nfs/yihou_agentbench

spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
set -e
docker rm -f $CTR 2>/dev/null || true

# --entrypoint '' bypasses infera-inject-host-ionic: it is a no-op without the
# bind mount, but skipping it keeps 'sleep infinity' from being re-exec'd.
docker run -d --name $CTR --network=host --ipc=host --shm-size=32G \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  -v /shared_nfs:/shared_nfs \
  --entrypoint '' $IMG sleep infinity >/dev/null
sleep 5

docker exec $CTR python3 -c 'import torch;print(\"GPUGATE\", torch.cuda.is_available(), torch.cuda.device_count())'
echo -n '  RDMA devices in container: '
docker exec $CTR bash -c 'ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE || echo 0'

# ---- mooncake early-send wait_event patch (AMD-AGI/Infera PR #56) ----
# Mandatory here: Case A prompts are 74K-235K tokens against a 65536 prefill
# chunk, so EVERY measured request is multi-chunk prefill -- the exact regime
# where mooncake RDMA-reads KV pages the writing forward has not finished. It
# never raises; long prompts just come back partially wrong.
echo '===== mooncake wait_event patch ====='
docker exec $CTR python3 $W/patches/patch_mooncake_early_send_wait_event.py

# A stale __pycache__ entry silently reverts a source patch -- that has already
# invalidated a full experiment in this repo. Drop the bytecode and re-verify
# from a fresh import, not from the .py.
docker exec $CTR bash -c 'find /sgl-workspace/sglang/python/sglang/srt/disaggregation -name \"*.pyc\" -delete'
docker exec $CTR python3 -c '
import importlib, subprocess, sys
import sglang.srt.disaggregation.mooncake.conn as m
import sglang.srt.disaggregation.common.utils as u
import sglang.srt.disaggregation.prefill as p
ok = True
for mod, want in ((m, \"wait_event\"), (u, \"wait_event\"), (p, \"wait_event\")):
    src = open(mod.__file__).read()
    n = src.count(want)
    print(\"  %-58s %s=%d\" % (mod.__name__, want, n))
    ok = ok and n > 0
print(\"MOONCAKE_WAIT_EVENT\", \"OK\" if ok else \"MISSING\")
'
docker exec $CTR bash -c 'for f in \$(find /sgl-workspace/sglang/python/sglang/srt/disaggregation -name \"conn.cpython-*.pyc\" -path \"*mooncake*\"); do echo -n \"  bytecode \$f: \"; strings \$f | grep -c wait_event; done'
" 2>&1 | grep -v libtinfow
