#!/bin/bash
# Start one engine container on a spur node from the MERGED-BRANCH image, health
# gate the GPUs, and prove the two fixes this cluster depends on are live in
# BYTECODE (not merely in the source, and not merely "the build log said so").
#
# NO in-container patching. That is the point: every fix is baked into the image
# by the branch's Dockerfile. The predecessor agenticbench run patched a running
# container for both of these; here they must already be there.
#
# Deliberate delta from the vultr kits: NO host-libionic bind mount. That exists
# for the vultr ionic fabric; spur's KV NIC is mlx5 and the container's own
# provider is correct. Keeping the mount would bind a path that does not exist.
#
# Usage: start_ctr.sh <job> <prefill|decode>
set -eu
JOB="${1:?job}"
ROLE="${2:?prefill|decode}"
IMG="${IMG:-infera/engine-sglang:final-pr}"
CTR="${CTR:-agbench_mtp}"

spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
set -e
docker rm -f $CTR 2>/dev/null || true

# --entrypoint '' bypasses infera-inject-host-ionic: a no-op without the bind
# mount, but skipping it keeps 'sleep infinity' from being re-exec'd.
docker run -d --name $CTR --network=host --ipc=host --shm-size=32G \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  -v /shared_nfs:/shared_nfs \
  --entrypoint '' $IMG sleep infinity >/dev/null
sleep 5

echo '--- image ---'
docker image inspect $IMG --format '  id={{.Id}}'
echo '--- gpu gate ---'
docker exec $CTR python3 -c 'import torch;print(\"  GPUGATE\", torch.cuda.is_available(), torch.cuda.device_count())'
echo -n '  RDMA PORT_ACTIVE in container: '
docker exec $CTR bash -c 'ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE || echo 0'

# ---- prove the fixes reached the BYTECODE, from a FRESH compile --------------
# A stale __pycache__ entry silently running unpatched bytecode has invalidated a
# full experiment on this stack twice. Import the module (which compiles it), then
# read the .pyc -- not the .py.
echo '--- baked-in fixes, verified in bytecode ---'
docker exec $CTR python3 -c '
import importlib, subprocess, sys

def pyc_has(modname, marker):
    m = importlib.import_module(modname)          # forces a compile if needed
    import os, glob
    d = os.path.join(os.path.dirname(m.__file__), \"__pycache__\")
    hits = 0
    base = os.path.basename(m.__file__)[:-3]
    for f in glob.glob(os.path.join(d, base + \".*.pyc\")):
        hits += subprocess.run([\"strings\", f], capture_output=True, text=True).stdout.count(marker)
    return hits

checks = [
    (\"sglang.srt.mem_cache.pool_host.common\", \"GLM52_ROCM_HOST_ALLOC\", \"ROCm hicache host alloc\"),
    (\"sglang.srt.disaggregation.mooncake.conn\", \"wait_event\",           \"mooncake early-send\"),
    (\"sglang.srt.disaggregation.common.utils\", \"wait_event\",           \"mooncake early-send (utils)\"),
    (\"sglang.srt.disaggregation.prefill\",      \"wait_event\",           \"mooncake early-send (prefill)\"),
]
bad = 0
for mod, marker, label in checks:
    n = pyc_has(mod, marker)
    print(\"  %-34s %-22s pyc_hits=%d  %s\" % (label, marker, n, \"OK\" if n else \"MISSING\"))
    bad += (n == 0)

# The ROCm fix must also be BEHAVIOURALLY live: the dispatch table has to point
# at pin_memory on HIP. A marker in the bytecode proves the file was edited; this
# proves the edit does what it claims.
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS
from sglang.srt.utils import is_hip
fn = ALLOC_MEMORY_FUNCS[\"cuda\"].__name__
want = \"alloc_with_pin_memory\" if is_hip() else \"alloc_with_host_register\"
print(\"  %-34s is_hip=%s -> %s  %s\" % (\"ALLOC_MEMORY_FUNCS dispatch\", is_hip(), fn, \"OK\" if fn == want else \"WRONG\"))
bad += (fn != want)

# The three DSA patches (PD + DPA + EAGLE MTP on gfx950). Same identifier list
# the build-time verifier uses -- identifiers, never comment markers, because the
# compiler discards comments and a comment marker reads as a false negative.
# TWO markers for p1, not one. The predecessor kit checked _p1v2_trim alone and
# then patched GLM52_P1V3 into the RUNNING container by hand (its REPRODUCE.md
# step 2, \"MANDATORY\"). On this branch P1V3 is baked into the image, and
# _p1v2_trim cannot tell the two apart -- it is satisfied by the older
# one-directional revision too. _p1v2_rows is introduced only by the P1V3 edits,
# so it is what proves the reversed-padding (real > padded) case is handled and
# that NO in-container patching is needed this time.
for mod, marker, label in [
    (\"sglang.srt.layers.attention.dsa.dsa_indexer\",  \"_p1v2_trim\",                        \"DSA p1: hip dp rows\"),
    (\"sglang.srt.layers.attention.dsa.dsa_indexer\",  \"_p1v2_rows\",                        \"DSA p1v3: reversed padding\"),
    (\"sglang.srt.layers.attention.dsa_backend\",      \"_glm52_match_page_table_rows\",      \"DSA p2b: page-table rows\"),
    (\"sglang.srt.speculative.eagle_worker_v2\",       \"requires_dp_attention_eager_forward\",\"DSA p3: draft-graph DP vote\"),
]:
    n = pyc_has(mod, marker)
    print(\"  %-34s %-22s pyc_hits=%d  %s\" % (label, marker, n, \"OK\" if n else \"MISSING\"))
    bad += (n == 0)

print(\"BYTECODE_GATE\", \"OK\" if bad == 0 else \"FAILED(%d)\" % bad)
'
" 2>&1 | grep -v libtinfow
