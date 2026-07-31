#!/usr/bin/env bash
# Strict reproduction of `tests/run_tests.sh e2e sglang mixed`, but only the GLM
# case (the matrix's gpt-oss / kimi cases need other models and would just skip
# or waste an hour). Mirrors run_tests.sh exactly: same image, same GPU_FLAGS,
# same etcd, same pytest invocation, model dir RO-mounted at its own path.
set -uo pipefail

REPO=/root/infera_e2e
IMG=infera/engine-sglang:test-local
ETCD_IMG=quay.io/coreos/etcd:v3.5.14
ETCD_CTR=infera-utest-etcd
MODEL_DIR=/mnt/vast/c_huggingface/e2e_models
SCRATCH=$(mktemp -d /tmp/infera-test.XXXXXX)
mkdir -p "$SCRATCH/hf" /tmp/infera-e2e-logs
: > "$SCRATCH/failures.txt"; chmod 666 "$SCRATCH/failures.txt" 2>/dev/null || true

# -k filter: only the GLM case unless overridden
KFILTER="${KFILTER:-GLM}"

GPU_FLAGS=(
  --init --privileged --ipc host --shm-size 16gb --ulimit memlock=-1
  --device /dev/kfd --device /dev/dri --group-add video --group-add render
  -v /boot:/boot:ro
)
SCRATCH_FLAGS=(-v "$SCRATCH":/scratch -e HF_HOME=/scratch/hf -v /tmp/infera-e2e-logs:/e2e-logs)
# Mount the real model tree over the placeholder so resolve_model()'s isdir()
# succeeds INSIDE the container (a symlink to an unmounted path would dangle).
E2E_FLAGS=(-v "$MODEL_DIR":"$MODEL_DIR":ro
           -v /mnt/vast/xiaobo/models/GLM-5.1-FP8:"$MODEL_DIR/zai-org/GLM-5.1-FP8":ro
           -e INFERA_E2E_MODEL_DIR="$MODEL_DIR")

echo "[e2e] model dir: $MODEL_DIR (read-only)"
docker rm -f "$ETCD_CTR" >/dev/null 2>&1 || true
echo "[e2e] starting temporary etcd ($ETCD_CTR)"
docker run -d --rm --name "$ETCD_CTR" --net host "$ETCD_IMG" \
  etcd --advertise-client-urls http://127.0.0.1:2379 \
       --listen-client-urls http://0.0.0.0:2379 >/dev/null
sleep 5
curl -sf -m 5 http://127.0.0.1:2379/version && echo " <- etcd up" || echo "ETCD NOT REACHABLE"

echo "----- e2e in $IMG — tests/e2e/pd_mixed/sglang/test_mixed.py (-k '$KFILTER') -----"
docker run --rm --name infera-utest-e2e --network host \
  "${GPU_FLAGS[@]}" "${SCRATCH_FLAGS[@]}" "${E2E_FLAGS[@]}" \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -v "$REPO":/workspace:ro -w /workspace --entrypoint bash "$IMG" -lc \
  "pip install -q pytest pytest-asyncio nats-py 2>/dev/null || true; \
   python3 -m pytest -p no:cacheprovider -o addopts= -rfE -v -s \
     -k '$KFILTER' tests/e2e/pd_mixed/sglang/test_mixed.py 2>&1 \
   | stdbuf -oL tee /scratch/.e2e.out; exit \${PIPESTATUS[0]}"
RC=$?

echo "===== RC=$RC ====="
docker rm -f "$ETCD_CTR" >/dev/null 2>&1 || true
cp "$SCRATCH/.e2e.out" /mnt/vast/c_huggingface/glm52_dsa_v0516/e2e_strict.out 2>/dev/null || true
echo "log -> /mnt/vast/c_huggingface/glm52_dsa_v0516/e2e_strict.out"
exit $RC
