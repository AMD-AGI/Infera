set -u
SRC=/root/kvaware_kvd_build
cd "$SRC"
LOG=/root/kvaware_kvd_build.log
: > "$LOG"

echo "===== STAGE 1: base image (infera/engine-sglang:kvaware-kvd-base) =====" | tee -a "$LOG"
docker build -f deploy/docker/Dockerfile.sglang \
  -t infera/engine-sglang:kvaware-kvd-base . >> "$LOG" 2>&1
rc=$?
echo "stage1 rc=$rc" | tee -a "$LOG"
[ $rc -ne 0 ] && { echo "STAGE1 FAILED"; tail -40 "$LOG"; exit 1; }

echo "===== STAGE 2: kvaware-kvd layer =====" | tee -a "$LOG"
docker build -f deploy/docker/Dockerfile.sglang.kvaware-kvd \
  --build-arg INFERA_SGLANG_IMAGE=infera/engine-sglang:kvaware-kvd-base \
  -t infera/engine-sglang:kvaware-kvd . >> "$LOG" 2>&1
rc=$?
echo "stage2 rc=$rc" | tee -a "$LOG"
[ $rc -ne 0 ] && { echo "STAGE2 FAILED"; tail -40 "$LOG"; exit 1; }

echo "===== DONE =====" | tee -a "$LOG"
docker images --format "{{.Repository}}:{{.Tag}} {{.Size}}" | grep kvaware-kvd | tee -a "$LOG"
