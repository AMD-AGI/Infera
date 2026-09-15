#!/usr/bin/env bash
# Create the owned container for the TP8/EP8/DPA decode point on this bare-metal host.
# Adapted from packups/glm52_tp4_noep_dpa_sweep_yihou.packup_20260911-1100/scripts/create_container_yihou.sh:
# the slurm/spur allocation guards and the crsuse2-m2m node guards are removed (this host is ours and
# has no scheduler); everything that affects the measurement is kept identical.
set -euo pipefail

WS=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$WS/.." && pwd)

CONTAINER=${CONTAINER:-yihou-glm52-tp8ep8-0914}
[[ "$CONTAINER" =~ ^yihou-[A-Za-z0-9_-]+$ ]] || { printf 'Container name must start with yihou-: %s\n' "$CONTAINER" >&2; exit 2; }

# Pinned by digest. This host does not have the spur image archive (b9a83742f631...); this digest is
# the local `rocm-llm-bench:latest`, probed to carry the same SGLang / AITER / torch as the packup.
IMAGE=${IMAGE:-sha256:b5aa5bd3d828285bff8dd01bb92e6106f569f5b1eab5de24558790b1b10a5dc0}
IMAGE_KEY=${IMAGE#sha256:}; IMAGE_KEY=${IMAGE_KEY:0:12}

MODEL=${MODEL:-/perf_apps/data/models/GLM-5.2-MXFP4}
GPU_DEVICES=${GPU_DEVICES:-0,1,2,3,4,5,6,7}
[[ "$GPU_DEVICES" =~ ^[0-7](,[0-7])*$ ]] || exit 2

# AITER writes lock files next to its tuned-GEMM CSVs in /tmp/aiter_configs, which is root-owned in
# the image; as a non-root user that fails. Bind a host-owned copy of the image's own CSVs over it so
# the tuned configs (which affect kernel selection) are preserved byte-for-byte and writable.
AITER_CONFIGS=${AITER_CONFIGS:-/var/tmp/yihou-glm52-aiter-configs}
if [[ ! -f "$AITER_CONFIGS/bf16_tuned_gemm.csv" ]]; then
    mkdir -p "$AITER_CONFIGS"
    docker run --rm --user 0:0 -v "$AITER_CONFIGS:/mnt-out" --entrypoint bash "${IMAGE}" -lc \
        "cp -a /tmp/aiter_configs/. /mnt-out/ && chown -R $(id -u):$(id -g) /mnt-out"
fi

# The host resolves this uid through LDAP/sssd, so it is absent from /etc/passwd and from the image.
# torch's inductor calls getpass.getuser() -> pwd.getpwuid() at import time and dies with
# `KeyError: getpwuid(): uid not found`. Synthesize a passwd/group file carrying our entry.
NSS_DIR=${NSS_DIR:-/var/tmp/yihou-glm52-nss}
mkdir -p "$NSS_DIR"
{ cat /etc/passwd; getent passwd "$(id -u)"; } > "$NSS_DIR/passwd"
{ cat /etc/group; getent group "$(id -g)"; getent group video; getent group render; } > "$NSS_DIR/group"

docker image inspect "$IMAGE" >/dev/null ||{ printf 'Pinned image not present: %s\n' "$IMAGE" >&2; exit 1; }
[[ -d "$MODEL" ]] || { printf 'Model path missing: %s\n' "$MODEL" >&2; exit 1; }
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    printf 'Refusing to clobber existing container: %s\n' "$CONTAINER" >&2; exit 1
fi

# The repo lives on an NFS home export with root_squash: a container running as root cannot write
# result files into the workspace (verified — `touch` returns EPERM). Run as the host user instead,
# with the video/render groups the ROCm devices need.
docker run -d --name "$CONTAINER" --label owner="$USER" --label task=glm52-internal-decode \
    -w / --network host --ipc=host --shm-size 32g \
    --user "$(id -u):$(id -g)" --group-add video --group-add render \
    --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
    -v "$REPO:$REPO" -v "$MODEL:$MODEL:ro" \
    -v "$AITER_CONFIGS:/tmp/aiter_configs" \
    -v "$NSS_DIR/passwd:/etc/passwd:ro" -v "$NSS_DIR/group:/etc/group:ro" \
    -e HOME=/tmp/yihou-home -e USER="$USER" -e LOGNAME="$USER" \
    -e PATH=/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/go/bin \
    -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 -e HIP_VISIBLE_DEVICES="$GPU_DEVICES" \
    -e AITER_JIT_DIR="/tmp/yihou-aiter-$IMAGE_KEY" \
    -e HF_HOME=/tmp/yihou-hf-cache -e HF_HUB_OFFLINE=1 \
    -e SGLANG_OPT_USE_TOPK_V2=false -e SGLANG_TIMEOUT_KEEP_ALIVE=900 \
    -e AITER_USE_FLYDSL_MOE_SORTING=1 -e SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1 \
    "$IMAGE" sleep infinity

printf 'Created %s from %s\n' "$CONTAINER" "$IMAGE"
