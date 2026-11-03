#!/usr/bin/env bash
# One-node Slurm job that readies a node for the verification: GPU / VRAM diagnostics (who holds memory on idle GPUs),
# the image (docker load from IMAGE_TAR if present, else build it with build_image.sh and save it to IMAGE_TAR), the
# 11-file base check, and the unit checks. Values from config/site.local.env; INFERA_REPO = a clone with commit fdbafb60.
#   sbatch -p PART -A ACCT -q QOS -N 1 --gres=gpu:8 --exclusive -t 03:00:00 -o LOG_ROOT/slurm-image-%j.out scripts/slurm_image.sbatch
#SBATCH -J lyl-vfix-img
#SBATCH -N 1
#SBATCH --exclusive
set -uo pipefail
PKG="${PKG:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
[[ -f "$PKG/config/recipe-e10.env" ]] || PKG="$(cd "${SLURM_SUBMIT_DIR:?}" && pwd)"
export PKG
set -a; . "$PKG/config/site.local.env"; set +a
echo "== $(date -u +%F' '%T) $(hostname) job ${SLURM_JOB_ID:-?}"
vram() { rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR>1 && $2 != "" {printf "%d ", 100 * $3 / $2}'; echo; }
echo "== VRAM % per GPU: $(vram)"
rocm-smi --showpids 2>&1 | sed -n 1,30p
amd-smi process 2>&1 | grep -vE "^\s*$" | head -40
docker ps -a --format '{{.Names}} {{.Image}} {{.Status}}' | head
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "== image present: $(docker image inspect -f '{{.Id}}' "$IMAGE")"
elif [[ -s "${IMAGE_TAR:-}" ]]; then
    echo "== loading $IMAGE_TAR"; docker load -i "$IMAGE_TAR"
fi
echo "== build / check"
INFERA_REPO="${INFERA_REPO:?}" CTX="${CTX:-$WORK_DIR/../ctx-fdbafb60}" SAVE_TAR="${IMAGE_TAR:-}" bash "$PKG/scripts/build_image.sh" || { echo "image step failed"; exit 1; }
echo "== unit checks"
bash "$PKG/scripts/run_unit.sh" "${PREFILL0_GPUS%%,*}"
echo "== VRAM % per GPU after: $(vram)"
echo "== done $(date -u +%T)"
