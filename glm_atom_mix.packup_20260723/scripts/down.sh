#!/bin/bash
# Tear down the ATOM GLM container and confirm card4-7 VRAM returns to baseline.
# IMPORTANT: removes ONLY our container. NEVER `docker rmi` the atom image
# (re-loading its 45GB tar nearly filled the node / and hung ssh) and NEVER prune
# (would nuke foreign titan/zirui/primus images). No slurm scancel.
set -u
CTR="${CTR:-glm_atom_c_hf}"
docker rm -f "$CTR" 2>/dev/null || true
echo "[down] removed $CTR; card4-7 should return to ~283MB:"
rocm-smi --csv --showmeminfo vram 2>/dev/null | grep -E "card[4-7]" | awk -F, '{print $1, int($3/1024/1024)"MB"}'
echo "[down] disk (leave no worse than ~34G free):"; df -h / | tail -1
