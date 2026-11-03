#!/usr/bin/env bash
# Environment probe inside a Slurm (spur) allocation: can this node run the kit (docker, image, GPUs, RDMA, paths)?
# Usage: sbatch -p PARTITION -A ACCOUNT -q QOS -N 1 --gres gpu:8 --exclusive -t 00:30:00 -o OUT slurm_probe.sh
set -u
echo "== $(date -u +%F' '%T) $(hostname) job=${SLURM_JOB_ID:-?} nodes=${SLURM_JOB_NODELIST:-?}"
ip -o -4 addr show | awk '{print $2, $4}' | grep -v "^lo"
echo "== user: $(id)"
echo "== docker:"; docker version --format '{{.Server.Version}}' 2>&1 | head -2
docker ps --format '{{.Names}} {{.Image}} {{.Status}}' 2>&1 | head
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' 2>&1 | grep -iE "infera|etcd|sglang" | head
docker info 2>/dev/null | grep -E "Docker Root Dir|Storage Driver"
echo "== GPUs:"; rocm-smi --showmemuse 2>&1 | grep -E "VRAM%" | awk '{print $NF}' | tr '\n' ' '; echo
echo "== RDMA:"; ls /sys/class/infiniband/ 2>&1 | tr '\n' ' '; echo; ls -la /lib/x86_64-linux-gnu/libionic.so* 2>&1 | head -3
echo "== disks:"; df -h / /mnt/m2m_nobackup /home /shared_nfs 2>&1 | tail -5
ls -la /mnt/m2m_nobackup/ 2>&1 | head; ls /mnt/m2m_nobackup/liyingli_3p1d 2>&1 | head -3
echo "== model:"; ls /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4 2>&1 | head -3
echo "== mem:"; free -g | head -2
echo "== enroot/pyxis:"; which enroot 2>&1; srun --help 2>&1 | grep -iE "container" | head -3
echo "== done $(date -u +%T)"
