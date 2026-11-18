#!/usr/bin/env bash
# On the second track-B node (hold_b.sbatch): stage the model and build infera-vllm:v0340-0c5e5e89 on the local NVMe
# (as the v3-258 preparation), fetch the AgentX dependencies, then the DRAM-only (OffloadingConnector, write-through)
# P4D4 3600 s control with the b1-r1 configuration (prefill GEM 0.88, 1.25 TB CPU tier).
R=/home/liyingli/bench_agentx/vllm-trackB-n4-20261118
export R
cd "$R" || exit 1
echo "$(date -u +%T) n4 on $(hostname -s)"
bash scripts/stage_model.sh /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4 /mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4 > logs/stage.log 2>&1 &
if ! docker image inspect infera-vllm:v0340-0c5e5e89 > /dev/null 2>&1; then
    d=/mnt/m2m_nobackup/liyingli_vpd/build-0c5e5e89
    rm -rf "$d"; mkdir -p "$d"
    git -C /home/liyingli/bench_agentx/baseline/Infera-vllm-glm52 archive 0c5e5e89 | tar -x -C "$d"
    ( cd "$d" && docker build --progress=plain -f llying/vllm-image-v0340-20261117/Dockerfile -t infera-vllm:v0340-0c5e5e89 . ) > logs/build.log 2>&1
    echo "$(date -u +%T) build rc=$?"
fi
wait
tail -1 logs/stage.log
docker image inspect infera-vllm:v0340-0c5e5e89 > /dev/null 2>&1 || { echo "no image"; exit 0; }
PREP_ONLY=1 bash scripts/agentx.sh prep-agentx > logs/prep-agentx.log 2>&1; tail -1 logs/prep-agentx.log
docker rm -f vpd-prefill vpd-decode vpd-router vpd-etcd vpd-agentx > /dev/null 2>&1
RUN_PREFIX=b1 DURATION=3600 bash scripts/p4d4_run.sh r4-3600-n4 score
exit 0
