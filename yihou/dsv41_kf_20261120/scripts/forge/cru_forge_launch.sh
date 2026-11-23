#!/bin/bash
# ON a crusoe node (spur exec), as yihou: start one Forge campaign per OP in its own container on its own GPU.
# usage: cru_forge_launch.sh MAX_HOURS GPU:OP [GPU:OP ...]   (credentials: ~/.yihou_forge.env, 0600, never printed)
set -uo pipefail
H=${1:?max hours}; shift
FR=/shared_nfs/yihou/dsv41_kf_20261120/forge
IMG=vllm/vllm-openai-rocm@sha256:694108707f333787b1041caa0e948dd18b86f2450365a94e38819f026275ee32
ENVF=$HOME/.yihou_forge.env
[ "$(stat -c %a "$ENVF")" = 600 ] || { echo "credential file must be 0600"; exit 2; }
docker image inspect "$IMG" >/dev/null 2>&1 || docker pull "$IMG" > "$FR/logs/pull.log" 2>&1
for spec in "$@"; do
  G=${spec%%:*} OP=${spec#*:} C=$FR/runs/$OP
  [[ "$OP" =~ ^mhc_seam_[a-z0-9_]+$ ]] || { echo "bad op $OP"; exit 2; }
  FORGE_ROOT=$FR bash "$FR/prep_campaign.sh" "$OP" >/dev/null
  mkdir -p "$C/logs"
  docker run -d --rm --name "yihou-dsv41kf-forge-$OP" --network host --ipc host --user "$(id -u):$(id -g)" \
    --device /dev/kfd --device /dev/dri --group-add video --group-add render --cap-add SYS_PTRACE \
    --env-file "$ENVF" -e HOME=$FR/home -e PYTHONUSERBASE=$FR/pyuser -e AKLR_KB_ROOT=$FR/aklr/harness/skills/kb \
    -e HIP_VISIBLE_DEVICES=$G -e PYTHONDONTWRITEBYTECODE=1 -v $FR:$FR -w "$C/engine" --entrypoint bash "$IMG" -c \
    "export PATH=$FR/home/.local/bin:$FR/pyuser/bin:\$PATH; python3 -m pip install --user -q -e $FR/hyperloom[forge] > $C/logs/pip.log 2>&1; \
     sh $C/ws/operators/$OP/run_forge.sh --workspace $C/engine --gpu 0 --max-hours $H --model 'Claude-Opus-5.5[1m]' \
       > $C/logs/forge.log 2>&1; echo \"forge rc=\$?\" >> $C/logs/forge.log" >/dev/null && echo "started $OP on GPU $G"
done
