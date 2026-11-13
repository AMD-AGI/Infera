#!/bin/bash
cd ~/dllm/Infera
export HF_HOME=/shared/models HF_HUB_CACHE=/shared/models HF_HUB_OFFLINE=1
rm -f ~/dllm/results/tp_steps.json
for tp in 1 2 4 8; do
  ~/dllm-env/bin/torchrun --nproc_per_node "$tp" ~/dllm/dllmx/tp_step.py ~/dllm/results/tp_steps.json 2>&1 \
    | grep -E '^\{|Error' | grep -v Warning
done
