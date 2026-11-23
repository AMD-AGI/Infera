#!/usr/bin/env bash
# Where are aiter's BF16 GEMM tuner and tuned tables in the vLLM image (CPU only, no GPU). Always exits 0 (spur step).
#   srun --jobid=JOB --overlap -N1 -n1 bash probe_aiter_tuner.sh OUT
out=${1:-/tmp/probe_aiter_tuner.txt}
docker run --rm --network none --entrypoint bash infera-vllm:v0340-0c5e5e89 -c '
    pip show amd-aiter 2>/dev/null | head -3
    A=$(python3 -c "import aiter, os; print(os.path.dirname(aiter.__file__))" 2>/dev/null); echo "aiter dir: $A"
    ls "$A"/configs 2>/dev/null | head -40; echo "-- model_configs:"; ls "$A"/configs/model_configs 2>/dev/null | grep -i -E "bf16|glm|dsv3|dsr1" | head -40
    echo "-- tuners:"; find / -xdev \( -name "gemm_a16w16_tune.py" -o -name "*bf16*tune*.py" \) 2>/dev/null | head
    echo "-- base_tuner:"; python3 -c "import aiter.utility.base_tuner as b; print(b.__file__)" 2>&1 | tail -1
    echo "-- hipblaslt in gradlib:"; python3 -c "import gradlib; print(gradlib.__file__)" 2>&1 | tail -1
' > "$out" 2>&1
echo "$(date -u +%T) probe done on $(hostname -s) -> $out"
exit 0
