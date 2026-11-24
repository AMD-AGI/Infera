#!/usr/bin/env bash
# Run an analysis command in the vLLM image on the node of one of my running jobs (the login node has no numpy and
# heavy work does not belong there):
#   ana_on_node.sh JOB 'python3 $B/scripts/ttft_analyze.py runs/<run>'
# $R and the llying directory are mounted at their own paths, cwd $R. srun forces a pty here, so binary output on
# stdout gets mangled: write files to the shared filesystem instead.
job=$1; shift
R=/home/liyingli/bench_agentx/vllm-pd-20261117
L=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying
timeout 900 srun --jobid="$job" --overlap bash -c "docker run --rm --network none -v $R:$R -v $L:$L -w $R \
    --entrypoint bash infera-vllm:v0340-0c5e5e89 -c '$*'" 2>&1 | grep -v -E '^spur: warning|^srun: warning'
