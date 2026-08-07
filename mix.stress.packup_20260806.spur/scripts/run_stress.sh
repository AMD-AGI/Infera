#!/bin/bash
# Task 3 — agentic stress via the INFERA driver (agent.agent_throughput), NOT the
# customer/AgentX bench (mission rule 5). Runs on the NODE HOST via the staged venv
# (the engine container has no agent module); talks to the router at :8100.
# init_sessions=8, max_inflight=16, max_sessions=24. ramp 400 + sustain 3600 (honest).
set -u
V="${V:-/shared_nfs/yihou_agentbench/venv/bin/python}"
MY_IP="${MY_IP:-10.245.148.191}"
YAML="${YAML:-/shared_nfs/yihou_final_pr/mix/scripts/stress_caseA.yaml}"
NAME="${NAME:-mix_stress_caseA}"
DATADIR="${DATADIR:-/shared_nfs/yihou_final_pr/mix/results/stress}"
mkdir -p "$DATADIR"
"$V" -m agent.agent_throughput \
  --workload-config "$YAML" \
  --server "http://$MY_IP:8100" --model glm5.2-mxfp4 \
  --tokenizer /shared_nfs/GLM-5.2-MXFP4 \
  --name "$NAME" --data-dir "$DATADIR" --dashboard-mode
