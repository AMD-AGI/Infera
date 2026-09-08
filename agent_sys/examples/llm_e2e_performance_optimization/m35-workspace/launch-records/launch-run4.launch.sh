#!/usr/bin/env bash
# RUN 4 — full real chain. Same as run 3 plus TWO instruction additions:
#   (a) capture etcd (and every started container) logs UNCONDITIONALLY;
#   (b) ONE retry on the named detokenizer signature, keeping attempt 1 logs,
#       dying loudly with both named if the retry also fails.
# Carried forward unchanged: the attributable-VRAM preflight (verified by
# known-answer test on run 3s kit), trace_end_ms=120000,
# kernel_table_min_launchers=0.
# Separate container/work_root per run is now the DEFAULT, not a deviation
# (leader, 2026-09-06): a shared work_root does not abort, so the only person
# who finds out is whoever reads the artefacts tomorrow.
# NOT replayed. m1 runs real: m2 executes deploy.sh/preflight.sh FROM the kit.
set -u
cd /home/yihou/dev/git.16-19/infera
LD=/data/yihou/e2e_verify_20260906/m35/launch-run4
env -u PYTHONPATH \
CLAUDE_CONFIG_DIR=/data/yihou/e2e_verify_20260906/m1/claude-config \
python3 /data/yihou/e2e_verify_20260906/m2/launch_chain.py \
  --stall-after 900 run \
  --package agent_sys/examples/llm_e2e_performance_optimization/e2e-flow \
  --demo-root /data/yihou/agent_sys_runroot \
  --timeout 21600 \
  --var jobid=29313 \
  --var node=smci355-ccs-aus-n04-25 \
  --var node_ip=10.235.192.131 \
  --var transport=local \
  --var model_name=Qwen/Qwen3-32B \
  --var model_path=/apps/data/models/Qwen3-32B \
  --var image=infera/engine-sglang:qwen3-local-20260906 \
  --var context_length=40960 \
  --var tp=4 --var expect_ranks=4 \
  --var dsa_args=none --var parser_args=none \
  --var mock_stages=none \
  --var work_root=/data/yihou/e2e_flow4 \
  --var scratch_root=/data/yihou/e2e_flow4/kfo \
  --var validate_work_root=/data/yihou/e2e_flow4/validate \
  --var container=yihou_e2e_chain4 \
  --var port_router=8101 --var port_worker=8102 --var port_etcd=8103 \
  --var measure_gpu=4 \
  --var magpie_root=/data/yihou/Magpie \
  --var aiperf_trace=/data/yihou/e2e_verify_20260906/m2/materials/conversation_trace.v2.jsonl \
  --var gsm8k_data=/data/yihou/e2e_verify_20260906/m2/materials/gsm8k_test.jsonl \
  --var trace_end_ms=120000 \
  --var bench_rounds=3 \
  --var adhoc_cases=3 \
  --var eval_thinking=none \
  --var forge_mock=1 \
  --var kernel_table_min_launchers=0 \
  --var instruction="$(cat "$LD/instruction.txt")"
