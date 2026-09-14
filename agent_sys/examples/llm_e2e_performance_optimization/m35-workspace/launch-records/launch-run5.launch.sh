#!/usr/bin/env bash
# RUN 5 — REPLAY. m1 and m2 are replayed from run 20260906T154908-d9c7af; the
# chain starts real work at m3. NOT AN ACCEPTANCE RUN and it cannot be one:
# SKIP-AHEAD.md's first page says replay is a debugging accelerator and never an
# acceptance path, and this run cannot satisfy Finish Standard 1. What it can
# answer is whether m3 -> m4 -> m5 -> packup works at all, which nobody has ever
# known. A clean full real run follows, with ~21 h of hold to hold it.
#
# REPLAYED HANDOFFS, all from run 20260906T154908-d9c7af (run 4), materialised by
# assets/lib/replay_root.py into replay_root_run4/ with PROMOTION.json beside them:
#     deploy_kit                        v1  40 files
#     profiling_evidence                v1  47 files
#     profiling_mode_off.bench_result   v0  17 files
#     profiling_mode_on.bench_result    v0  17 files
#     profiling_mode_on.kernel_table    v0   8 files
#     profiling_mode_on.profile_result  v0  18 files
#
# TWO CAVEATS THE TOOL RAISED AND I AM CARRYING, NOT SUPPRESSING:
#  1. --threshold 1 --allow-unstable. The tool's default bar is 3 consecutive
#     clean runs; we have 1. Only deploy_kit clears even at 1. The other five
#     report "cannot tell - no recorded discriminator": the run tree does not
#     record E2E_MOCK_STAGES, so the artefact cannot say whether m2 was mocked.
#     I know it was real first-hand - I watched the bring-ups, the aiperf loads
#     and the 825176 GPU kernel events - but that is my observation, not the
#     artefact's.
#  2. "A replayed record can name a container that no longer exists. Until the
#     seam is answered, treat a failure in the consuming stage as possibly this
#     and not that stage'"'"'s defect." So an m3 failure here is suspect until
#     shown otherwise.
#
# CODE CHANGE IN THIS RUN: identify.py now makes logical_operator unique
# (commit 649af26b). That is what run 4 refused on.
# Carried forward: the attributable-VRAM preflight, unconditional log capture,
# trace_end_ms=120000, kernel_table_min_launchers=0 (kept - m2 is replayed so
# check_kernel_table is not re-executing; the clean run drops it).
# NO log-text retry.
set -u
cd /home/yihou/dev/git.16-19/infera
LD=/data/yihou/e2e_verify_20260906/m35/launch-run5
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
  --var mock_stages=m1,m2 \
  --var mock_root=/data/yihou/e2e_verify_20260906/m35/replay_root_run4 \
  --var m1_agent=runner \
  --var m2_agent=runner \
  --var work_root=/data/yihou/e2e_flow5 \
  --var scratch_root=/data/yihou/e2e_flow5/kfo \
  --var validate_work_root=/data/yihou/e2e_flow5/validate \
  --var container=yihou_e2e_chain5 \
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
