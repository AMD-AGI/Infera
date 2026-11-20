#!/usr/bin/env bash
# Self-test of the AgentX client runtime (no server, no GPU). Run ON the node before the first run_point.sh, or on the
# dev host with --no-docker. Checks: venv python + agentx-harness 1.0.14, InferenceX import, offline tokenizer load,
# offline dataset load (393 traces @ 23f152f6), and that infx's Plan/replay_argv accept the env run_point.sh sets
# (prints the exact `agentx profile` argv for a dummy point). Output also saved to $DSV41_NFS_ROOT/selftest/.
#
# Usage: client_selftest.sh [--no-docker] [--image IMAGE]
#   --image   client image (default: the R1/R2 server image; must already be on the node)
#
# How the runtime was built (2026-11-19 on a16-19, Ubuntu 22.04 x86_64, uv 0.12.9; reproducible, all inside /yihou/ paths):
#   R=/perf_apps/yihou/dsv41_vllm_pd_20261119/venv-agentx
#   UV_CACHE_DIR=$R/uv-cache UV_PYTHON_INSTALL_DIR=$R/uv-python uv venv --python 3.11 --managed-python $R/venv
#   UV_CACHE_DIR=$R/uv-cache uv pip install --python $R/venv/bin/python --no-config \
#       -r <ws>/source/InferenceX/inferencex-e2e/infx/bench/agentic/requirements.txt   # agentx-harness==1.0.14, datasets>=4.7.0
#   (freeze: $R/freeze.txt). Python is uv-managed CPython 3.11.16 (python-build-standalone) under $R/uv-python, so the venv
#   does not depend on the host/container python; it is mounted at the same path inside the client container.
#   HF cache: HF_HOME=/perf_apps/yihou/dsv41_vllm_pd_20261119/hf_home
#     hf download --repo-type dataset semianalysisai/cc-traces-weka-062126 --revision 23f152f6f0f9399a85901b89a6458def0ef16729
#     printf 23f152f6f0f9399a85901b89a6458def0ef16729 > hub/datasets--semianalysisai--cc-traces-weka-062126/refs/main
#     python: datasets.load_dataset(..., revision=<sha>) once online (builds the Arrow cache used offline)
#     hf download deepseek-ai/DeepSeek-V4.1-Flash config.json tokenizer.json tokenizer_config.json chat_template.jinja (main=314113f8)
set -uo pipefail
KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WS=$(cd "$KIT_DIR/../.." && pwd)
NFS_ROOT=${DSV41_NFS_ROOT:-/perf_apps/yihou/dsv41_vllm_pd_20261119}
IX=${KIT_IX:-$NFS_ROOT/stage/inferencex-e2e}   # staged: docker cannot read HOME NFS on nodes
VENV_ROOT=$NFS_ROOT/venv-agentx
VENV_PY=$VENV_ROOT/venv/bin/python
HFH=$NFS_ROOT/hf_home
IMAGE="vllm/vllm-openai-rocm:nightly-60bd6dd2edd175a1c55fd3488f871e4c3d2ef27d@sha256:67a3a0fb79893fc5a0fcff5f9b739ba8fec7e920875a12e0574c2e9f50cd814c"
NODOCKER=0
while [ $# -gt 0 ]; do
  case "$1" in --no-docker) NODOCKER=1 ;; --image) IMAGE=${2:?}; shift ;; *) echo "unknown option $1" >&2; exit 2 ;; esac
  shift
done
case "$NFS_ROOT" in */yihou/*) ;; *) echo "DSV41_NFS_ROOT must contain /yihou/" >&2; exit 2 ;; esac
OUTD=$NFS_ROOT/selftest; [ "$NODOCKER" = 1 ] && OUTD=$WS/debug/kit_selftest   # dev-host runs stay in the workspace
mkdir -p "$OUTD"
OUT=$OUTD/$(hostname -s)-$(date -u +%Y%m%dT%H%M%SZ).log

read -r -d '' PY <<'EOF'
import importlib.metadata as md, json, os, platform, sys, tempfile, time
from pathlib import Path
print("python", sys.version.split()[0], sys.executable, "glibc", platform.libc_ver())
v = md.version("agentx-harness"); print("agentx-harness", v); assert v == "1.0.14", v
from infx.bench.agentic.venv import Runtime
from infx.bench.agentic.run import Plan
from infx.bench.agentic.replay import replay_argv
rt = Runtime.for_job(os.environ); print("runtime.active (skips bootstrap):", rt.active()); assert rt.active()
t = time.time()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V4.1-Flash", trust_remote_code=True)
print("tokenizer", type(tok).__name__, len(tok), "%.1fs" % (time.time() - t))
t = time.time()
from datasets import load_dataset
ds = load_dataset("semianalysisai/cc-traces-weka-062126", split="train")
print("dataset rows", len(ds), "%.1fs" % (time.time() - t), ds.cache_files[0]["filename"]); assert len(ds) == 393
res = Path(tempfile.mkdtemp(prefix="yihou-selftest-"))
env = dict(os.environ, RESULT_DIR=str(res), AGENTIC_OUTPUT_DIR=str(res), RESULT_FILENAME="inferencex_result",
           MODEL="deepseek-ai/DeepSeek-V4.1-Flash", SERVED_MODEL_NAME="deepseek-ai/DeepSeek-V4.1-Flash",
           TOKENIZER="deepseek-ai/DeepSeek-V4.1-Flash", MODEL_PREFIX="dsv41flash", FRAMEWORK="vllm", PRECISION="fp4",
           CONC="128", DURATION="3600", AIPERF_EXPERIMENTAL_FAST="0", TP="2", EP_SIZE="1", PP_SIZE="1", PCP_SIZE="1",
           DCP_SIZE="1", DP_ATTENTION="true", SPEC_DECODING="mtp", IS_MULTINODE="false", DISAGG="false", EVAL_ONLY="false",
           RUN_EVAL="false", IS_AGENTIC="1", SCENARIO_TYPE="agentic-coding", ENABLE_AGENTX_POWER="false",
           REQUIRE_POWER="false", KV_OFFLOADING="dram", KV_OFFLOAD_BACKEND="umbp", KV_OFFLOAD_BACKEND_METADATA='{"name":"umbp"}',
           TOTAL_CPU_DRAM_GB="449", AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS="300",
           AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID="false", AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING="0",
           WEKA_LOADER_OVERRIDE="semianalysis_cc_traces_weka_062126", AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:",
           AIPERF_SERVER_URL="http://127.0.0.1:8000", AIPERF_SERVER_METRICS_URLS="http://127.0.0.1:6100/metrics")
plan = Plan.from_env(env)
print("plan ok: power", plan.power, "| metric prefix", plan.required_metric_prefix, "| result_dir", plan.result_dir)
print("argv:", " ".join(replay_argv(plan.replay, rt.agentx)))
for fast in ("1",):
    p2 = Plan.from_env(dict(env, AIPERF_EXPERIMENTAL_FAST=fast))
    print("fast argv:", " ".join(replay_argv(p2.replay, rt.agentx)))
res.rmdir()
print("SELFTEST PASS")
EOF

ENVS=(AIPERF_RUNTIME_DIR="$VENV_ROOT" PYTHONPATH="$IX" PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
  HF_HOME="$HFH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TMPDIR=/tmp)
if [ "$NODOCKER" = 1 ]; then
  mkdir -p "$OUTD/tmp"
  ( cd "$OUTD/tmp" && env "${ENVS[@]}" TMPDIR="$OUTD/tmp" "$VENV_PY" -c "$PY" ) 2>&1 | grep -v 'Warning: You are sending' | tee "$OUT"
else
  command -v docker >/dev/null || { echo "docker not found" >&2; exit 2; }
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image not on node: $IMAGE" >&2; exit 2; }
  LOC=${DSV41_LOCAL_ROOT:-/data/yihou/dsv41_vllm_pd_20261119}/client/selftest; mkdir -p "$LOC/tmp"
  D=(docker run --rm --name "yihou-dsv41-selftest-client-$$" --network none --user "$(id -u):$(id -g)"
     -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v "$HFH:$HFH" -v "$VENV_ROOT:$VENV_ROOT:ro" -v "$IX:$IX:ro"
     -v "$LOC/tmp:/tmp")
  for e in "${ENVS[@]}" HOME=/tmp LD_LIBRARY_PATH= LD_PRELOAD=; do D+=(-e "$e"); done
  "${D[@]}" --entrypoint "$VENV_PY" "$IMAGE" -c "$PY" 2>&1 | grep -v 'Warning: You are sending' | tee "$OUT"
fi
grep -q 'SELFTEST PASS' "$OUT" && echo "saved $OUT" || { echo "SELFTEST FAILED (log $OUT)"; exit 1; }
