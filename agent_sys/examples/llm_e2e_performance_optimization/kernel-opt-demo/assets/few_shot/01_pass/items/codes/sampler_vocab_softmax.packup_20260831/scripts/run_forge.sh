#!/usr/bin/env bash
# Stage 2b — forge-loop on the REAL sglang decode hot kernel found in Stage 2a:
# the sampler's vocabulary softmax, [8, 151936] fp32, 55.59 us with ATen.
#
# Runs INSIDE the kf-mi300b container.
set -euo pipefail

REPO=/home/yihou/dev/git.16-19/KernelForge
SCRATCH=/tmp/yihou/kf_mission_20260831
TASK_DIR="$REPO/experiments/mission_workspace/task_sampler_softmax"
WORKSPACE="${1:-$SCRATCH/runs/s2_forge_sampler_softmax}"

GPU_TARGET="${GPU_TARGET:-gfx942}"
GPU_TYPE="${GPU_TYPE:-mi300x}"
MAX_HOURS="${MAX_HOURS:-3.0}"        # > 2.0 or Analysis degrades to static-only
FORGE_MODEL="${FORGE_MODEL:-Claude-Sonnet-5[1m]}"

echo "==> workspace: $WORKSPACE"
# Never `rm -rf "$VAR"`. Refuse a dirty workspace; caller supplies a fresh path.
if [ -e "$WORKSPACE" ]; then
  echo "error: $WORKSPACE already exists. Pass a fresh path as \$1." >&2
  exit 1
fi
mkdir -p "$WORKSPACE"
cp "$TASK_DIR"/{sampler_softmax_kernel.py,driver.py,graph_harness.py,program.md} "$WORKSPACE/"

cd "$WORKSPACE"
git init -q
git config user.email "forge-mission@local"
git config user.name "forge-mission"
cat > .gitignore <<'EOF'
__pycache__/
*.pyc
*.log
build/
forge_experiments/
EOF
git add -A
git commit -q -m "forge: initial sampler-softmax workspace" || true

export IS_SANDBOX=1
export PYTHONUNBUFFERED=1
export KA_EVENTS_STDOUT=1
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-3}"
export KNOWLEDGE_LOCAL_ROOT="${KNOWLEDGE_LOCAL_ROOT:-$SCRATCH/knowledge}"
export CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-$SCRATCH/claude_cfg}"
mkdir -p "$CLAUDE_CONFIG_DIR"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://llm-api.amd.com/Anthropic}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-dummy}"
# Secret is NOT baked in. Export AMD_LLM_GATEWAY_SUBSCRIPTION_KEY before running
# (see REPRODUCE.md "Secrets needed"). Fail loudly rather than sending an empty header.
: "${AMD_LLM_GATEWAY_SUBSCRIPTION_KEY:?set AMD_LLM_GATEWAY_SUBSCRIPTION_KEY (AMD LLM gateway APIM key)}"
export ANTHROPIC_CUSTOM_HEADERS="${ANTHROPIC_CUSTOM_HEADERS:-Ocp-Apim-Subscription-Key: $AMD_LLM_GATEWAY_SUBSCRIPTION_KEY}"
# Triton JIT cache must not land on the root_squashed NFS home.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH/triton_cache}"
mkdir -p "$TRITON_CACHE_DIR"

echo "==> launching forge-loop (gpu=$GPU_TYPE/$GPU_TARGET, max_hours=$MAX_HOURS, model=$FORGE_MODEL)"
exec kernel-agents forge-loop \
  --kernel "$WORKSPACE/sampler_softmax_kernel.py" \
  --driver "$WORKSPACE/driver.py" \
  --workspace "$WORKSPACE" \
  --experiments-dir "$WORKSPACE/forge_experiments" \
  --result-json "$WORKSPACE/forge_experiments/forge_result.json" \
  --program-md-file "$WORKSPACE/program.md" \
  --fellow triton-fellow \
  --gpu-target "$GPU_TARGET" \
  --gpu-type "$GPU_TYPE" \
  --framework sglang \
  --operator-name sampler_vocab_softmax \
  --snr-threshold 30.0 \
  --max-hours "$MAX_HOURS" \
  --git-branch forge-optimize \
  --target-functions "sampler_softmax" \
  --model "$FORGE_MODEL"
