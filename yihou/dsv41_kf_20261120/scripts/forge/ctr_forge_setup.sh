#!/bin/bash
# INSIDE the Forge container (serving image): install Hyperloom (AKLR branch) [forge] + ruff, the Claude CLI, check the
# gateway, run the skill preflight. Env: FORGE_ROOT (contains hyperloom/, aklr/, ws/ and runs/), credentials via --env-file.
set -uo pipefail
R=${FORGE_ROOT:?}
export HOME=$R/home; mkdir -p "$HOME"
cd "$R"
echo "== pip install hyperloom[forge] ruff"
python3 -m pip install -q -e "$R/hyperloom[forge]" "ruff==0.15.12" > "$R/logs/pip.log" 2>&1 || { tail -20 "$R/logs/pip.log"; exit 3; }
python3 -c "import kernelforge, torch; print('kernelforge', kernelforge.__file__, 'torch', torch.__version__)"
echo "== claude CLI"
if ! command -v claude >/dev/null && [ ! -x "$HOME/.local/bin/claude" ]; then
  curl -fsSL --connect-timeout 20 https://claude.ai/install.sh | bash > "$R/logs/claude-install.log" 2>&1 || { tail -5 "$R/logs/claude-install.log"; exit 4; }
fi
export PATH=$HOME/.local/bin:$PATH
claude --version
echo "== gateway check"
timeout 120 claude -p "reply with the single word ok" --model "${CLAUDE_MODEL:?}" 2>&1 | tail -2
echo "== forge-loop help"
python3 -m kernelforge.cli forge-loop --help 2>&1 | head -3
echo "SETUP DONE"
