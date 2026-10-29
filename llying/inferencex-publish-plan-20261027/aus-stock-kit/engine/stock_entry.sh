#!/usr/bin/env bash
# Container entry on the stock image: do at start what the InferenceX job does
# (install Infera, patch SGLang), then exec the worker or router command.
# The kit is bind-mounted read-only at /kit. STOCK_PATCHES=0 skips the SGLang
# patches (the router needs none).
set -euo pipefail

# The runtime's 1024 soft limit runs out at a few hundred router connections.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

pip install --quiet --no-cache-dir --no-deps /kit/artifacts/*.whl
install -m 0755 /kit/artifacts/infera-router /usr/local/bin/infera-router
python3 -c "import shutil, sys, infera.engine.sglang; sys.exit(shutil.which('infera-router') is None)"

[[ "${STOCK_PATCHES:-1}" == 0 ]] || bash /kit/patches/apply.sh

exec "$@"
