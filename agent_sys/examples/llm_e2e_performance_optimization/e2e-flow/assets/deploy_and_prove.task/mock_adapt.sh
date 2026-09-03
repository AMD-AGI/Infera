#!/usr/bin/env bash
# MOCK-MAP (A) + (I): make the sealed `deploy_kit` conform to what this package
# requires of one. **An adaptation is a step after the copy, not a variant of
# the copy** — `mock.sh` puts the sealed bytes down verbatim, on purpose, and
# this adds only what did not exist when they were sealed.
#
#     mock_adapt.sh <content dir>          # the handoff's content/, holding items/
#
# Two gaps, both real and both the schema doing its job rather than an oversight:
#
#   (A) `codes/environment.yaml` did not exist. No sealed handoff carries the
#       document `environment.schema.json` describes — the *fields* existed as
#       `items/env/deployment.json`, but three the schema requires (`gpu_arch`,
#       `gpu_count`, `image_id`) were never recorded as data.
#   (I) The runtime contract did not exist. The sealed kit spells all six
#       concepts `DK_*`, so the shim below is a **rename**, not new behaviour —
#       each line falls back to the `DK_` name the kit already honours, and the
#       kit's own defaults still apply when neither is set.
#
# **One definition, two callers.** `../check_deploy_kit.validator/gate.sh` builds
# its positive fixture with this same script, so the thing the mock run produces
# and the thing the both-directions gate calls "a conforming kit" cannot drift
# apart. That is the same reason `deploy_kit.layout.yaml` is one file.
#
# `env_render.py --new` writes the record rather than a heredoc here, because it
# is the **real producer's** path (`readme.md` STEPS step 5) and it validates
# before it writes. A mock that wrote the document a different way would be
# testing a different producer.
set -eu

CONTENT="${1:?usage: mock_adapt.sh <content dir>}"
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:-$(cd "$(dirname "$0")/../.." && pwd)}}"

CODES="$CONTENT/items/codes"
[ -d "$CODES" ] || { echo "mock_adapt: no $CODES — was mock.sh run?" >&2; exit 1; }

PACKUP="$(find "$CODES" -maxdepth 1 -type d -name '*.packup_*' | head -1)"
[ -n "$PACKUP" ] || { echo "mock_adapt: no <name>.packup_<YYYYMMDD> under $CODES" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# (A) the environment record.
#
# The three facts a variable cannot carry are **the sealed run's own measured
# values**, read off its `environment.md`: gfx950, eight cards, and the image
# digest it recorded. Inventing plausible ones would make the mock a fiction
# about hardware; these are a real bring-up's numbers, which is the whole
# premise of mocking from sealed artefacts rather than synthesising.
: "${MOCK_GPU_ARCH:=gfx950}"
: "${MOCK_GPU_COUNT:=8}"
: "${MOCK_IMAGE_ID:=sha256:92ed065bdc3958bdb62fdb5c2c4b88ad9fa45c9b355b763f3098a6185b0668e6}"
: "${MOCK_ENDPOINT:=http://${E2E_NODE_IP:-127.0.0.1}:${E2E_PORT_ROUTER:-8101}}"

# **Not bare `python3`.** `env_render.py` validates before it writes, and
# `../lib/schema.py:109` imports `referencing` to do it. Measured on this host:
# `/usr/bin/python3` has `yaml` and `jsonschema` but **not `referencing`**, so a
# body that takes the policy `PATH` dies inside the validator with a
# `ModuleNotFoundError` after `mock.sh` has already written 38 files — which is
# how this was found. Probe for one that can, and say so if none can.
PY=""
for candidate in "${AGENT_SYS_DEMO_PYTHON:-}" python3 /usr/bin/python3; do
  [ -n "$candidate" ] || continue
  if "$candidate" -c 'import jsonschema, referencing, yaml' >/dev/null 2>&1; then PY="$candidate"; break; fi
done
if [ -z "$PY" ]; then
  echo "mock_adapt: no interpreter here can import jsonschema, referencing and yaml;" >&2
  echo "env_render.py validates before it writes and cannot run without them." >&2
  exit 2
fi

"$PY" "$PKG/assets/lib/env_render.py" --new --content-type code --out "$CONTENT" \
  --set "fixed.gpu_arch=${MOCK_GPU_ARCH}" \
  --set "fixed.gpu_count=${MOCK_GPU_COUNT}" \
  --set "fixed.image_id=${MOCK_IMAGE_ID}" \
  --set "runtime.endpoint=${MOCK_ENDPOINT}"

# --------------------------------------------------------------------------- #
# (I) the runtime contract.
#
# Appended to `scripts/env.sh`, which every other script in the sealed kit
# sources. Idempotent: a second run of this adaptation must not append a second
# copy, because a mock run that is re-driven is the normal case.
ENVSH="$PACKUP/scripts/env.sh"
[ -f "$ENVSH" ] || { echo "mock_adapt: no $ENVSH to adapt" >&2; exit 1; }

if ! grep -q 'E2E_KIT_RUN_TAG' "$ENVSH"; then
  cat >> "$ENVSH" <<'EOF'

# --- deploy_kit.layout.yaml runtime_contract (added by mock_adapt.sh) --------
# A rename, not new behaviour: each falls back to the `DK_` name this kit already
# honours, so an unset caller gets the sealed run's own defaults byte for byte.
: "${E2E_KIT_RUN_TAG:=${DK_RUN_TAG:-$(date +%Y%m%d-%H%M%S)-$$}}"
: "${E2E_KIT_PORT_BASE:=${DK_PORT_BAND_LO:-8100}}"
: "${E2E_KIT_WORK_ROOT:=${DK_WORK_ROOT:-/mnt/m2m_nobackup/yihou/deploy}}"
: "${E2E_KIT_ENGINE_EXTRA_ARGS:=}"
: "${E2E_KIT_ENGINE_EXTRA_ENV:=}"
: "${E2E_KIT_ROUTER_EXTRA_ARGS:=}"
EOF
fi

echo "mock_adapt: ${PACKUP##*/} now carries codes/environment.yaml and the runtime contract" >&2
