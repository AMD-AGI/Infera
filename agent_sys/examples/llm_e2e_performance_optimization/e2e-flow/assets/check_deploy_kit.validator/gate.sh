#!/usr/bin/env bash
# Gate `check_deploy_kit` in both directions against the real sealed kit.
#
# **A validator shown only to reject may reject everything; a validator shown
# only to accept is worse.** Both directions, or it is not gated. This script is
# the worked example of that for this package — copy its shape, not its rules.
#
#     bash assets/check_deploy_kit.validator/gate.sh [<scratch dir>]
#
# It runs in about a second, touches no GPU, and writes only under the scratch
# directory (default `/tmp/check_deploy_kit_gate.$$`). It reads the sealed kit
# from `$E2E_MOCK_ROOT` and never writes there.
#
# The positive fixture is the sealed 38-file kit **plus the two things this
# package added after it was sealed**: `codes/environment.yaml` (CONTRACT.md §2)
# and the five `runtime_contract` parameters. That is the whole delta between a
# kit the previous stage produced and one this stage accepts, and keeping it
# visible here is the point — a reader can see exactly what was added and
# nothing about the inherited rules is taken on trust.
#
# The negative fixture plants **thirteen** faults, one per rule this validator
# owns, and asserts each is reported. They are listed in the `want` array below
# with the rule each exercises.
set -eu

: "${E2E_MOCK_ROOT:=/shared_nfs/yihou/agent_sys/cheat_for_mock}"
SEALED="$E2E_MOCK_ROOT/stage1-deploy/deploy_kit/content"
PKG="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${1:-/tmp/check_deploy_kit_gate.$$}"

[ -d "$SEALED" ] || { echo "no sealed kit at $SEALED; set E2E_MOCK_ROOT" >&2; exit 2; }
mkdir -p "$WORK"

# A validation zone is a directory holding args.json / inputs.json /
# materials.json, with the body run in it (`validator/phase.py:236`).
zone() {
  mkdir -p "$1"
  printf '%s\n' '{"layout": "deploy_kit.layout"}' > "$1/args.json"
  printf '%s\n' '["h1"]'                          > "$1/inputs.json"
  printf '{"h1": "%s"}\n' "$2"                    > "$1/materials.json"
}

run() { (cd "$1" && AGENT_SYS_DEMO_PACKAGE="$PKG" python3 "$PKG/assets/check_deploy_kit.validator/check.py"); }

# --------------------------------------------------------------------- positive
GOOD="$WORK/good"
rm -rf "$GOOD"; cp -a "$SEALED" "$GOOD"
PACKUP="$(find "$GOOD/items/codes" -maxdepth 1 -type d -name '*.packup_*' | head -1)"

# The positive fixture is built by **the same script the mock run uses**,
# `../deploy_and_prove.task/mock_adapt.sh`, so "a conforming kit" cannot mean two
# different things in the two places that decide it. It calls `env_render.py`,
# which is the real producer's path and validates before it writes.
#
# `env_render.build()` reads the run's `E2E_*` variables; outside a zone there
# are none, so the facts the sealed run recorded are supplied here. They are that
# run's own measured values, not invented ones.
export E2E_NODE=crsuse2-m2m-079
export E2E_NODE_IP=127.0.0.1
export E2E_IMAGE=infera/engine-sglang:gfx950-local
export E2E_MODEL_NAME=Qwen/Qwen3.6-27B
export E2E_MODEL_PATH=/shared_nfs/yihou/models/Qwen3.6-27B
export E2E_CONTAINER=dbg_deploy_sgl_20260902-113414-81355
export E2E_TRANSPORT=spur
export E2E_TP=1
export E2E_CTX=32768
export E2E_PORT_ROUTER=8106
export AGENT_SYS_DEMO_PACKAGE="$PKG"
# **The fixture supplies its own digest, and that is the honest form.**
# `mock_adapt.sh` normally *discovers* `image_id` on the node, because a digest
# asserted rather than measured is what broke rung 0 — the record named an image
# that exists nowhere. This gate has no node and builds a fixture rather than a
# deployment, so it says the digest explicitly instead of pretending to have
# looked. The value is the sealed run's own, which is what the fixture is of.
export MOCK_IMAGE_ID=sha256:92ed065bdc3958bdb62fdb5c2c4b88ad9fa45c9b355b763f3098a6185b0668e6
# No `replayed_from` here: this fixture stands for a kit produced by a real
# bring-up, so it must be held to the strict `rendered_from` comparison. Setting
# it would make the gate stop testing the rule it exists to test.
export MOCK_REPLAYED_FROM=""

bash "$PKG/assets/deploy_and_prove.task/mock_adapt.sh" "$GOOD"

zone "$WORK/zone_good" "$GOOD"
echo "=== positive: the sealed kit, adapted, must PASS ==="
run "$WORK/zone_good"
grep -q '"h1": true' "$WORK/zone_good/verdict.json" \
  || { echo "GATE FAILED: the real kit was refused"; exit 1; }

# --------------------------------------------------------------------- negative
BAD="$WORK/bad"
rm -rf "$BAD"; cp -a "$GOOD" "$BAD"
PACKUP="$(find "$BAD/items/codes" -maxdepth 1 -type d -name '*.packup_*' | head -1)"

python3 - "$BAD/items/codes/environment.yaml" <<'PY'
import sys, yaml
p = sys.argv[1]; d = yaml.safe_load(open(p))
d['fixed']['gpu_arch'] = 'MI355X'      # 1. a product, not an architecture
del d['fixed']['image_id']             # 2. a floating tag is not a reproduction
del d['runtime']['endpoint']           # 3. required by the schema
d['fixed']['image'] = 'other/img:v9'   # 4. environment.md renders a different image
# 12. more cards taken than the node has. The sealed record carries no
# `gpu_devices` at all (it predates the field), so the fault has to be PLANTED
# rather than mutated -- which is itself the reason `on_absent` is `skip` in the
# layout: absence is the sealed kit's honest state, and only a present-and-wrong
# list is a lie this validator can catch today.
d['fixed']['gpu_count'] = 8
d['fixed']['gpu_devices'] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
yaml.safe_dump(d, open(p, 'w'), sort_keys=False)
PY
# 13. the handoff's own README, whose absence the seal refuses and this layout
# could not see until 2026-09-04. Planted by REMOVAL, which is how it occurred:
# two real bring-ups produced kits complete in items/ and missing this file.
rm -f "$BAD/README.md"
printf '{"model": "/models/qwen3.6-27b"}\n' > "$PACKUP/results/bad_model_id.json"   # 5
rm -f "$PACKUP/results/router_workers.json" "$PACKUP/results/verification.json"     # 6
printf '%s\n' '#!/bin/sh' 'CTR_NAME=dbg_deploy_sgl' \
  'docker run -d --name "$CTR_NAME" --publish 8106:8106 img' > "$PACKUP/scripts/bad.sh"  # 7, 8
python3 - "$PACKUP/REPRODUCE.md" <<'PY'
import re, sys
p = sys.argv[1]; t = open(p).read()
open(p, 'w').write(re.sub(r'(?im)^(\s*#{1,6}\s*)Expected\s+output\b', r'\1Results you may see', t))
PY
                                                                                   # 9
rm -f "$PACKUP/scripts/wait_ready.sh"                                              # 10
# 11 — removed from **every** file under scripts/, not only env.sh. The
# adaptation now also installs the stub kit, whose `stub_env.sh` declares the
# same parameters, so stripping one file left the contract satisfied by the
# other and this planted fault silently stopped firing.
for f in "$PACKUP"/scripts/*.sh; do
  grep -v 'E2E_KIT_ENGINE_EXTRA_ARGS' "$f" > "$f.new" && mv "$f.new" "$f"
done

zone "$WORK/zone_bad" "$BAD"
echo
echo "=== negative: thirteen planted faults, each must be reported ==="
run "$WORK/zone_bad" > "$WORK/bad.out" 2>&1 || true
cat "$WORK/bad.out"

grep -q '"h1": false' "$WORK/zone_bad/verdict.json" \
  || { echo "GATE FAILED: a kit with eleven faults passed"; exit 1; }

# Each entry is (rule it exercises) -> a fragment that must appear in the output.
want=(
  "gpu_arch"                       # 1  schema: pattern
  "image_id"                       # 2  schema: required
  "endpoint"                       # 3  schema: required, nested
  "does not render fixed.image"    # 4  environment.md is a rendering
  "bad_model_id.json"              # 5  served name is a filesystem path
  "router-side reading"            # 6  mode read back from two components
  "CTR_NAME is fixed here"         # 7  frozen and bound
  "binds a literal"                # 8  a literal in a binding flag
  "Expected output"                # 9  the reproducer's only criterion
  "wait_ready.sh: missing"         # 10 the readiness entrypoint
  "E2E_KIT_ENGINE_EXTRA_ARGS"      # 11 the runtime contract
  "cannot take more cards"         # 12 record-internal: len(gpu_devices) <= gpu_count
  "content/README.md: missing"     # 13 the handoff's own README -- the seal refuses without it
)
missed=0
for fragment in "${want[@]}"; do
  grep -qF "$fragment" "$WORK/bad.out" || { echo "GATE FAILED: nothing reported for: $fragment"; missed=1; }
done
[ "$missed" = 0 ] || exit 1

echo
echo "GATE PASSED: the real kit is accepted and all ${#want[@]} planted faults are reported."
echo "scratch: $WORK"
