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
# The negative fixture plants **eleven** faults, one per rule this validator
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

cat > "$GOOD/items/codes/environment.yaml" <<'YAML'
schema_version: 1
fixed:
  node: crsuse2-m2m-079
  gpu_arch: gfx950
  gpu_count: 8
  image: infera/engine-sglang:gfx950-local
  image_id: sha256:92ed065bdc3958bdb62fdb5c2c4b88ad9fa45c9b355b763f3098a6185b0668e6
  dockerfile: null
  rocm: 7.2.0
  model_name: Qwen/Qwen3.6-27B
  model_path: /shared_nfs/yihou/models/Qwen3.6-27B
  served_model_name: Qwen/Qwen3.6-27B
  tp_size: 1
  deploy_mode: mix
  context_length: 32768
  scripts: {package: e2e-flow, commit: 6d7a3d3, entrypoints: [scripts/deploy.sh]}
runtime:
  container: dbg_deploy_sgl_20260902-113414-81355
  ports: {router: 8106, worker: 8107, etcd: 8105}
  endpoint: http://127.0.0.1:8106
  transport: spur
  started_at: '2026-09-02T11:34:14Z'
YAML

# The runtime contract, as a kit produced by this package's own producer carries
# it. The sealed kit already has all five concepts under its own `DK_*` names, so
# this is a rename and not new behaviour.
cat >> "$PACKUP/scripts/env.sh" <<'EOF'

# --- deploy_kit.layout.yaml runtime_contract -------------------------------
: "${E2E_KIT_RUN_TAG:=${DK_RUN_TAG:-$(date +%Y%m%d-%H%M%S)-$$}}"
: "${E2E_KIT_PORT_BASE:=${DK_PORT_BAND_LO:-8100}}"
: "${E2E_KIT_WORK_ROOT:=${DK_WORK_ROOT:-/mnt/m2m_nobackup/yihou/deploy}}"
: "${E2E_KIT_ENGINE_EXTRA_ARGS:=}"
: "${E2E_KIT_ENGINE_EXTRA_ENV:=}"
EOF

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
d['fixed']['node'] = 'some-other-node' # 4. environment.md now disagrees
yaml.safe_dump(d, open(p, 'w'), sort_keys=False)
PY
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
grep -v 'E2E_KIT_ENGINE_EXTRA_ARGS' "$PACKUP/scripts/env.sh" > "$PACKUP/scripts/env.new"
mv "$PACKUP/scripts/env.new" "$PACKUP/scripts/env.sh"                              # 11

zone "$WORK/zone_bad" "$BAD"
echo
echo "=== negative: eleven planted faults, each must be reported ==="
run "$WORK/zone_bad" > "$WORK/bad.out" 2>&1 || true
cat "$WORK/bad.out"

grep -q '"h1": false' "$WORK/zone_bad/verdict.json" \
  || { echo "GATE FAILED: a kit with eleven faults passed"; exit 1; }

# Each entry is (rule it exercises) -> a fragment that must appear in the output.
want=(
  "gpu_arch"                       # 1  schema: pattern
  "image_id"                       # 2  schema: required
  "endpoint"                       # 3  schema: required, nested
  "does not render fixed.node"     # 4  environment.md is a rendering
  "bad_model_id.json"              # 5  served name is a filesystem path
  "router-side reading"            # 6  mode read back from two components
  "CTR_NAME is fixed here"         # 7  frozen and bound
  "binds a literal"                # 8  a literal in a binding flag
  "Expected output"                # 9  the reproducer's only criterion
  "wait_ready.sh: missing"         # 10 the readiness entrypoint
  "E2E_KIT_ENGINE_EXTRA_ARGS"      # 11 the runtime contract
)
missed=0
for fragment in "${want[@]}"; do
  grep -qF "$fragment" "$WORK/bad.out" || { echo "GATE FAILED: nothing reported for: $fragment"; missed=1; }
done
[ "$missed" = 0 ] || exit 1

echo
echo "GATE PASSED: the real kit is accepted and all ${#want[@]} planted faults are reported."
echo "scratch: $WORK"
