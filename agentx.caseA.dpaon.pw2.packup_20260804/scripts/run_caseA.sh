#!/bin/bash
# Invoke the CUSTOMER's replay_caseA.sh unmodified. Everything here is env only.
#
# Why each value differs from the script's default:
#   SERVED  our served-model-name is glm5.2-mxfp4, not GLM-5.2-MXFP4
#   TOK     the script mounts only $HERE, /models, /shared_nfs — not /mnt/vast,
#           so the tokenizer files are staged into $HERE/tokenizer
#   CONCS   operator decision: 8 and 16 (conc=1 is unsupported by the scenario)
#   DUR     900 is the scenario's enforced minimum (inferencex_agentx_mvp.py:33);
#           the script's default of 300 would be rejected by the validator
#   IMG     our locally built image; upstream's default rocm/atom-dev:latest does
#           not have the aiperf agentx fork installed
set -uo pipefail
HERE=/root/agentx_20260803/bench

export URL="${URL:-http://10.2.122.78:8100}"
export SERVED="${SERVED:-glm5.2-mxfp4}"
export TOK="${TOK:-$HERE/tokenizer}"
export CONCS="${CONCS:-8 16}"
export DUR="${DUR:-900}"
export IMG="${IMG:-aiperf-agentx:v1.0}"
export OUT="${OUT:-/root/agentx_20260803/results}"

echo "=== $(date -u +%FT%TZ) launching customer replay_caseA.sh ==="
echo "URL=$URL SERVED=$SERVED CONCS='$CONCS' DUR=$DUR IMG=$IMG"
md5sum "$HERE/replay_caseA.sh"          # prove the script is unmodified

cd "$HERE" && bash ./replay_caseA.sh
