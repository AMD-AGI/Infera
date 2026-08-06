#!/bin/bash
# Invoke the CUSTOMER's replay_caseA.sh UNMODIFIED. Everything here is env only.
#
#   SERVED  our served-model-name is glm5.2-mxfp4, not GLM-5.2-MXFP4
#   TOK     the script mounts only $HERE (+ /models, /shared_nfs), so the
#           tokenizer is staged into $HERE/tokenizer
#   OUT     MUST live under $HERE. The customer script mounts only $HERE into
#           the aiperf container, so an --output-artifact-dir outside it lands in
#           the CONTAINER namespace and dies with the container -- the run then
#           records FAILED in summary.csv even though it completed.
#   DUR     900 is the scenario's enforced minimum (inferencex_agentx_mvp.py);
#           the script's own default of 300 would be rejected by the validator
#   CONCS   operator decision: 8 and 16 (conc=1 unsupported by the scenario)
#   IMG     locally built; upstream's default rocm/atom-dev:latest lacks the fork
set -uo pipefail
HERE=/shared_nfs/yihou_agentx_caseA/bench
export URL="${URL:-http://10.245.145.242:8190}"
export SERVED="${SERVED:-glm5.2-mxfp4}"
export TOK="${TOK:-$HERE/tokenizer}"
export CONCS="${CONCS:-8 16}"
export DUR="${DUR:-900}"
export IMG="${IMG:-aiperf-agentx:v1.0}"
export OUT="${OUT:-$HERE/results}"

echo "=== $(date -u +%FT%TZ) launching customer replay_caseA.sh ==="
echo "URL=$URL SERVED=$SERVED CONCS='$CONCS' DUR=$DUR IMG=$IMG OUT=$OUT"
md5sum "$HERE/replay_caseA.sh"   # prove the customer script is unmodified
cd "$HERE" && bash ./replay_caseA.sh
