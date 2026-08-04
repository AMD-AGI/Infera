#!/bin/bash
# Run the CUSTOMER's replay_caseA.sh UNMODIFIED against the deployment the
# example kit brought up. Everything here is environment only.
#
# Runs on the JUMP HOST.
#
# OUT is deliberately INSIDE $HERE. replay_caseA.sh mounts only $HERE into the
# aiperf container, so an OUT outside it writes the artifacts into the container
# namespace where they die on exit — and the sweep then prints FAILED for a run
# that actually succeeded. The customer kit's own defect; avoided rather than
# patched, because their code must stay untouched.
set -uo pipefail
HERE=/root/glm52_example_agentx/bench

export URL="${URL:-http://10.2.122.78:8100}"   # the kit's router, never a leg's own port
export SERVED="${SERVED:-glm5.2-mxfp4}"        # our served-model-name; their default is wrong for us
export TOK="${TOK:-$HERE/tokenizer}"           # the container cannot see /mnt/vast
export CONCS="${CONCS:-8}"                     # conc=1 is unsupported by the scenario
export DUR="${DUR:-900}"                       # the scenario's ENFORCED minimum; their default 300 is rejected
export IMG="${IMG:-aiperf-agentx:v1.0}"
export OUT="${OUT:-$HERE/results}"

echo "=== $(date -u +%FT%TZ) launching customer replay_caseA.sh ==="
echo "URL=$URL SERVED=$SERVED CONCS='$CONCS' DUR=$DUR IMG=$IMG OUT=$OUT"
md5sum "$HERE/replay_caseA.sh"    # 7cde1afc627c7e4868eac0fd13741baa == unmodified

cd "$HERE" && bash ./replay_caseA.sh
