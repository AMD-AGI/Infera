#!/usr/bin/env bash
# Experiment 01 — sglang arg-compatibility matrix. NO GPU work, ~2 min.
# Proves WHY kvd needs kvaware on a PD decode leg.
set -uo pipefail
JUMP="${JUMP:-root@149.28.124.225}"; NODE="${NODE:-chi2879}"
IMAGE="${IMAGE:-infera/engine-sglang:pd-unified}"
CTR="${CTR:-argchk_$$}"   # unique name: never collide with someone else's container
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/../results}"

J(){ ssh -o StrictHostKeyChecking=no "$JUMP" "ssh -o StrictHostKeyChecking=no $NODE '$1'" 2>&1 | grep -v "^Warning: Permanently"; }
cleanup(){ echo "[cleanup] removing $CTR"; J "docker rm -f $CTR >/dev/null 2>&1"; }
trap cleanup EXIT

echo "== starting throwaway container $CTR on $NODE"
J "docker run -d --name $CTR --network=host -v /mnt/vast:/mnt/vast --entrypoint '' $IMAGE sleep infinity >/dev/null && echo ok" | grep -q ok \
  || { echo "FATAL: container start failed"; exit 1; }

echo "== staging argcheck.py"
ssh -o StrictHostKeyChecking=no "$JUMP" "ssh -o StrictHostKeyChecking=no $NODE 'cat > /tmp/argcheck.py'" < "$HERE/argcheck.py"
J "docker cp /tmp/argcheck.py $CTR:/tmp/argcheck.py >/dev/null"

echo "== running matrix (expect 1-4,6 OK; 5,7,8 FAIL)"
J "docker exec $CTR python3 /tmp/argcheck.py 2>&1 | grep -E '^\[(OK|FAIL|EXIT)'" | tee "$OUT/arg_matrix_verdicts.observed.txt"

echo
ok=$(grep -c '^\[OK' "$OUT/arg_matrix_verdicts.observed.txt" || true)
fail=$(grep -c '^\[FAIL' "$OUT/arg_matrix_verdicts.observed.txt" || true)
echo "== got $ok OK / $fail FAIL   (expected 5 OK / 3 FAIL)"
[ "$ok" = "5" ] && [ "$fail" = "3" ] && echo "== VERDICT: PASS (matches results/arg_matrix_verdicts.txt)" \
  || echo "== VERDICT: MISMATCH — compare against results/arg_matrix_verdicts.txt"
