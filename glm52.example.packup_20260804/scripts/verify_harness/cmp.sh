#!/usr/bin/env bash
# Per-leg attribution of the engine argv, plus the bench ladder — the two things
# the fixes are allowed to move. Everything else must be byte-identical.
d="$1"
for t in up_mtpoff up_mtpon leg_prefill leg_decode; do
  [ -f "$d/$t.trace" ] || continue
  grep "^### DOCKER" "$d/$t.trace" 2>/dev/null | while read -r l; do
    case "$l" in *disaggregation-mode*) ;; *) continue;; esac
    role=$(echo "$l" | grep -o 'disaggregation-mode..[a-z]*' | head -1 | grep -o '[a-z]*$')
    echo "$t/$role spec=$(echo "$l"|grep -c speculative-algorithm) dpa=$(echo "$l"|grep -c enable-dp-attention) ep=$(echo "$l"|grep -c ep.size)"
  done
done
echo "-- bench --"
grep -o "max-concurrency [0-9]* --num-prompts [0-9]*" "$d/bench.trace" 2>/dev/null
