#!/usr/bin/env bash
# Prove the new tests actually fail on pre-fix code.
#
# A test that passes both before and after a fix is worthless. This reverts each
# infera fix in place, re-runs its test module, asserts it FAILS, then restores.
# Runs INSIDE the engine container (needs sglang importable).
set -u
cd /opt/infera

pass=0
fail=0

check() {  # name, file, test-module, revert-python
  local name="$1" file="$2" mod="$3" revert="$4"
  cp "$file" /tmp/$(basename "$file").fixed
  python3 -c "$revert" || { echo "  [$name] REVERT SCRIPT BROKE"; fail=1; return; }
  if python3 -m pytest "$mod" -q >/tmp/rev_$name.log 2>&1; then
    echo "  [$name] ✗ tests PASSED on reverted code — they do not test the fix"
    fail=1
  else
    echo "  [$name] ✓ tests fail on reverted code ($(grep -cE '^FAILED' /tmp/rev_$name.log) failed)"
    pass=$((pass + 1))
  fi
  cp /tmp/$(basename "$file").fixed "$file"
  find . -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
}

echo "=== revert-and-fail checks ==="

check bigram infera/router/kv_event/client.py tests/unit/router/test_kv_event_bigram.py '
p = "infera/router/kv_event/client.py"
s = open(p).read()
s = s.replace("        tokens = _flat_tokens(ev.token_ids)\n        n = len(tokens) // bs",
              "        n = len(ev.token_ids) // bs")
s = s.replace("            chunk = tokens[i * bs : (i + 1) * bs]",
              "            chunk = ev.token_ids[i * bs : (i + 1) * bs]")
open(p, "w").write(s)
'

check decode_radix infera/engine/sglang/args.py tests/engine/sglang/test_decode_radix_vs_speculative.py '
p = "infera/engine/sglang/args.py"
s = open(p).read()
i = s.index("        # SGLang rejects the decode radix cache outright")
marker = "            remaining.append(\"--disaggregation-decode-enable-radix-cache\")"
j = s.index(marker, i) + len(marker)
s = s[:i] + "        remaining.append(\"--disaggregation-decode-enable-radix-cache\")" + s[j:]
open(p, "w").write(s)
'

check decode_kvd infera/engine/sglang/kvd_wiring.py tests/engine/sglang/test_decode_leg_gating.py '
p = "infera/engine/sglang/kvd_wiring.py"
s = open(p).read()
i = s.index("def _skip_kvd_on_decode_leg")
j = s.index("async def awire_infera_kvd_backend")
s = s[:i] + s[j:]
s = s.replace("    if _skip_kvd_on_decode_leg(args):\n        return\n", "")
open(p, "w").write(s)
'

echo
if [ "$fail" -ne 0 ]; then
  echo "REVERT CHECK FAILED — at least one test does not exercise its fix" >&2
  exit 1
fi
echo "=== all $pass fixes are covered by tests that fail without them ==="
