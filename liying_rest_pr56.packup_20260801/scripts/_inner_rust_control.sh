#!/usr/bin/env bash
# Runs INSIDE the container. The real gate for P1.
#
# WHY THIS AND NOT THE LIVE ROUTER A/B
# ------------------------------------
# The first attempt ran the unpatched and patched infera-router binaries against
# the live PD pair. It did not discriminate: the *unpatched* binary read
# cache_hits=51, because the live prefill leg runs WITHOUT --speculative-algorithm
# (MTP is on the decode leg; `is_eagle` only switches the radix key to bigrams on
# a leg that has it). Plain ints decode fine on both binaries, so the comparison
# was vacuous -- a green result there would have proved nothing.
#
# Bringing an MTP prefill leg up costs a ~9 min cold start per leg and would
# still only exercise the same decoder. Instead drive the decoder over a REAL
# ZMQ socket with the exact bigram wire shape, and run it BOTH WAYS:
#
#   unpatched kv_event.rs -> must FAIL (this is the bug)
#   patched   kv_event.rs -> must PASS
#
# A test that only passes after the fix is evidence; one that never had a chance
# to fail is not.
set -u
cd /opt/infera/rust || exit 1
export PATH="$HOME/.cargo/bin:$PATH"
LIBCLANG_PATH="$(dirname "$(find /opt/rocm* /usr/lib /usr/lib64 -name 'libclang.so*' 2>/dev/null | head -1)")"
case "$LIBCLANG_PATH" in ""|".") LIBCLANG_PATH=/opt/rocm/llvm/lib;; esac
export LIBCLANG_PATH

SRC=router/src/kv_event.rs
cp "$SRC" /tmp/kv_event.patched.rs

echo "############ CONTROL: revert as_u32_any, expect FAILURE ############"
python3 - "$SRC" <<'PY'
import sys, re
p = sys.argv[1]
s = open(p).read()
# Put back the pre-fix as_u32_vec (ints only) and drop the helper, i.e. exactly
# the code that shipped before 01b0534.
patched = """fn as_u32_vec(v: &rmpv::Value) -> Vec<u32> {
    v.as_array()
        .map(|a| a.iter().filter_map(as_u32_any).collect())
        .unwrap_or_default()
}"""
original = """fn as_u32_vec(v: &rmpv::Value) -> Vec<u32> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| as_u64_any(x).map(|n| n as u32))
                .collect()
        })
        .unwrap_or_default()
}"""
assert patched in s, "patched as_u32_vec not found -- did the file change?"
s = s.replace(patched, original)
# Remove the helper and its doc comment.
s = re.sub(r"/// One flat token id.*?\n}\n\n", "", s, flags=re.S)
assert "fn as_u32_any" not in s, "helper still present after revert"
open(p, "w").write(s)
print("  reverted: as_u32_any removed, as_u32_vec back to ints-only")
PY

# The in-crate unit test references the bigram case too; keep only the
# integration test in this control run so the failure is unambiguous.
cargo test --offline --release --test kv_event_zmq 2>&1 | tail -20
echo "control_rc=${PIPESTATUS[0]}"

echo
echo "############ TREATMENT: restore the fix, expect PASS ############"
cp /tmp/kv_event.patched.rs "$SRC"
grep -c as_u32_any "$SRC"
cargo test --offline --release --test kv_event_zmq 2>&1 | tail -20
echo "treatment_rc=${PIPESTATUS[0]}"
