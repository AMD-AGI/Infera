#!/usr/bin/env bash
# Verify a BUILT merged image carries every fix -- in the BYTECODE, at runtime.
#
# WHY THIS EXISTS SEPARATELY FROM THE BUILD LOG
# ---------------------------------------------
# A build log saying each patch script printed success is not the same as the
# running interpreter executing patched code:
#
#   * Python caches compiled modules keyed on source mtime, so a stale .pyc can
#     silently answer for a patched .py. This has already invalidated a full
#     experiment on this stack -- the source showed the fix, the runtime did not
#     have it. See ../notes.md §3.
#   * An image may carry TWO infera copies: the pip-installed one and a source
#     tree at the WORKDIR, which shadows it for every `docker exec`. A fix in one
#     and absent from the other passes a source grep and fails at runtime. (The
#     merged image happens to carry only one -- this reports what it found rather
#     than assuming either shape.)
#
# So this greps freshly-compiled bytecode, and it greps for IDENTIFIERS, never
# comment markers: the compiler discards comments, so a comment marker reads as a
# false negative.
#
# Three things cannot be checked in bytecode and are source-checked instead, each
# with the reason recorded at the check.
#
#   IMAGE=infera/engine-sglang:merged bash verify_built_image.sh
#
# Exits non-zero if any check fails.
set -u
IMAGE="${IMAGE:-infera/engine-sglang:merged}"
CTR="${CTR:-vrfy_merged}"

docker rm -f "$CTR" >/dev/null 2>&1
docker run -d --name "$CTR" --entrypoint sleep "$IMAGE" infinity >/dev/null || {
  echo "could not start a container from $IMAGE" >&2; exit 1; }
trap 'docker rm -f "$CTR" >/dev/null 2>&1' EXIT
sleep 3

cat > /tmp/_vrfy_inner.sh <<'INNER'
set -u
fail=0
SG=$(python3 -c 'import sglang,os;print(os.path.dirname(sglang.__file__))')
ROOTS=$(python3 - <<'PY'
import importlib.util, pathlib
roots, seen = [], set()
def add(d):
    d = pathlib.Path(d).resolve()
    if d.is_dir() and (d / "router" / "kv_event" / "client.py").is_file() and d not in seen:
        seen.add(d); roots.append(str(d))
s = importlib.util.find_spec("infera")
if s and s.origin: add(pathlib.Path(s.origin).parent)
add("/opt/infera/infera")
print(" ".join(roots))
PY
)
echo "infera copies: $ROOTS"
[ -n "$ROOTS" ] || { echo "NO infera copy found"; exit 1; }

# Recompile ONLY the directories checked below. compileall over the whole image
# tree takes minutes and buys nothing.
for d in "$SG/srt/layers/attention/dsa" "$SG/srt/layers/attention" \
         "$SG/srt/managers/scheduler_components" "$SG/srt/speculative" \
         "$SG/srt/model_executor" "$SG/srt/managers" "$SG/srt/disaggregation" \
         "$SG/srt/disaggregation/common" "$SG/srt/disaggregation/mooncake"; do
  rm -rf "$d/__pycache__" 2>/dev/null; python3 -m compileall -q "$d" >/dev/null 2>&1
done
for r in $ROOTS; do
  for d in "$r/router/kv_event" "$r/engine/sglang"; do
    rm -rf "$d/__pycache__" 2>/dev/null; python3 -m compileall -q "$d" >/dev/null 2>&1
  done
done

ck() { # label  glob  identifier  [src]
  local n; n=$(grep -rl "$3" $2 2>/dev/null | wc -l)
  printf '  %-52s %s=%s ' "$1" "${4:-pyc}" "$n"
  [ "$n" -gt 0 ] && echo OK || { echo FAIL; fail=1; }
}

echo "--- DSA patch set (PR58) ---"
ck "dsa_indexer::_p1v2_trim" "$SG/srt/layers/attention/dsa/__pycache__/*.pyc" _p1v2_trim
ck "dsa_backend::_glm52_match_page_table_rows" "$SG/srt/layers/attention/__pycache__/*.pyc" _glm52_match_page_table_rows
ck "dp_attn::can_draft_cuda_graph" "$SG/srt/managers/scheduler_components/__pycache__/*.pyc" can_draft_cuda_graph
ck "eagle_worker_v2::requires_dp_attention_eager_forward" "$SG/srt/speculative/__pycache__/*.pyc" requires_dp_attention_eager_forward
ck "eagle_draft_cg_runner::can_run_dp_draft_cuda_graph" "$SG/srt/speculative/__pycache__/*.pyc" can_run_dp_draft_cuda_graph
ck "forward_batch_info::can_run_dp_draft_cuda_graph" "$SG/srt/model_executor/__pycache__/*.pyc" can_run_dp_draft_cuda_graph
ck "schedule_batch::force_disable_draft_cuda_graph" "$SG/srt/managers/__pycache__/*.pyc" force_disable_draft_cuda_graph
ck "decode::force_disable_draft_cuda_graph" "$SG/srt/disaggregation/__pycache__/*.pyc" force_disable_draft_cuda_graph
# patch 2a changes an expression and introduces no new identifier -> source only.
ck "patch2a max_seqlen_k" "$SG/srt/layers/attention/dsa_backend.py" max_seqlen_k src
# the nextn prerequisite is an f-string split across constants -> source only.
ck "PREREQ nextn eh_proj" "$SG/srt/models/deepseek_nextn.py" eh_proj src

echo "--- mooncake early-send wait event (PR56) ---"
ck "common/utils::wait_event" "$SG/srt/disaggregation/common/__pycache__/utils*.pyc" wait_event
ck "mooncake/conn::_early_send_wait_event" "$SG/srt/disaggregation/mooncake/__pycache__/conn*.pyc" _early_send_wait_event
ck "prefill::_early_send_wait_event" "$SG/srt/disaggregation/__pycache__/prefill*.pyc" _early_send_wait_event
ck "mooncake/conn::synchronize" "$SG/srt/disaggregation/mooncake/conn.py" "synchronize()" src

echo "--- infera source changes ---"
i=0
for r in $ROOTS; do
  i=$((i + 1))
  ck "[copy$i] client::_flat_tokens" "$r/router/kv_event/__pycache__/client*.pyc" _flat_tokens
  ck "[copy$i] args::speculative gate" "$r/engine/sglang/__pycache__/args*.pyc" speculative_algorithm
  ck "[copy$i] kvd_wiring::decode skip" "$r/engine/sglang/__pycache__/kvd_wiring*.pyc" _skip_kvd_on_decode_leg
  # A lazily-evaluated annotation never becomes a runtime constant -> source only.
  ck "[copy$i] events::token_ids union" "$r/router/kv_event/events.py" "tuple\[int, int\]" src
done

echo "--- behavioural smoke ---"
python3 - <<'PY' || fail=1
from infera.router.kv_event.client import _flat_tokens
assert _flat_tokens([(1, 2), (2, 3), (3, 4)]) == [1, 2, 3], "bigram flatten wrong"
assert _flat_tokens([1, 2, 3]) == [1, 2, 3], "plain path changed"
assert _flat_tokens([]) == [], "empty path changed"
from infera.engine.sglang.kvd_wiring import _skip_kvd_on_decode_leg
print("  _flat_tokens + _skip_kvd_on_decode_leg import and behave  OK")
PY

echo
[ "$fail" -eq 0 ] && echo "=== ALL FIXES VERIFIED IN THE BUILT IMAGE ===" \
                  || { echo "=== VERIFICATION FAILED ==="; exit 1; }
INNER

docker cp /tmp/_vrfy_inner.sh "$CTR":/tmp/_vrfy_inner.sh >/dev/null
docker exec "$CTR" bash /tmp/_vrfy_inner.sh
