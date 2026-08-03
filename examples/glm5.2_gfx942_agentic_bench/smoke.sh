#!/usr/bin/env bash
# Smoke-test the router: worker list, one chat request, and the decode leg's RDMA
# hand-off lines. Run on the prefill node inside the container.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

SERVER="${SERVER:-$ROUTER_URL}"
PROMPT="${PROMPT:-What is 127 * 31? Answer with the number only.}"
MAX_TOKENS="${MAX_TOKENS:-128}"

echo "== workers =="   # expect one prefill + one decode
curl -s "$SERVER/v1/workers" | python3 -m json.tool 2>/dev/null || curl -s "$SERVER/v1/workers"

echo; echo "== chat =="
# Build the JSON body with python so a PROMPT/MODEL containing quotes can't break it.
BODY="$(MODEL="$MODEL" PROMPT="$PROMPT" MAX_TOKENS="$MAX_TOKENS" python3 -c 'import json, os; print(json.dumps({"model": os.environ["MODEL"], "messages": [{"role": "user", "content": os.environ["PROMPT"]}], "max_tokens": int(os.environ["MAX_TOKENS"]), "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}))')"
curl -s "$SERVER/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY" \
  | python3 -m json.tool 2>/dev/null \
  || { echo "(request failed; check $LOG_DIR/{router,prefill,decode}.log)"; exit 1; }

echo; echo "== rdma hand-off (decode log) =="
if [[ -f "$LOG_DIR/decode.log" ]]; then
  grep -aE "GID index|installTransport|mooncake" "$LOG_DIR/decode.log" | tail -5 \
    || echo "(no hand-off lines yet)"
else
  echo "(decode.log is on $DECODE_NODE)"
fi

[[ "$KVD" == "1" ]] || exit 0

echo; echo "== kvd =="
# Both ROCm hicache patches decide whether the FIRST write-back survives: one
# faults the process, the other kills the scheduler. Assert them as the engine
# loaded them -- a stale .pyc shadowing patched source looks identical from the
# outside until it doesn't.
python3 -c "
from sglang.srt.mem_cache.pool_host import common, mla
print('host allocator :', common.ALLOC_MEMORY_FUNCS['cuda'].__name__,
      '(marker', getattr(common, 'GLM52_ROCM_HOST_ALLOC', 'MISSING') + ')')
# The staged write-back marker is a local, so look in the loaded code object of
# the method rather than in the module namespace.
gated = [
    cls.__name__
    for cls in vars(mla).values()
    if isinstance(cls, type)
    and 'GLM52_ROCM_STAGED_WRITE_BACK'
    in getattr(getattr(cls, '_init_write_back_staging_buffers', None),
               '__code__', type('x', (), {'co_varnames': ()})).co_varnames
]
print('staged JIT gate :', ', '.join(gated) if gated else 'MISSING -- the '
      'scheduler will die on the first reused prefix (see README gotcha 7)')
" 2>/dev/null || echo "patch markers  : (could not import sglang here)"

# Enough tokens to fill several pages: hicache writes to storage per page, so a
# prompt shorter than one page produces no kvd traffic and proves nothing.
FILLER="$(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 120)")"
BEFORE="$(python3 -m infera.kvd.statctl --socket "$KVD_SOCKET" | python3 -c 'import json,sys; print(json.load(sys.stdin)["sets_total"])')"
BODY="$(MODEL="$MODEL" PROMPT="$FILLER" python3 -c 'import json, os; print(json.dumps({"model": os.environ["MODEL"], "messages": [{"role": "user", "content": os.environ["PROMPT"] + " Reply with OK."}], "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}))')"
curl -s "$SERVER/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY" > /dev/null
sleep 5   # write-back to storage is asynchronous w.r.t. the response
python3 -m infera.kvd.statctl --socket "$KVD_SOCKET"
AFTER="$(python3 -m infera.kvd.statctl --socket "$KVD_SOCKET" | python3 -c 'import json,sys; print(json.load(sys.stdin)["sets_total"])')"

# A value larger than the biggest tablespace slot is REJECTED, not split, so L3
# stays empty while everything else looks healthy. Name it here or spend an
# afternoon on it later.
if grep -aq "value_exceeds_slot_bytes" "$LOG_DIR/kvd.log" 2>/dev/null; then
  echo "FAIL: kvd rejected pages that outgrew their slot -- raise KVD_TABLESPACE_POOLS"
  grep -a "value_exceeds_slot_bytes" "$LOG_DIR/kvd.log" | tail -2
  exit 1
fi
if (( AFTER > BEFORE )); then
  echo "OK: kvd took $((AFTER - BEFORE)) writes from the prefill engine"
else
  echo "FAIL: no kvd writes ($BEFORE -> $AFTER). Check that prefill logs"
  echo "      'hicache storage backend' and that KVD=1 reached launch_prefill.sh:"
  grep -aiE "hierarchical|hicache|kvd" "$LOG_DIR/prefill.log" | tail -10
  exit 1
fi
