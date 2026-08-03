#!/usr/bin/env bash
# Launch the infera-kvd daemon on the prefill node, before the prefill engine.
# Run inside the engine container:
#   bash launch/launch_kvd.sh
#
# The engine's --infera-kvd-socket probe refuses to start if this daemon is not
# answering, so the order matters: kvd, then prefill.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/env.sh"

if [[ "$KVD" != "1" ]]; then
  echo "[kvd] KVD=0 — no daemon, plain PD deployment"
  exit 0
fi

LOG="${LOG:-$LOG_DIR/kvd.log}"

mkdir -p "$(dirname "$KVD_SOCKET")" "$KVD_L3_DIR" \
  || { echo "[kvd] cannot create $KVD_SOCKET dir or $KVD_L3_DIR" >&2; exit 1; }

pkill -f "infera\.kvd .*--socket ${KVD_SOCKET}( |$)" 2>/dev/null || true
# A stale socket file from a killed daemon still passes `test -S`, and the
# engine's probe then fails with Connection refused instead of waiting.
rm -f "$KVD_SOCKET"
sleep 1

# --use-tablespace is load-bearing: without it --long-path builds the legacy
# file-per-block region instead of the container-file tablespace (bounded file
# count, O_DIRECT, journal-recovered index).
nohup python3 -m infera.kvd \
  --socket "$KVD_SOCKET" \
  --max-bytes "$KVD_RAM_BYTES" \
  --long-path "$KVD_L3_DIR" --long-bytes "$KVD_LONG_BYTES" \
  --io-mode "$KVD_IO_MODE" \
  --use-tablespace --tablespace-pools "$KVD_TABLESPACE_POOLS" \
  > "$LOG" 2>&1 &

# Wait until the daemon actually answers an RPC, not merely until the socket
# file appears — that is what the engine's startup probe does, so anything
# weaker here just moves the failure into engine startup.
for _ in $(seq 60); do
  python3 - "$KVD_SOCKET" <<'PY' 2>/dev/null && break
import asyncio, sys
from infera.kvd.client import KvdClient
async def main():
    c = KvdClient(sys.argv[1], client_id="launch-probe")
    await asyncio.wait_for(c.connect(), timeout=3)
    await asyncio.wait_for(c.stats(), timeout=3)
    await c.close()
asyncio.run(main())
PY
  sleep 1
done
python3 -m infera.kvd.statctl --socket "$KVD_SOCKET" >/dev/null 2>&1 \
  || { echo "[kvd] daemon not answering on $KVD_SOCKET"; tail -40 "$LOG"; exit 1; }

echo "[kvd] up: socket=$KVD_SOCKET ram=$KVD_RAM_BYTES l3=$KVD_L3_DIR ($KVD_LONG_BYTES); log=$LOG"
# The classifier decides O_DIRECT vs buffered for L3. A shared mount landing on
# buffered is the difference between reading L3 under the TTFT budget and inside
# it, so print the verdict rather than leaving it in the log.
grep -aE "L3 io_mode|io_mode:|selfcheck|storage_selfcheck" "$LOG" | head -5 || true
