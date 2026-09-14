#!/bin/bash
# Does a SECOND torch.profiler window wedge the engine, and is it `with_stack`
# or the second session itself?
#
# WHY: 2026-09-06, run 20260906T180412-6b7c19. Window 1 (with_stack=0) stopped
# with 200 and flushed 155 MB. Window 2 (with_stack=1, 3 s) started with 200 and
# its /stop_profile never returned; the engine's last log line is the second the
# stop was issued, and it never served again. TWO variables changed between the
# two windows and this separates them.
#
# It needs an engine already serving. It brings nothing up and tears nothing
# down. It issues /start_profile and /stop_profile and one trivial load.
#
#   ARM=repro   window 1 with_stack=0, window 2 with_stack=1   (does it reproduce)
#   ARM=nostack window 1 with_stack=0, window 2 with_stack=0   (is it the 2nd session)
#
# Run the `repro` arm FIRST. If it does not reproduce, `nostack` answers nothing.
#
#   WORKER=http://<host>:<worker-port>  bash probe_two_windows.sh
set -uo pipefail

WORKER="${WORKER:-}"
ARM="${ARM-repro}"
WIN1="${WIN1:-10}"
WIN2="${WIN2:-3}"

[ -n "$WORKER" ] || { echo "ABORT: set WORKER=http://host:port (the engine worker, not the router)"; exit 1; }
case "$ARM" in repro) S2=true ;; nostack) S2=false ;; *) echo "ABORT: ARM must be repro|nostack"; exit 1 ;; esac

NOW=$(date -u +%Y%m%dT%H%M%SZ)
OUT="/data/yihou/e2e_verify_20260906/m2/twowindow_${ARM}_${NOW}"
case "$OUT" in *yihou*) : ;; *) echo "ABORT: out path lacks yihou"; exit 1 ;; esac
mkdir -p "$OUT/w1" "$OUT/w2" || exit 1
echo "out: $OUT   arm=$ARM   window2 with_stack=$S2"

ts() { date -u +%H:%M:%S.%3NZ; }
note() { echo "[$(ts)] $*" | tee -a "$OUT/probe.log"; }

# A generate probe is the only liveness test that means anything here: the
# router and /health answer without the engine generating a single token.
alive() {
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' -m "${1:-20}" -X POST "$WORKER/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen3-32B","messages":[{"role":"user","content":"hi"}],"max_tokens":4}')
  echo "$code"
}

note "baseline generate -> $(alive 30)"

# a trivial background load so neither window opens on an idle engine
( for _ in $(seq 1 400); do
    curl -s -o /dev/null -m 60 -X POST "$WORKER/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"Qwen/Qwen3-32B","messages":[{"role":"user","content":"count to twenty"}],"max_tokens":64}'
  done ) >/dev/null 2>&1 &
LOAD=$!
note "load pid $LOAD"
sleep 5

window() {  # $1=dir $2=seconds $3=with_stack(true|false)
  local d="$1" s="$2" ws="$3"
  note "WINDOW $d: start with_stack=$ws"
  curl -s -m 60 -X POST "$WORKER/start_profile" -H 'Content-Type: application/json' \
    -d "{\"output_dir\":\"$d\",\"record_shapes\":true,\"with_stack\":$ws,\"activities\":[\"CPU\",\"GPU\"]}" \
    | tee -a "$OUT/probe.log"; echo
  sleep "$s"
  note "WINDOW $d: stop (120 s cap)"
  curl -s -m 120 -w '\nSTOP_HTTP=%{http_code} t=%{time_total}s\n' -X POST "$WORKER/stop_profile" \
    | tee -a "$OUT/probe.log"
  note "WINDOW $d: files=$(ls -1 "$d" 2>/dev/null | wc -l)"
  note "WINDOW $d: generate after stop -> $(alive 30)"
}

window "$OUT/w1" "$WIN1" false
window "$OUT/w2" "$WIN2" "$S2"

note "post-probe generate (60 s cap) -> $(alive 60)"
kill "$LOAD" 2>/dev/null
rocm-smi --showuse --showmemuse > "$OUT/rocm-smi.after.txt" 2>&1
{ echo "captured_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "arm=$ARM worker=$WORKER win1=${WIN1}s/stack=false win2=${WIN2}s/stack=$S2"; } > "$OUT/GENERATED.txt"

echo
echo "=== how to read it ==="
echo "  w2 stop 200 + files + generate 200  -> did NOT reproduce; this arm answers nothing further"
echo "  w2 stop times out + generate fails  -> reproduced. TAKE THE STACK NOW:"
echo "       bash /data/yihou/e2e_verify_20260906/m2/capture_engine_stack.sh"
echo "     and take it BEFORE stopping the container - it is unrecoverable after."
