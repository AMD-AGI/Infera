#!/usr/bin/env bash
# Cut a torch-profiler window out of a running load. RUNS ON THE COMPUTE NODE.
# Adapted from examples/sglang_1p1d_glm5.2/engine/capture.sh, collapsed to the
# single `mixed` role a MIX deployment registers.
#
# Requires a load already in flight: an idle window profiles an empty scheduler
# loop, not the model. Start 06_aiperf_replay.sh in the background first.
#
# Called twice per profiled round, which is what WITH_STACK and OUT_SUBDIR are
# for. See the header of section 5/6 for why the two windows are separate rather
# than one window with stacks on.
set -u
MY_IP="${NODE_IP:?}"
ROUTER_PORT="${ROUTER_PORT:-8100}"
CTR="${CTR:-glm53_mix}"
TRACE_OUT="${TRACE_OUT:?}"
WARMUP_S="${WARMUP_S:-30}"
WINDOW_S="${WINDOW_S:-15}"
REQUIRE_LOAD="${REQUIRE_LOAD:-1}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
# Ask the profiler for the Python call stack behind every launch. Off for the
# measurement window, on for the short second window `kernel_scan` resolves
# launchers from.
WITH_STACK="${WITH_STACK:-0}"
# Which directory under the tag the traces land in, so two windows in one round
# do not overwrite each other.
OUT_SUBDIR="${OUT_SUBDIR:-mixed}"

case "$WITH_STACK" in
  0|1) ;;
  *) echo "  ABORT: WITH_STACK must be 0 or 1, got '$WITH_STACK'"; exit 1 ;;
esac

URL="http://$MY_IP:$ROUTER_PORT"
OUT="$TRACE_OUT/$TAG/$OUT_SUBDIR"

load_running(){ pgrep -f 'aiperf profile' >/dev/null 2>&1; }

echo "===== 1/6 preflight ====="
mounted=$(docker inspect -f "{{range .Mounts}}{{if eq .Destination \"$TRACE_OUT\"}}{{.RW}}{{end}}{{end}}" "$CTR")
[ "$mounted" = "true" ] || { echo "  ABORT: $TRACE_OUT not mounted rw in $CTR"; exit 1; }

code=$(docker exec "$CTR" curl -s -o /dev/null -w '%{http_code}' -m 10 \
  -X POST "$URL/v1/admin/profile/start?role=__probe__")
case "$code" in
  400) echo "  control plane ON (probe -> 400 invalid role, as expected)" ;;
  403) echo "  ABORT: router has profiling disabled; bring up with PROFILE=1"; exit 1 ;;
"") echo "  ABORT: no answer from the router at $URL"; exit 1 ;;
  *)   echo "  unexpected probe status '$code'; continuing" ;;
esac

n=$(docker exec "$CTR" curl -s -m10 "$URL/v1/workers" | tr -d ' \r' | grep -c '"disagg_mode":"mixed"')
[ "${n:-0}" -ge 1 ] || { echo "  ABORT: no mixed worker registered; start?role=mixed would 404"; exit 1; }
echo "  mixed workers registered: $n"

if [ "$REQUIRE_LOAD" = "1" ] && ! load_running; then
  echo "  ABORT: no aiperf load in flight. Start 06_aiperf_replay.sh first."
  exit 1
fi

echo "===== 2/6 wait for requests to actually reach the engine ====="
# Not a sleep: a cold AIPerf synthesizes every prompt before sending anything, so
# "the container is up" and "the engine is busy" can be minutes apart. Warm-up is
# only meaningful once the second is true.
seen=0
for i in $(seq 1 120); do
  if docker exec "$CTR" bash -c "grep -acE 'Decode batch|Prefill batch' /tmp/glm53_mix.log" | grep -qv '^0$'; then
    seen=1; echo "  engine has batches after $((i*5))s"; break
  fi
  load_running || { echo "  ABORT: the load exited before sending anything"; exit 1; }
  sleep 5
done
[ "$seen" = "1" ] || { echo "  ABORT: engine never reported a batch"; exit 1; }

echo "===== 3/6 warm-up ${WARMUP_S}s ====="
# Skipped rather than slept when zero. The second window of a round opens on an
# engine the first one already warmed, and re-serving the warm-up would spend
# the replay's remaining trace time on nothing.
if [ "$WARMUP_S" -gt 0 ]; then
  sleep "$WARMUP_S"
  load_running || echo "  WARN: the load finished during warm-up; the window may catch an idle engine"
else
  echo "  skipped (WARMUP_S=0; the engine is already under load)"
fi

echo "===== 4/6 output directory ====="
# SGLang does not create output_dir. When it is missing the export fails inside
# the profiler callback long after /start_profile answered 200 -- you find out at
# the end, with an empty directory and no error anywhere.
#
# Created from the HOST, not with `docker exec mkdir`. The engine container runs
# as root, so a directory it creates on the bind mount is root-owned and this
# user cannot later create $OUT/megapie inside it -- which is exactly where
# Magpie writes. Root can still write its trace files into a directory we own,
# so making it here costs nothing and keeps the analysis step unprivileged.
mkdir -p "$OUT" || { echo "  ABORT: cannot create $OUT"; exit 1; }
docker exec "$CTR" test -d "$OUT" || { echo "  ABORT: $OUT not visible in $CTR"; exit 1; }

echo "===== 5/6 start (window ${WINDOW_S}s, with_stack=${WITH_STACK}) ====="
# `with_stack` MUST be explicit either way: SGLang defaults it to True, and a
# measurement window taken with stacks on is one nobody can afford.
#
# **Measured on smci355-ccs-aus-n04-29, 2026-09-01** (`temp/manual/FINDINGS.md`),
# same workload profiled twice in the engine image: 2,996,700 bytes of trace
# against 228,553, so **13.1x uncompressed and 16.5x gzipped**, from 9,565
# `python_function` events against none. The kernel count and the total kernel
# time were identical across the pair, so stacks do not distort the measurement
# -- they only cost bytes. Applied to this package's measured 60.5 MB per rank
# for a 15 s window, stacks on would be about 1 GB per rank and 8 GB for the
# round.
#
# So the round takes two windows: this one long and without stacks for the
# ranking, and a short one with stacks that only has to hold a few launches per
# kernel. `assets/analyze/launchers.py` votes over three probes per kernel and
# reads two rank files, so seconds of stack trace answer what 8 GB would.
#
# record_shapes=true is what gives Magpie its Input Shapes column, in both
# windows. activities are spelled out so the engine does not choose.
if [ "$WITH_STACK" = "1" ]; then STACK_JSON=true; else STACK_JSON=false; fi
BODY=$(printf '{"output_dir":"%s","record_shapes":true,"with_stack":%s,"activities":["CPU","GPU"]}' \
  "$OUT" "$STACK_JSON")
docker exec "$CTR" curl -sS -m 60 -X POST -H 'Content-Type: application/json' \
  -d "$BODY" "$URL/v1/admin/profile/start?role=mixed" || { echo "  ABORT: profile start failed"; exit 1; }
echo
WIN_START=$(date -Is)
sleep "$WINDOW_S"

echo "===== 6/6 stop + flush ====="
# The router's HTTP client has a 30 s read timeout; a stop that flushes eight
# ranks can exceed it while the engine keeps writing. A failure here is not
# evidence the stop failed -- the byte-count check below is what decides.
docker exec "$CTR" curl -sS -m 180 -X POST -H 'Content-Type: application/json' \
  -d '{}' "$URL/v1/admin/profile/stop?role=mixed" \
  || echo "  stop returned an error (often just the 30 s read timeout); checking files"
echo
WIN_STOP=$(date -Is)

# A trace file appearing is not a trace file being finished: torch writes it from
# the profiler callback after stop has already returned. Wait for the byte count
# to stop moving rather than for a fixed sleep.
prev=-1
for _ in $(seq 1 "${FLUSH_TRIES:-30}"); do
  cur=$(du -sb "$OUT" 2>/dev/null | cut -f1); cur="${cur:-0}"
  if [ "$cur" -gt 0 ] && [ "$cur" = "$prev" ]; then
    echo "  stable at $cur bytes"; break
  fi
  prev="$cur"; sleep 10
done

echo "--- trace files ---"
ls -l "$OUT" 2>/dev/null
ranks=$(ls "$OUT"/*.trace.json.gz 2>/dev/null | wc -l)
echo "window: $WIN_START .. $WIN_STOP"
echo "ranks: $ranks"
echo "with_stack: $WITH_STACK"
[ "$ranks" -gt 0 ] && echo "CAPTURE_OK $OUT" || { echo "CAPTURE_FAIL: no trace files"; exit 1; }
