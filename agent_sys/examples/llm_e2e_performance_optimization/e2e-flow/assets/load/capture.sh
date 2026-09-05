#!/usr/bin/env bash
# Cut a torch-profiler window out of a running load. RUNS ON THE COMPUTE NODE.
# Adapted from examples/sglang_1p1d_glm5.2/engine/capture.sh, collapsed to the
# single `mixed` role a MIX deployment registers.
#
# Requires a load already in flight: an idle window profiles an empty scheduler
# loop, not the model. Start 06_aiperf_replay.sh in the background first.
#
# Called twice per profiler-attached round, which is what WITH_STACK and OUT_SUBDIR are
# for. See the header of section 5/6 for why the two windows are separate rather
# than one window with stacks on.
set -u
MY_IP="${NODE_IP:?}"
ROUTER_PORT="${ROUTER_PORT:-8100}"
# No default: a container name is bound on a shared host whose `docker ps`
# shows every tenant's containers, so it is passed, never guessed.
CTR="${CTR:?CTR=the engine container this capture drives}"
#: Where the traces land **on the host**. mkdir, du and ls use this one.
TRACE_OUT="${TRACE_OUT:?}"
#: The same directory **as the engine container sees it**, which is not the same
#: string. This file used to assume it was — `docker exec test -d "$OUT"` was
#: literally that assumption written down — and it held only because the
#: bring-up it was written against mounted the trace directory at the same path
#: inside. m1's `deploy_kit` mounts its work root at `/workdir`, declares where
#: in `deployment.json`'s `work_root_in_container`, and is right to: mandating
#: "same path inside" would overturn a working convention for every kit.
#:
#: Defaults to `$TRACE_OUT`, so the same-path convention keeps working unchanged
#: and `profiling-demo`'s behaviour is byte-identical.
#:
#: Getting this wrong is not a crash: SGLang writes to the path the **engine**
#: sees, so a container-side path that is not the mount lands in the container
#: layer, `/start_profile` still answers 200, and the host sees an empty
#: directory at the end with no error anywhere.
TRACE_OUT_IN_CONTAINER="${TRACE_OUT_IN_CONTAINER:-$TRACE_OUT}"
WARMUP_S="${WARMUP_S:-30}"
WINDOW_S="${WINDOW_S:-15}"
REQUIRE_LOAD="${REQUIRE_LOAD:-1}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
# Ask the profiler for the Python call stack behind every launch. Off for the
# measurement window, on for the short second window the ranking step resolves
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
#: The same directory named twice, once per side of the mount. Every use below
#: is one or the other and never both, and which it is is stated at the use.
OUT="$TRACE_OUT/$TAG/$OUT_SUBDIR"
OUT_IN_CONTAINER="$TRACE_OUT_IN_CONTAINER/$TAG/$OUT_SUBDIR"

load_running(){ pgrep -f 'aiperf profile' >/dev/null 2>&1; }

echo "===== 1/6 preflight ====="
# `.Destination` is the **container** side of the mount, so it is compared
# against the container path. Comparing it against the host path is the failure
# this pair of variables exists to prevent.
#
# **The mount does not have to sit exactly on the trace directory.** An earlier
# version demanded `.Destination == $TRACE_OUT_IN_CONTAINER`, and that refused
# every real bring-up: m1's kit mounts one rw parent
# (`/mnt/m2m_nobackup/yihou`) and the traces land several levels below it. The
# directory was writable and the capture aborted anyway — measured on two nodes,
# runs p4_b (275) and p4_a (088), 2026-09-05.
#
# So: find the mount that actually governs this path — the **longest**
# destination that is the path or an ancestor of it — and require *that* one to
# be rw. Longest wins because a read-only mount nested under an rw parent still
# governs what is under it, and taking any ancestor would read that as writable.
# No ancestor at all is still an abort: the path is then in the container's
# writable layer, which is the silent failure the header of this file warns
# about — SGLang writes, `/start_profile` answers 200, and the host sees an
# empty directory.
governing_dest=''
governing_rw=''
while IFS='	' read -r dest rw; do
  [ -n "$dest" ] || continue
  case "$TRACE_OUT_IN_CONTAINER" in
    "$dest"|"$dest"/*) ;;
    *) continue ;;
  esac
  [ "${#dest}" -gt "${#governing_dest}" ] || continue
  governing_dest="$dest"
  governing_rw="$rw"
done <<EOF
$(docker inspect -f '{{range .Mounts}}{{.Destination}}	{{.RW}}{{"\n"}}{{end}}' "$CTR")
EOF
[ "$governing_rw" = "true" ] || {
  if [ -n "$governing_dest" ]; then
    echo "  ABORT: $TRACE_OUT_IN_CONTAINER is governed by the read-only mount $governing_dest in $CTR"
  else
    echo "  ABORT: $TRACE_OUT_IN_CONTAINER is not under any mount in $CTR — writes would land in the container layer and the host would see nothing"
  fi
  echo "  (host side: $TRACE_OUT). Mount destinations $CTR actually has:"
  docker inspect -f '{{range .Mounts}}    {{.Destination}} rw={{.RW}}{{"\n"}}{{end}}' "$CTR"
  exit 1
}
echo "  traces are governed by mount $governing_dest (rw) in $CTR"

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
# Created on the host, then checked **through the mount** at the container path.
# This is the one place the two names have to agree about the same bytes, so it
# is also the check that catches a wrong `TRACE_OUT_IN_CONTAINER` — before the
# window opens, rather than as an empty directory at the end.
docker exec "$CTR" test -d "$OUT_IN_CONTAINER" || {
  echo "  ABORT: $OUT (host) is not visible at $OUT_IN_CONTAINER inside $CTR"
  echo "  The two are the same directory through the bind mount; if they are not,"
  echo "  SGLang will write into the container layer and the host will see nothing."
  exit 1
}

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
# `output_dir` is read by the **engine**, so it is the container path.
BODY=$(printf '{"output_dir":"%s","record_shapes":true,"with_stack":%s,"activities":["CPU","GPU"]}' \
  "$OUT_IN_CONTAINER" "$STACK_JSON")
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
