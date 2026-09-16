#!/usr/bin/env bash
# Track 1 follow-up: warmup1 P:D ratio sweep with 2 and 3 prefill workers.
#
# The 137/138 sweep concluded that prefill is the sole limiting stage: the
# prefill in-run count sits pinned at the DP width while its queue grows without
# bound, ITL P90 stays flat while TTFT P90 grows ~8x, and the decode side burns
# 251 output tok/s against 258k input tok/s.  The direct test of that claim is
# to add prefill workers and leave everything else alone, so each point here is
# the P8D8 profile with 2 or 3 prefill instances against the same single D8.
#
# Same rules as the first sweep, because the points have to stay comparable:
# fresh launch per point (max-running and cuda-graph-max-bs must equal the
# concurrency, and a cold prefix cache stops points inheriting each other's
# cache state), mem_fraction_static=0.85, warmup1, DURATION=1200.
set -uo pipefail

BENCH=/home/liyingli/bench_agentx/Infera/bench/glm5p2_pd
CAMPAIGN="$BENCH/results/20260915_warmup1_pdratio"
PLAN="${PLAN_FILE:-$CAMPAIGN/points.tsv}"
INDEX="$CAMPAIGN/index.tsv"
EXPECT_IMAGE_ID=sha256:6ff85f4a43ae4b2773f59e827f230e23a657ae86ebf3d00a73b75bf94d5f9695
IMAGE_REF=infera/engine-sglang:glm52-v518-c29bd17-b02ab81
DURATION=1200

log() { printf '[pdratio %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
rsh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2"; }

[[ -f "$INDEX" ]] || printf 'phase\ttopology\tconc\tmem\tverdict\tper_gpu_tps\tnote\n' >"$INDEX"

# --- topology definitions.  Per-instance shape is config.p8d8.sh unchanged
# (prefill TP8/DP8/DPA, decode TP8/DP8/DPA); only the instance count moves, via
# the topology file.  The container prefix is per topology so a stale stack from
# the other topology can never be mistaken for this one's.
#
# BOOTSTRAP_PORT_BASE moves off the config's 18998 because the bases are only
# three apart: the Nth instance gets base+N, so a fourth instance asks for
# bootstrap 19001, which is engine-0's port.  validate_topology caught it and
# refused to launch, which is why the first attempt's 3P1D points died in two
# seconds.  18900 leaves room for sixteen instances.  It is set for every
# topology rather than just the four-instance ones so the campaign's points
# stay configured identically; a port number is not a performance parameter.
topo_args() {
  printf '%s\n' "BOOTSTRAP_PORT_BASE=18900"
  case "$1" in
    p8x2d8) printf '%s\n' "CONFIG=$BENCH/config.p8d8.sh" \
              "TOPOLOGY=$CAMPAIGN/topology.2p1d.tsv" \
              "CONTAINER_PREFIX=glm52-pd-p8x2-d8-v518" ;;
    p8x3d8) printf '%s\n' "CONFIG=$BENCH/config.p8d8.sh" \
              "TOPOLOGY=$CAMPAIGN/topology.3p1d.tsv" \
              "CONTAINER_PREFIX=glm52-pd-p8x3-d8-v518" ;;
    p8x2d8x2) printf '%s\n' "CONFIG=$BENCH/config.p8d8.sh" \
              "TOPOLOGY=$CAMPAIGN/topology.2p2d.tsv" \
              "CONTAINER_PREFIX=glm52-pd-p8x2-d8x2-v518" ;;
    *) echo "unknown topology: $1" >&2; return 1 ;;
  esac
}
topo_prefix() {
  case "$1" in
    p8x2d8) echo glm52-pd-p8x2-d8-v518 ;;
    p8x3d8) echo glm52-pd-p8x3-d8-v518 ;;
    p8x2d8x2) echo glm52-pd-p8x2-d8x2-v518 ;;
  esac
}
prefill_nodes() {
  case "$1" in
    p8x2d8) echo crsuse2-m2m-137 crsuse2-m2m-140 ;;
    p8x3d8) echo crsuse2-m2m-137 crsuse2-m2m-140 crsuse2-m2m-141 ;;
    p8x2d8x2) echo crsuse2-m2m-137 crsuse2-m2m-140 ;;
  esac
}
decode_nodes() {
  case "$1" in
    p8x2d8|p8x3d8) echo crsuse2-m2m-138 ;;
    p8x2d8x2) echo crsuse2-m2m-138 crsuse2-m2m-141 ;;
  esac
}

stop_stack() {  # $1 topology  $2 log
  local args; mapfile -t args < <(topo_args "$1")
  (cd "$BENCH" && ./stop.sh "${args[@]}" "RUN_LOG=$2" >/dev/null 2>&1 </dev/null) || true
}

collect_evidence() {  # $1 topology  $2 out
  local prefix node i=0
  prefix="$(topo_prefix "$1")"
  for node in $(prefill_nodes "$1"); do
    rsh "$node" "docker logs ${prefix}-prefill-$i" >"$2/logs/prefill-$i.log" 2>&1 || true
    i=$((i + 1))
  done
  i=0
  for node in $(decode_nodes "$1"); do
    rsh "$node" "docker logs ${prefix}-decode-$i" >"$2/logs/decode-$i.log" 2>&1 || true
    i=$((i + 1))
  done
  rsh "crsuse2-m2m-137" "docker logs ${prefix}-router" >"$2/logs/router.log" 2>&1 || true
}

any_log_matches() {  # $1 glob  $2 pattern
  local f
  for f in $1; do
    [[ -f "$f" ]] || continue
    grep -qE "$2" "$f" 2>/dev/null && return 0
  done
  return 1
}

oor_note() {  # classify the failure so the report does not have to guess
  local out="$1" n=""
  any_log_matches "$out/logs/prefill-*.log" "HSA_STATUS_ERROR_OUT_OF_RESOURCES|out of resources" && n="prefill-OOR"
  any_log_matches "$out/logs/decode-*.log" "HSA_STATUS_ERROR_OUT_OF_RESOURCES|out of resources" && n="${n:+$n,}decode-OOR"
  any_log_matches "$out/logs/decode-*.log" "PERMISSION_FAULT|memory access fault" && n="${n:+$n,}decode-gpu-fault"
  grep -qE "InvalidInferenceResultError" "$out/bench/runner.log" 2>/dev/null && n="${n:+$n,}empty-responses"
  echo "${n:-unclassified}"
}

run_point() {  # $1 phase  $2 topology  $3 conc  $4 mem  $5 extra
  local phase="$1" topo="$2" conc="$3" mem="$4" extra="$5"
  local out="$CAMPAIGN/$topo/c$conc"
  local -a extra_args=()
  if [[ -f "$out/VERDICT" ]]; then
    log "$phase $topo C$conc already has a verdict, skipping"
    return 0
  fi
  rm -rf "$out"; mkdir -p "$out"/{logs,launch,bench,evidence}

  local args; mapfile -t args < <(topo_args "$topo")
  args+=("PREFILL_MEM_FRACTION=$mem" "DECODE_MEM_FRACTION=$mem"
         "PREFILL_MAX_RUNNING=$conc" "PREFILL_GRAPH_MAX_BS=$conc"
         "DECODE_MAX_RUNNING=$conc" "DECODE_GRAPH_MAX_BS=$conc"
         "AGENTX_WARMUP_REQUESTS_PER_LANE=1")
  if [[ -n "$extra" && "$extra" != "-" ]]; then
    read -ra extra_args <<<"$extra"
    args+=("${extra_args[@]}")
  fi

  printf '%s\n' "${args[@]}" >"$out/args.txt"
  log "=== $phase $topo C$conc mem=$mem extra=$extra ==="

  # Refuse to start a ~1 hour point that the disk cannot hold.  The first
  # attempt spent 33 minutes on R2 and lost it to a full volume, which is worse
  # than not starting: the engines were occupied and the result was nothing.
  local free_gb
  free_gb=$(df -BG --output=avail "$CAMPAIGN" | tail -1 | tr -dc '0-9')
  if (( free_gb < 20 )); then
    log "$phase $topo C$conc ABORTED: only ${free_gb}G free on /home, need 20G"
    echo "ABORTED_DISK" >"$out/VERDICT"
    printf '%s\t%s\t%s\t%s\tABORTED_DISK\t\t%sG free\n' "$phase" "$topo" "$conc" "$mem" "$free_gb" >>"$INDEX"
    return 1
  fi

  # never launch on top of a live stack: the idle gate would fail anyway
  stop_stack "$topo" "$out/logs/pre-stop.log"

  if ! (cd "$BENCH" && ./launch.sh "${args[@]}" \
        "RUN_LOG=$out/logs/launch.log" "OUT_DIR=$out/launch" </dev/null); then
    log "$phase $topo C$conc LAUNCH_FAILED"
    collect_evidence "$topo" "$out"
    echo "LAUNCH_FAILED" >"$out/VERDICT"
    printf '%s\t%s\t%s\t%s\tLAUNCH_FAILED\t\t%s\n' "$phase" "$topo" "$conc" "$mem" "see logs/launch.log" >>"$INDEX"
    stop_stack "$topo" "$out/logs/stop.log"
    return 1
  fi

  # --- live gate: conc == max-running == graph-max-bs on EVERY worker, the
  # instance count is what the metric's GPU denominator assumes, and all workers
  # run the pinned image.  Fail closed.
  # A point that caps prefill below the concurrency has to tell the gate, so
  # the relaxed bound is checked rather than skipped.  Read back out of the
  # args that were actually passed, so the two cannot drift apart.
  local -a gate_conc=()
  local prefill_cap
  prefill_cap=$(printf '%s\n' "${args[@]}" | sed -n 's/^PREFILL_MAX_RUNNING=//p' | tail -1)
  [[ -n "$prefill_cap" && "$prefill_cap" != "$conc" ]] && gate_conc=(--prefill-conc "$prefill_cap")

  if ! python3 "$CAMPAIGN/gate.py" --launch-dir "$out/launch" --conc "$conc" \
        "${gate_conc[@]}" \
        --mem "$mem" --topology "$topo" --expect-image "$EXPECT_IMAGE_ID" \
        --output "$out/evidence/live-gate.json"; then
    log "$phase $topo C$conc LIVE_GATE_FAILED"
    echo "LIVE_GATE_FAILED" >"$out/VERDICT"
    printf '%s\t%s\t%s\t%s\tLIVE_GATE_FAILED\t\t%s\n' "$phase" "$topo" "$conc" "$mem" "see evidence/live-gate.json" >>"$INDEX"
    stop_stack "$topo" "$out/logs/stop.log"
    return 1
  fi

  local rc verdict per_gpu note
  (cd "$BENCH" && ./agentx_bench.sh "${args[@]}" "OUT_DIR=$out/bench" \
      "CONC=$conc" "DURATION=$DURATION" </dev/null)
  rc=$?
  collect_evidence "$topo" "$out"

  if [[ $rc -eq 0 && -f "$out/bench/agentx_conc$conc.json" ]]; then
    verdict=PASS
    per_gpu=$(python3 -c "
import json,sys
d=json.load(open('$out/bench/agentx_conc$conc.json'))
print(f\"{d['request_metrics']['throughput']['per_gpu']['total_tput_tps']:.2f}\")" 2>/dev/null || echo "")
    note=""
  else
    verdict=BENCH_FAILED
    per_gpu=""
    note=$(oor_note "$out")
  fi
  # A point's aiperf exports are ~1.5 GB, dominated by server_metrics_export.json,
  # and /home is a shared volume that ran out underneath the first attempt --
  # the R2 client died mid-warmup with no timeout and no error, and the verdict
  # could not be written.  Everything the report reads survives in
  # agentx_conc$conc.json and metrics_sample.log, so a passing point drops them.
  # A failing point keeps them, because that is when they are worth having.
  if [[ "$verdict" == PASS ]]; then
    rm -rf "$out/bench/aiperf_artifacts"
  fi

  echo "$verdict" >"$out/VERDICT"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$phase" "$topo" "$conc" "$mem" "$verdict" "$per_gpu" "$note" >>"$INDEX"
  log "$phase $topo C$conc $verdict per_gpu=$per_gpu $note"

  stop_stack "$topo" "$out/logs/stop.log"
  return 0
}

# Read the whole plan up front: launch.sh / agentx_bench.sh shell out to ssh,
# and ssh reads stdin; a loop fed straight from the plan file loses the
# remaining lines after the first point.
mapfile -t PLAN_LINES <"$PLAN"

# --- image gate once up front, across every node a point still to run will
# touch.  This campaign added two nodes that were carrying a DIFFERENT image
# under the same tag (a Sep 8 build, 62 layers vs 64), which is exactly the
# failure the first sweep was burned by.
#
# The set is derived from the plan rather than hard-coded, so a node that no
# remaining point uses cannot fail the gate: node 141's docker daemon went
# inactive mid-campaign, and a fixed four-node list refused to start the 2P1D
# points that do not go near it.
declare -A GATE_NODES=()
for plan_line in "${PLAN_LINES[@]}"; do
  IFS=$'\t' read -r phase topo conc mem extra <<<"$plan_line"
  [[ "$phase" =~ ^#|^$ ]] && continue
  [[ -z "${topo:-}" || -z "${conc:-}" ]] && continue
  [[ -f "$CAMPAIGN/$topo/c$conc/VERDICT" ]] && continue
  for n in $(prefill_nodes "$topo") $(decode_nodes "$topo"); do GATE_NODES["$n"]=1; done
done
if [[ ${#GATE_NODES[@]} -eq 0 ]]; then
  log "every planned point already has a verdict; nothing to do"
  exit 0
fi
for n in "${!GATE_NODES[@]}"; do
  got=$(rsh "$n" "docker image inspect --format '{{.Id}}' $IMAGE_REF" 2>/dev/null)
  [[ "$got" == "$EXPECT_IMAGE_ID" ]] || { log "FATAL: $n image is ${got:-MISSING}, expected $EXPECT_IMAGE_ID"; exit 1; }
done
log "image gate PASS on ${#GATE_NODES[@]} nodes: ${!GATE_NODES[*]}"

consecutive_fail=0
last_topo=""
for plan_line in "${PLAN_LINES[@]}"; do
  IFS=$'\t' read -r phase topo conc mem extra <<<"$plan_line"
  [[ "$phase" =~ ^#|^$ ]] && continue
  [[ -z "${topo:-}" || -z "${conc:-}" ]] && continue
  [[ "$topo" != "$last_topo" ]] && { consecutive_fail=0; last_topo="$topo"; }
  if [[ $consecutive_fail -ge 2 ]]; then
    log "skipping $phase $topo C$conc: two consecutive failures on $topo"
    printf '%s\t%s\t%s\t%s\tSKIPPED\t\ttwo prior consecutive failures\n' "$phase" "$topo" "$conc" "$mem" >>"$INDEX"
    continue
  fi
  if run_point "$phase" "$topo" "$conc" "$mem" "$extra"; then
    v=$(cat "$CAMPAIGN/$topo/c$conc/VERDICT" 2>/dev/null || echo UNKNOWN)
    [[ "$v" == PASS ]] && consecutive_fail=0 || consecutive_fail=$((consecutive_fail + 1))
  else
    consecutive_fail=$((consecutive_fail + 1))
  fi
done

log "sweep finished; index:"
cat "$INDEX"
