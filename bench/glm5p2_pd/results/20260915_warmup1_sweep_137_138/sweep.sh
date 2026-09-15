#!/usr/bin/env bash
# Track 1: warmup1 concurrency sweep on 137/138 for P4D8 and P8D8.
#
# One point == fresh launch -> live-config gate -> bench -> artifact capture ->
# stop.  Fresh launch per point is mandatory because max-running and
# cuda-graph-max-bs must equal the concurrency, and it also guarantees a cold
# prefix cache so points do not inherit each other's cache state.
#
# Config family is plain config defaults + warmup1, which is what the existing
# P8D8 pinned warmup1 C32/C48/C64 used (mem_fraction_static=0.85, prefill
# HSA_NO_SCRATCH_RECLAIM=0, no active GC, PD_DP_RANK_AFFINITY=1).  Deviating
# from it would make the new points incomparable with those three.
set -uo pipefail

BENCH=/home/liyingli/bench_agentx/Infera/bench/glm5p2_pd
CAMPAIGN="$BENCH/results/20260915_warmup1_sweep_137_138"
PLAN="$CAMPAIGN/points.tsv"
INDEX="$CAMPAIGN/index.tsv"
EXPECT_IMAGE_ID=sha256:6ff85f4a43ae4b2773f59e827f230e23a657ae86ebf3d00a73b75bf94d5f9695
PREFILL_NODE=crsuse2-m2m-137
DECODE_NODE=crsuse2-m2m-138
DURATION=1200

log() { printf '[sweep %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
rsh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2"; }

[[ -f "$INDEX" ]] || printf 'phase\ttopology\tconc\tmem\tverdict\tper_gpu_tps\tnote\n' >"$INDEX"

# --- topology definitions.  P8D8 has its own config file; P4D8 is config.sh's
# default topology, spelled out here so the launch record is explicit.
topo_args() {
  case "$1" in
    p8d8) printf '%s\n' "CONFIG=$BENCH/config.p8d8.sh" ;;
    p4d8) printf '%s\n' \
            "CONFIG=$BENCH/config.sh" \
            "PREFILL_GPU_DEVICES=0,1,2,3" "PREFILL_TP=4" "PREFILL_EP=1" \
            "PREFILL_DP=4" "PREFILL_DPA=1" \
            "DECODE_GPU_DEVICES=0,1,2,3,4,5,6,7" "DECODE_TP=8" "DECODE_EP=1" \
            "DECODE_DP=8" "DECODE_DPA=1" \
            "CONTAINER_PREFIX=glm52-pd-p4dpa-d8dpa-v518" ;;
    *) echo "unknown topology: $1" >&2; return 1 ;;
  esac
}
topo_prefix() {
  case "$1" in
    p8d8) echo glm52-pd-p8dpa-d8dpa-v518 ;;
    p4d8) echo glm52-pd-p4dpa-d8dpa-v518 ;;
  esac
}
expect_gpus() { case "$1" in p8d8) echo 8 8 ;; p4d8) echo 4 8 ;; esac; }

stop_stack() {  # $1 topology
  local args; mapfile -t args < <(topo_args "$1")
  (cd "$BENCH" && ./stop.sh "${args[@]}" "RUN_LOG=$2" >/dev/null 2>&1 </dev/null) || true
}

run_point() {  # $1 phase  $2 topology  $3 conc  $4 mem  $5 extra
  local phase="$1" topo="$2" conc="$3" mem="$4" extra="$5"
  local out="$CAMPAIGN/$topo/c$conc"
  local -a args=() extra_args=()
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

  # --- live gate: the whole point of the sweep is that conc == max-running ==
  # graph-max-bs and that the topology is what we claim.  Fail closed.
  if ! python3 "$CAMPAIGN/gate.py" --launch-dir "$out/launch" --conc "$conc" \
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
  echo "$verdict" >"$out/VERDICT"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$phase" "$topo" "$conc" "$mem" "$verdict" "$per_gpu" "$note" >>"$INDEX"
  log "$phase $topo C$conc $verdict per_gpu=$per_gpu $note"

  stop_stack "$topo" "$out/logs/stop.log"
  return 0
}

collect_evidence() {  # $1 topology  $2 out
  local prefix; prefix="$(topo_prefix "$1")"
  rsh "$PREFILL_NODE" "docker logs ${prefix}-prefill-0" >"$2/logs/prefill.log" 2>&1 || true
  rsh "$DECODE_NODE"  "docker logs ${prefix}-decode-0"  >"$2/logs/decode.log" 2>&1 || true
  rsh "$PREFILL_NODE" "docker logs ${prefix}-router"    >"$2/logs/router.log" 2>&1 || true
}

oor_note() {  # classify the failure so the report does not have to guess
  local out="$1" n=""
  grep -qE "HSA_STATUS_ERROR_OUT_OF_RESOURCES|out of resources" "$out/logs/prefill.log" 2>/dev/null && n="prefill-OOR"
  grep -qE "HSA_STATUS_ERROR_OUT_OF_RESOURCES|out of resources" "$out/logs/decode.log" 2>/dev/null && n="${n:+$n,}decode-OOR"
  grep -qE "PERMISSION_FAULT|memory access fault" "$out/logs/decode.log" 2>/dev/null && n="${n:+$n,}decode-gpu-fault"
  grep -qE "InvalidInferenceResultError" "$out/bench/runner.log" 2>/dev/null && n="${n:+$n,}empty-responses"
  echo "${n:-unclassified}"
}

# --- image gate once up front: this campaign has already been burned by a tag
# that silently pointed at different content.
for n in "$PREFILL_NODE" "$DECODE_NODE"; do
  got=$(rsh "$n" "docker image inspect --format '{{.Id}}' infera/engine-sglang:glm52-v518-c29bd17-b02ab81")
  [[ "$got" == "$EXPECT_IMAGE_ID" ]] || { log "FATAL: $n image is $got, expected $EXPECT_IMAGE_ID"; exit 1; }
done
log "image gate PASS on both nodes"

consecutive_fail=0
last_topo=""
# Read the whole plan up front.  launch.sh / agentx_bench.sh shell out to ssh,
# and ssh reads stdin; with the loop fed straight from the plan file it ate the
# remaining lines and the sweep stopped after one point.
mapfile -t PLAN_LINES <"$PLAN"
for plan_line in "${PLAN_LINES[@]}"; do
  IFS=$'\t' read -r phase topo conc mem extra <<<"$plan_line"
  [[ "$phase" =~ ^#|^$ ]] && continue
  [[ -z "${topo:-}" || -z "${conc:-}" ]] && continue
  [[ "$topo" != "$last_topo" ]] && { consecutive_fail=0; last_topo="$topo"; }
  if [[ $consecutive_fail -ge 2 ]]; then
    log "skipping $phase $topo C$conc: two consecutive failures on $topo, higher concurrency cannot help"
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
