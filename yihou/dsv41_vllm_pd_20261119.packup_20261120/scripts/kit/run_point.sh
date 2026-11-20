#!/usr/bin/env bash
# Run ONE scored AgentX point (InferenceX contract) against an already-running server. Run ON the GPU node
# (e.g. `ssh <node> setsid nohup bash <ws>/scripts/kit/run_point.sh ... >/dev/null 2>&1 &`).
#
# Usage: run_point.sh <tag> <server-url> <concurrency> <gpus-for-denominator> [official|fast]
#   tag        the server's tag (serve.sh); results go to $DSV41_NFS_ROOT/runs/<tag>/bench/ (must not hold a previous run)
#   server-url client endpoint (router if any), e.g. http://127.0.0.1:8000
#   concurrency AgentX lanes (one concurrency per fresh server: CONC_LIST is not set)
#   gpus       GPU denominator = harness env TP. For DP layouts and PD this must be the TOTAL GPUs of the deployment
#              (InferenceX computes num_gpus = TP*PP*PCP from env; DP is not counted) -- the "TP = total GPUs" gotcha.
#   mode       official (default): DURATION=3600, preset warmup 10 req/lane, seed 42 (preset), failed-request threshold 0.10
#              fast: AIPERF_EXPERIMENTAL_FAST=1 -> 1200 s, 1 warmup req/lane (screening only, never for comparisons)
#
# What runs: python3 -m infx.bench agentic (InferenceX inferencex-e2e @ source/InferenceX, the CI entry point used by
# benchmarks/srt_agentic.sh) with the prebuilt venv (agentx-harness==1.0.14) at $DSV41_NFS_ROOT/venv-agentx/venv. Using the
# venv's own python makes infx skip its per-job venv bootstrap (Runtime.active()); the venv is mounted read-only so the
# bootstrap's rmtree could never touch it. Dataset/tokenizer come from our HF cache $DSV41_NFS_ROOT/hf_home with
# HF_HUB_OFFLINE=1: refs/main of the dataset is pinned to 23f152f6 (verified sha256 of traces.jsonl) and the tokenizer
# files are deepseek-ai/DeepSeek-V4.1-Flash@314113f8 (identical sha256 to /perf_apps/chaox/models/DeepSeek-V4.1-Flash).
# The client runs in a docker container (yihou-dsv41-<tag>-client, --rm, no GPU devices) as the calling uid.
#
# Env overrides (all optional):
#   DSV41_NFS_ROOT (/perf_apps/yihou/dsv41_vllm_pd_20261119)  DSV41_LOCAL_ROOT (/data/yihou/dsv41_vllm_pd_20261119)
#   KIT_IX            InferenceX inferencex-e2e dir (default <ws>/source/InferenceX/inferencex-e2e, mounted read-only)
#   KIT_CLIENT_IMAGE  image to host the client (default: IMAGE of runs/<tag>/kit/config.env, i.e. the server image already on the node)
#   KIT_METRICS_URLS  comma list of Prometheus URLs (default: METRICS_URLS of runs/<tag>/kit/config.env (PD: P and D engines,
#                     written by serve_pd.sh), else engine http://<url host>:<PORT>/metrics, else <server-url>/metrics).
#                     Must expose vllm:* (AIPERF_REQUIRED_SERVER_METRIC_PREFIX='vllm:').
#   KIT_SERVER_LOG    server log copied to bench/server.log before the run (InferenceX parses "GPU KV cache size" from it);
#                     default SERVER_LOG_PATH of the config (PD: the DECODE log), else runs/<tag>/server.log. PD also copies
#                     PREFILL_SERVER_LOG_PATH to bench/server_prefill.log.
#   KIT_HEALTH_WAIT_S wait up to N s for <server-url>/health == 200 (default 600)
#   KIT_ALLOW_USED_SERVER=1  allow a server whose vllm:request_success counter is already > 0 (official refuses by default)
#   KIT_STOP_BY       ISO-8601 UTC time the node must be released by; refuse if the point cannot finish before it
#   KIT_DP_ATTENTION / KIT_KV_OFFLOADING (none|dram) / KIT_KV_BACKEND / KIT_KV_BACKEND_VERSION / KIT_CPU_DRAM_GB /
#   KIT_ROUTER_NAME / KIT_ROUTER_VERSION / KIT_DISAGG / KIT_EP  result metadata (defaults derived from runs/<tag>/kit/config.env;
#                     DISAGG from the config's DISAGG (PD: true). IS_MULTINODE stays false even for PD: InferenceX then takes
#                     num_gpus = TP env = gpus arg (P+D GPUs) and keeps the single-node result layout summarize.py reads.)
#   KIT_IMAGE_LABEL   server image recorded in the result (default: config IMAGE)
#   KIT_EXTRA_ENV     "K=V K2=V2" extra env for the client (e.g. AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true);
#                     whether CI 448242 set any session-affinity header env is NOT verified.
#   KIT_NO_SUMMARY=1  skip summarize.py at the end
#   KIT_DRY_RUN=1     no docker/curl: render manifest + client docker argv into bench/ and print them, then exit
set -uo pipefail

KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WS=$(cd "$KIT_DIR/../.." && pwd)
PROJ=dsv41_vllm_pd_20261119
NFS_ROOT=${DSV41_NFS_ROOT:-/perf_apps/yihou/$PROJ}
LOCAL_ROOT=${DSV41_LOCAL_ROOT:-/data/yihou/$PROJ}
# docker daemon (root) cannot read HOME NFS on the nodes (root_squash) -> mount only staged copies on /perf_apps
STAGE=$NFS_ROOT/stage
IX=${KIT_IX:-$STAGE/inferencex-e2e}
VENV_ROOT=${KIT_VENV_ROOT:-$NFS_ROOT/venv-agentx}
# The uv venv holds absolute symlinks to its build path; on another cluster mount it at that path (KIT_VENV_CTR).
VENV_CTR=${KIT_VENV_CTR:-$VENV_ROOT}
VENV_PY=$VENV_CTR/venv/bin/python
HF_HOME_DIR=${KIT_HF_HOME:-$NFS_ROOT/hf_home}
MODEL_ID=deepseek-ai/DeepSeek-V4.1-Flash
DATASET_REV=23f152f6f0f9399a85901b89a6458def0ef16729

die() { echo "run_point.sh: ERROR: $*" >&2; exit 2; }
[ $# -ge 4 ] || die "usage: run_point.sh <tag> <server-url> <concurrency> <gpus-for-denominator> [official|fast]"
TAG=$1; URL=${2%/}; CONC=$3; GPUS=$4; MODE=${5:-official}

# ---- guards -------------------------------------------------------------------------------------------------------
[[ "$TAG" =~ ^[a-z0-9][a-z0-9._-]{0,55}$ ]] || die "unsafe tag '$TAG'"
[[ "$URL" =~ ^http://[A-Za-z0-9._-]+:[0-9]+$ ]] || die "server-url must look like http://host:port (got '$URL')"
[[ "$CONC" =~ ^[1-9][0-9]{0,4}$ ]] || die "bad concurrency '$CONC'"
[[ "$GPUS" =~ ^[1-9][0-9]{0,2}$ ]] || die "bad gpus '$GPUS'"
case "$MODE" in official) FAST=0; DURATION=3600 ;; fast) FAST=1; DURATION=1200 ;; *) die "mode must be official|fast" ;; esac
for p in "$NFS_ROOT" "$LOCAL_ROOT"; do case "$p" in */yihou/*) ;; *) die "path '$p' does not contain /yihou/" ;; esac; done
[ -f "$VENV_ROOT/venv/pyvenv.cfg" ] || die "venv missing: $VENV_ROOT/venv (build per scripts/kit/client_selftest.sh header)"
[ -f "$IX/infx/bench/agentic/run.py" ] || die "InferenceX inferencex-e2e not found at $IX"
[ -f "$HF_HOME_DIR/hub/datasets--semianalysisai--cc-traces-weka-062126/refs/main" ] || die "dataset cache missing in $HF_HOME_DIR"
[ "$(cat "$HF_HOME_DIR/hub/datasets--semianalysisai--cc-traces-weka-062126/refs/main")" = "$DATASET_REV" ] \
  || die "dataset refs/main in $HF_HOME_DIR is not $DATASET_REV"
DRY=${KIT_DRY_RUN:-0}
[ "$DRY" = 1 ] || command -v docker >/dev/null || die "docker not found"

RUN=$NFS_ROOT/runs/$TAG
B=$RUN/bench
LOC=$LOCAL_ROOT/client/$TAG
CNAME=yihou-dsv41-$TAG-client
[ -e "$B/aiperf_artifacts" ] && die "$B/aiperf_artifacts exists: fresh output dir per point (new tag + fresh server)"
mkdir -p "$B" "$LOC/tmp" || die "cannot create $B or $LOC"
# SECURITY (review 11-20): runs/<tag> may sit in a shared 1777 dir -> bench/ must be a real dir we own, not group/other-writable.
[ -d "$B" ] && [ ! -L "$B" ] && [ "$(stat -c %u "$B")" = "$(id -u)" ] || die "unsafe bench dir $B (symlink or not ours)"
chmod 0755 "$B" || die "cannot chmod $B"
[ ! -L "$B/run_point.log" ] || die "refusing symlinked $B/run_point.log"
L=$B/run_point.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }

# ---- config of the server (written by serve.sh) -----------------------------------------------------------------------
kit_cfg_ok() {  # kit_cfg_ok <file>: refuse to source a config not owned by us or writable by group/other (shared NFS)
  [ -f "$1" ] && [ "$(stat -c %u "$1")" = "$(id -u)" ] && [ "$(( 0$(stat -c %a "$1") & 022 ))" = 0 ]
}
cfgv() {  # cfgv <VAR> : scalar from runs/<tag>/kit/config.env, empty if absent
  [ -f "$RUN/kit/config.env" ] || return 0
  kit_cfg_ok "$RUN/kit/config.env" || die "unsafe $RUN/kit/config.env (owner/mode); refusing to source"
  bash -c 'source "$1" >/dev/null 2>&1; printf "%s" "${!2:-}"' _ "$RUN/kit/config.env" "$1"
}
C_IMAGE=$(cfgv IMAGE); C_TP=$(cfgv BENCH_TP); C_CONC=$(cfgv CONCURRENCY); C_PORT=$(cfgv PORT); C_DP=$(cfgv DP_SIZE)
C_UMBP=$(cfgv UMBP_ENABLE); C_DRAM=$(cfgv UMBP_HOST_DRAM_GB); C_ROUTER=$(cfgv ROUTER_ENABLE); C_RVER=$(cfgv ROUTER_VERSION)
C_ID=$(cfgv CFG_ID); C_MURLS=$(cfgv METRICS_URLS); C_SLOG=$(cfgv SERVER_LOG_PATH); C_PSLOG=$(cfgv PREFILL_SERVER_LOG_PATH)
C_DISAGG=$(cfgv DISAGG)
[ -n "$C_TP" ] && [ "$C_TP" != "$GPUS" ] && die "gpus=$GPUS but config $C_ID says BENCH_TP=$C_TP (GPU denominator mismatch)"
[ -n "$C_CONC" ] && [ "$C_CONC" != "$CONC" ] && say "WARN concurrency $CONC differs from config $C_ID CONCURRENCY=$C_CONC"
CLIENT_IMAGE=${KIT_CLIENT_IMAGE:-$C_IMAGE}
[ -n "$CLIENT_IMAGE" ] || die "no client image (set KIT_CLIENT_IMAGE or provide runs/$TAG/kit/config.env)"
[ "$DRY" = 1 ] || docker image inspect "$CLIENT_IMAGE" >/dev/null 2>&1 || die "client image not on this node: $CLIENT_IMAGE"
URL_HOST=$(echo "$URL" | sed -E 's#^http://([^:/]+):.*#\1#')
if [ -n "${KIT_METRICS_URLS:-}" ]; then METRICS=$KIT_METRICS_URLS
elif [ -n "$C_MURLS" ]; then METRICS=$C_MURLS
elif [ -n "$C_PORT" ]; then METRICS=http://$URL_HOST:$C_PORT/metrics
else METRICS=$URL/metrics; fi
SERVER_LOG=${KIT_SERVER_LOG:-${C_SLOG:-$RUN/server.log}}

DP_ATTN=${KIT_DP_ATTENTION:-$([ "${C_DP:-1}" -gt 1 ] 2>/dev/null && echo true || echo false)}
if [ -n "${KIT_KV_OFFLOADING:-}" ]; then KVO=$KIT_KV_OFFLOADING; elif [ "$C_UMBP" = 1 ]; then KVO=dram; else KVO=none; fi
KVB=""; KVB_META=""; DRAM_GB=${KIT_CPU_DRAM_GB:-0}
if [ "$KVO" = dram ]; then
  KVB=${KIT_KV_BACKEND:-umbp}
  if [ -n "${KIT_KV_BACKEND_VERSION:-}" ]; then KVB_META="{\"name\":\"$KVB\",\"version\":\"$KIT_KV_BACKEND_VERSION\"}"
  else KVB_META="{\"name\":\"$KVB\"}"; fi
  [ "$DRAM_GB" -gt 0 ] 2>/dev/null || DRAM_GB=${C_DRAM:-0}
  [ "$DRAM_GB" -gt 0 ] 2>/dev/null || die "KV_OFFLOADING=dram needs KIT_CPU_DRAM_GB > 0"
fi
ROUTER_META=""
R_NAME=${KIT_ROUTER_NAME:-$([ "$C_ROUTER" = 1 ] && echo vllm-router)}
R_VER=${KIT_ROUTER_VERSION:-$C_RVER}
[ -n "$R_NAME" ] && [ -n "$R_VER" ] && ROUTER_META="{\"name\":\"$R_NAME\",\"version\":\"$R_VER\"}"

# ---- lease check --------------------------------------------------------------------------------------------------
if [ -n "${KIT_STOP_BY:-}" ]; then
  need=$(( DURATION + 1800 + 900 ))   # profile + warmup budget (preset grace 1800 s) + dataset load/drain margin
  end=$(date -u -d "$KIT_STOP_BY" +%s 2>/dev/null) || die "bad KIT_STOP_BY"
  [ $(( $(date +%s) + need )) -le "$end" ] || die "lease too short: need ${need}s before $KIT_STOP_BY"
fi

# ---- server readiness + freshness ---------------------------------------------------------------------------------
IFS=, read -r -a MURLS <<< "$METRICS"
if [ "$DRY" != 1 ]; then
say "point start tag=$TAG url=$URL conc=$CONC gpus(TP env)=$GPUS mode=$MODE metrics=$METRICS image=$CLIENT_IMAGE"
t0=$(date +%s); WAIT=${KIT_HEALTH_WAIT_S:-600}
until [ "$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$URL/health")" = 200 ]; do
  [ $(( $(date +%s) - t0 )) -ge "$WAIT" ] && die "server $URL/health not 200 after ${WAIT}s"
  sleep 15
done
curl -s -m 20 "$URL/v1/models" > "$B/models.json" || true
grep -q "\"$MODEL_ID\"" "$B/models.json" || say "WARN $URL/v1/models does not list $MODEL_ID (served-model-name mismatch?)"
i=0; used=0
for m in "${MURLS[@]}"; do
  if curl -s -m 20 -f "$m" > "$B/metrics_before_$i.txt"; then
    n=$(awk '/^vllm:request_success(_total)?[{ ]/{s+=$NF} END{printf "%d", s+0}' "$B/metrics_before_$i.txt")
    [ "$n" -gt 0 ] && used=$((used + n))
    grep -q '^vllm:' "$B/metrics_before_$i.txt" || say "WARN $m exposes no vllm:* metric (harness will fail its prefix check)"
  else
    say "WARN cannot scrape $m"
  fi
  i=$((i + 1))
done
if [ "$used" -gt 0 ]; then
  if [ "$MODE" = official ] && [ "${KIT_ALLOW_USED_SERVER:-0}" != 1 ]; then
    die "server already served $used requests: official points need a fresh server (or KIT_ALLOW_USED_SERVER=1)"
  fi
  say "WARN server already served $used requests (not fresh)"
fi
if [ -f "$SERVER_LOG" ]; then cp "$SERVER_LOG" "$B/server.log"; else say "WARN no server log at $SERVER_LOG (KV pool tokens will be null)"; fi
if [ -n "$C_PSLOG" ]; then
  if [ -f "$C_PSLOG" ]; then cp "$C_PSLOG" "$B/server_prefill.log"; else say "WARN no prefill server log at $C_PSLOG"; fi
fi
fi   # DRY

# ---- manifest -----------------------------------------------------------------------------------------------------
IX_SHA=$(git -C "$IX" rev-parse HEAD 2>/dev/null || echo unknown)
python3 - "$B/manifest.json" <<EOF 2>/dev/null || say "WARN manifest not written (no host python3)"
import json, sys
json.dump({
  "tag": "$TAG", "server_url": "$URL", "concurrency": $CONC, "gpus_denominator": $GPUS, "mode": "$MODE",
  "duration": $DURATION, "fast": $FAST, "metrics_urls": "$METRICS", "client_image": "$CLIENT_IMAGE",
  "server_image_label": "${KIT_IMAGE_LABEL:-$C_IMAGE}", "config_id": "$C_ID", "inferencex_dir": "$IX",
  "inferencex_sha": "$IX_SHA", "venv": "$VENV_ROOT/venv", "agentx_harness": "1.0.14", "hf_home": "$HF_HOME_DIR",
  "dataset_revision": "$DATASET_REV", "dp_attention": "$DP_ATTN", "disagg": "${KIT_DISAGG:-${C_DISAGG:-false}}", "kv_offloading": "$KVO", "kv_backend_metadata": '$KVB_META',
  "router_metadata": '$ROUTER_META', "host": "$(hostname -s)", "started_utc": "$(date -u +%FT%TZ)",
  "slurm_job": "${SLURM_JOB_ID:-}", "extra_env": "${KIT_EXTRA_ENV:-}"}, open(sys.argv[1], "w"), indent=2)
EOF

# Tokenizer as a local dir (InferenceX replay.py:37 allows a checkpoint path): aiperf's HF-id lookup in offline mode
# demands a COMPLETE cached snapshot (incl. 48 weight shards) and failed on 11-19 (IncompleteSnapshotError).
TOK_DIR=${KIT_TOKENIZER_DIR:-$(ls -d "$HF_HOME_DIR"/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/*/ 2>/dev/null | head -1)}
TOK_DIR=${TOK_DIR%/}
[ -f "$TOK_DIR/tokenizer.json" ] || die "tokenizer dir not found: '$TOK_DIR'"
# ---- client container ---------------------------------------------------------------------------------------------
ENVS=(
  RESULT_DIR="$B" AGENTIC_OUTPUT_DIR="$B" RESULT_FILENAME=inferencex_result
  MODEL="$MODEL_ID" SERVED_MODEL_NAME="$MODEL_ID" TOKENIZER="$TOK_DIR" MODEL_PREFIX=dsv41flash FRAMEWORK=vllm PRECISION=fp4
  CONC="$CONC" DURATION="$DURATION" AIPERF_EXPERIMENTAL_FAST="$FAST"
  TP="$GPUS" EP_SIZE="${KIT_EP:-1}" PP_SIZE=1 PCP_SIZE=1 DCP_SIZE=1 DP_ATTENTION="$DP_ATTN"
  SPEC_DECODING=mtp SPEC_ALGORITHM=dspark SPEC_DRAFT_MODEL=native
  IS_MULTINODE=false DISAGG="${KIT_DISAGG:-${C_DISAGG:-false}}" EVAL_ONLY=false RUN_EVAL=false IS_AGENTIC=1 SCENARIO_TYPE=agentic-coding
  ENABLE_AGENTX_POWER=false REQUIRE_POWER=false
  KV_OFFLOADING="$KVO" KV_OFFLOAD_BACKEND="$KVB" KV_OFFLOAD_BACKEND_METADATA="$KVB_META" TOTAL_CPU_DRAM_GB="$DRAM_GB"
  ROUTER_METADATA="$ROUTER_META"
  AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS=300 AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=false
  AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING=0 WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
  AIPERF_REQUIRED_SERVER_METRIC_PREFIX='vllm:' AIPERF_SERVER_URL="$URL" AIPERF_SERVER_METRICS_URLS="$METRICS"
  IMAGE="${KIT_IMAGE_LABEL:-$C_IMAGE}" RUNNER_TYPE=mi355x-openai
  AIPERF_RUNTIME_DIR="$VENV_CTR" PYTHONPATH="$IX" PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
  HF_HOME="$HF_HOME_DIR" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
  HOME=/tmp/home TMPDIR=/tmp XDG_CACHE_HOME=/tmp/xdg LD_LIBRARY_PATH= LD_PRELOAD=   # image GPU libs not needed by the client
)
# shellcheck disable=SC2206
[ -n "${KIT_EXTRA_ENV:-}" ] && ENVS+=($KIT_EXTRA_ENV)
DOCKER=(docker run --rm --name "$CNAME" --network host --user "$(id -u):$(id -g)" --ulimit core=0
  --ulimit nofile=1048576:1048576 --shm-size 8g --label "yihou.dsv41.tag=$TAG" --label yihou.dsv41.role=client
  -v "$B:$B" -v "$HF_HOME_DIR:$HF_HOME_DIR" -v "$VENV_ROOT:$VENV_CTR:ro" -v "$IX:$IX:ro" -v "$LOC/tmp:/tmp"
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro)   # uid lookups (getpwuid) for the non-root client
for e in "${ENVS[@]}"; do DOCKER+=(-e "$e"); done
DOCKER+=(--entrypoint bash "$CLIENT_IMAGE" -c "mkdir -p /tmp/home /tmp/xdg && cd /tmp && exec $VENV_PY -m infx.bench agentic")
printf '%q ' "${DOCKER[@]}" > "$B/client_docker_cmd.txt"; echo >> "$B/client_docker_cmd.txt"
if [ "$DRY" = 1 ]; then say "DRY RUN: client argv in $B/client_docker_cmd.txt"; cat "$B/manifest.json" "$B/client_docker_cmd.txt"; exit 0; fi

if docker ps -a --format '{{.Names}}' | grep -qx "$CNAME"; then
  st=$(docker inspect -f '{{.State.Status}}' "$CNAME" 2>/dev/null)
  [ "$st" = exited ] || die "client container $CNAME exists in state '$st'"
  case "$CNAME" in yihou-dsv41-*-client) docker rm "$CNAME" >/dev/null && say "removed stale exited $CNAME" ;; esac
fi

on_term() {
  say "SIGTERM/INT: stopping $CNAME"
  case "$CNAME" in yihou-dsv41-*) docker stop -t 30 "$CNAME" >/dev/null 2>&1 ;; esac
  echo "terminated" > "$B/exit_code.txt"; exit 143
}
trap on_term TERM INT HUP

say "client start ($CNAME); log $B/client.log"
"${DOCKER[@]}" > "$B/client.log" 2>&1 &
DPID=$!
last_ok=$(date +%s)
while kill -0 "$DPID" 2>/dev/null; do
  sleep 60
  if [ "$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$URL/health")" = 200 ]; then last_ok=$(date +%s)
  else say "WARN server $URL/health not 200 for $(( $(date +%s) - last_ok ))s"; fi
done
wait "$DPID"; RC=$?
echo "$RC" > "$B/exit_code.txt"
say "client exited rc=$RC"
i=0
for m in "${MURLS[@]}"; do curl -s -m 20 -f "$m" > "$B/metrics_after_$i.txt" || say "WARN cannot scrape $m after run"; i=$((i + 1)); done
grep -E 'Saved aggregated|Requests:|QPS:|Validated required|ERROR' "$B/client.log" | tail -n 8 | tee -a "$L"

if [ "${KIT_NO_SUMMARY:-0}" != 1 ] && [ -f "$B/inferencex_result.json" ]; then
  mkdir -p "$NFS_ROOT/results"
  docker run --rm --network none --user "$(id -u):$(id -g)" -v "$RUN:$RUN:ro" -v "$STAGE/kit:$STAGE/kit:ro" \
    -v "$NFS_ROOT/results:$NFS_ROOT/results" -v "$VENV_ROOT:$VENV_CTR:ro" --entrypoint "$VENV_PY" "$CLIENT_IMAGE" \
    -I "$STAGE/kit/summarize.py" "$RUN" --mode "$MODE" --ledger "$NFS_ROOT/results/ledger.csv" 2>&1 | tee -a "$L"
fi
say "point done rc=$RC (server NOT stopped: stop.sh $TAG; fresh server per point)"
exit "$RC"
