#!/usr/bin/env bash
# One-shot health check of our serving/bench runs; compact status for the 10-minute watchdog teammate. Read-only:
# never stops, removes or restarts anything.
#
# Usage: watchdog.sh <tag> [<tag> ...]
#   On the node: runs locally. From the dev host: set KIT_NODE=<node> (ssh, BatchMode); if ssh is refused
#   (pam_slurm_adopt, early .extern end) and KIT_JOBID=<slurm job> is set, retries via
#   `ssh openai-login srun --jobid <job> --overlap -N1 -n1` (the documented openai fallback).
# Per tag: containers yihou-dsv41-<tag>* (name+status), endpoint /health, server-log age + last startup marker,
# error lines (Traceback|ERROR|OOM|...) in the last KIT_TAIL lines of runs/<tag>/{server,router,umbp,entry}.log,
# bench state (exit code / elapsed / last client line / records so far), then one GPU line (use% and VRAM%).
# Exit 0 = nothing alarming, 1 = at least one ALERT, 3 = node unreachable.
set -uo pipefail
KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SELF=$KIT_DIR/$(basename "${BASH_SOURCE[0]}")
NFS_ROOT=${DSV41_NFS_ROOT:-/perf_apps/yihou/dsv41_vllm_pd_20261119}
TAIL=${KIT_TAIL:-3000}
[ $# -ge 1 ] || { echo "usage: watchdog.sh <tag> [<tag> ...]  (KIT_NODE=<node> [KIT_JOBID=<job>] from the dev host)" >&2; exit 2; }
for t in "$@"; do [[ "$t" =~ ^[a-z0-9][a-z0-9._-]{0,55}$ ]] || { echo "unsafe tag '$t'" >&2; exit 2; }; done

# ---- remote dispatch ----------------------------------------------------------------------------------------------
if [ -n "${KIT_NODE:-}" ] && [ "${KIT_NODE%%.*}" != "$(hostname -s)" ]; then
  remote="DSV41_NFS_ROOT=$NFS_ROOT KIT_TAIL=$TAIL bash $SELF $*"
  out=$(timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=10 "$KIT_NODE" "$remote" 2>&1); rc=$?
  if [ $rc -eq 255 ] || grep -q 'pam_slurm_adopt\|Access denied' <<<"$out"; then
    if [ -n "${KIT_JOBID:-}" ]; then
      out=$(timeout 180 ssh -o BatchMode=yes -o ConnectTimeout=10 openai-login \
        "srun --jobid ${KIT_JOBID} --overlap -N1 -n1 bash -c '$remote'" 2>&1); rc=$?
      echo "[via srun --overlap job ${KIT_JOBID}]"
    else
      echo "UNREACHABLE $KIT_NODE: $(tail -n 1 <<<"$out" | cut -c1-160) (set KIT_JOBID for the srun --overlap fallback)"
      exit 3
    fi
  fi
  echo "$out"; exit $rc
fi

# ---- local checks -------------------------------------------------------------------------------------------------
now=$(date +%s); ALERT=0
alert() { ALERT=1; echo "  ALERT $*"; }
ERR_RE='Traceback|EngineDeadError|OutOfMemory|out of memory|OOM|Segmentation fault|core dumped|Killed|HSA_STATUS_ERROR|hipError|RuntimeError|(^|[[:space:]])ERROR[[:space:]]|NCCL error|RCCL error'
BENIGN_RE='HIPBLAS_STATUS|hipblaslt.*fallback'
MARK_RE='Loading weights|Model loading took|GPU KV cache size|Capturing CUDA graph|Graph capturing finished|init engine|Application startup complete|Uvicorn running|listening'
echo "watchdog $(date -u +%FT%TZ) host=$(hostname -s)"
for t in "$@"; do
  R=$NFS_ROOT/runs/$t; B=$R/bench
  echo "[$t]"
  ctrs=$(docker ps -a --format '{{.Names}}|{{.Status}}' 2>/dev/null | awk -F'|' -v p="yihou-dsv41-$t" '$1==p || index($1, p"-")==1')
  if [ -z "$ctrs" ]; then echo "  containers: none"; else
    while IFS='|' read -r n s; do
      echo "  ctr $n: $s"
      case "$s" in Up*) ;; *) [ "$n" != "yihou-dsv41-$t-client" ] && alert "$n not running ($s)";; esac
    done <<<"$ctrs"
  fi
  # endpoint
  url=""; code=""
  [ -f "$B/manifest.json" ] && url=$(sed -n 's/.*"server_url": *"\([^"]*\)".*/\1/p' "$B/manifest.json" | head -1)
  if [ -z "$url" ] && [ -f "$R/kit/config.env" ] && [ "$(stat -c %u "$R/kit/config.env")" = "$(id -u)" ] \
     && [ "$(( 0$(stat -c %a "$R/kit/config.env") & 022 ))" = 0 ]; then
    p=$(bash -c 'source "$1" >/dev/null 2>&1; if [ "${ROUTER_ENABLE:-0}" = 1 ]; then echo "$ROUTER_PORT"; else echo "$PORT"; fi' _ "$R/kit/config.env")
    [ -n "$p" ] && url=http://127.0.0.1:$p
  fi
  if [ -n "$url" ]; then
    code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$url/health")
    echo "  health $url: $code"
    if [ "$code" != 200 ] && grep -q '^Up' <<<"$(cut -d'|' -f2 <<<"$ctrs" | head -1)"; then
      echo "  (not ready: startup in progress or hung -- see startup marker / log age)"
    fi
  else
    echo "  health: no endpoint known (no bench/manifest.json or config.env)"
  fi
  # logs
  for lf in server router umbp entry; do
    f=$R/$lf.log; [ -f "$f" ] || continue
    age=$(( now - $(stat -c %Y "$f") ))
    errs=$(tail -n "$TAIL" "$f" | grep -E "$ERR_RE" | grep -Ev "$BENIGN_RE")
    ne=$(grep -c . <<<"$errs"); [ -z "$errs" ] && ne=0
    line="  $lf.log: age ${age}s errors(last $TAIL lines)=$ne"
    if [ "$lf" = server ]; then
      mk=$(tail -n 20000 "$f" | grep -E "$MARK_RE" | tail -n 1 | cut -c1-140)
      [ -n "$mk" ] && line="$line | last marker: $mk"
    fi
    echo "$line"
    [ "$ne" -gt 0 ] && alert "$lf.log: $(tail -n 1 <<<"$errs" | cut -c1-200)"
  done
  # bench
  if [ -d "$B" ]; then
    if [ -f "$B/exit_code.txt" ]; then
      rc=$(cat "$B/exit_code.txt"); sc=""
      [ -f "$B/inferencex_result.json" ] && sc=$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(round(d["request_metrics"]["throughput"]["per_gpu"]["total_tput_tps"],1))' "$B/inferencex_result.json" 2>/dev/null)
      echo "  bench: DONE rc=$rc ${sc:+score=$sc tok/s/GPU}"
      [ "$rc" != 0 ] && alert "bench rc=$rc (see $B/client.log)"
    elif [ -f "$B/run_point.log" ]; then
      st=$(head -n 1 "$B/run_point.log" | cut -d' ' -f1); el=$(( now - $(date -u -d "$st" +%s 2>/dev/null || echo "$now") ))
      dur=$(sed -n 's/.*"duration": *\([0-9]*\).*/\1/p' "$B/manifest.json" 2>/dev/null | head -1)
      nrec=$( [ -f "$B/aiperf_artifacts/profile_export.jsonl" ] && wc -l < "$B/aiperf_artifacts/profile_export.jsonl" || echo n/a)
      last=$(grep -v '^\s*$' "$B/client.log" 2>/dev/null | tail -n 1 | tr -d '\r' | cut -c1-160)
      echo "  bench: RUNNING elapsed ${el}s (profile ${dur:-?}s + warmup/drain) records=$nrec"
      [ -n "$code" ] && [ "$code" != 200 ] && alert "bench running but $url/health=$code"
      echo "  client last: $last"
      cage=$(( now - $(stat -c %Y "$B/client.log" 2>/dev/null || echo "$now") ))
      [ "$cage" -gt 900 ] && alert "client.log silent for ${cage}s"
      grep -q "^yihou-dsv41-$t-client|Up" <<<"$ctrs" || alert "bench running per run_point.log but client container not Up"
      grep -qE 'WARN server .*health not 200' <(tail -n 3 "$B/run_point.log") && alert "$(tail -n 1 "$B/run_point.log" | cut -c1-160)"
    else
      echo "  bench: not started"
    fi
  fi
done
# GPUs (one line)
if command -v rocm-smi >/dev/null 2>&1; then
  g=$(rocm-smi --showuse --showmemuse --csv 2>/dev/null | awk -F, 'NR>1 && $1 ~ /card/ {gsub("card","",$1); printf "%s:%s%%/%s%% ", $1, $2, $3}')
  echo "GPU use%/VRAM%: ${g:-unavailable}"
fi
[ $ALERT = 1 ] && { echo "STATUS: ALERT"; exit 1; }
echo "STATUS: OK"
