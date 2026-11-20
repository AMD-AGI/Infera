#!/usr/bin/env bash
# E7 PD correctness smoke. Run ON the node after serve_pd.sh (PD stack, tag <pd-tag>) and serve.sh (colocated fp8_ds_mla
# reference) are both healthy:
#   1. greedy prompts (short, code, one > 8192 prompt tokens, and that long one again) through the PD router and through the
#      reference; diff the outputs (pd_smoke.py; stdlib python, every HTTP call has a timeout)
#   2. P/D Prometheus deltas: Mooncake transfers/bytes on P, failed transfers/recvs, external (p2p) prefix hits on D
#   3. grep the router / P / D logs for the bootstrap engine-id query, Mooncake init and every known failure line
# Pair: PD = configs/e7s-pd-mooncake-1p1d-dp2-smoke.env, reference = configs/e7r-colo-dsmla-ref-smoke.env (both "standard"
# rejection; under "synthetic" rejection outputs are not comparable). See scripts/kit/PD_README.md.
#
# Usage: pd_smoke.sh <pd-tag> <ref-url> [options]
#   pd-tag      tag given to serve_pd.sh (router role config at $DSV41_NFS_ROOT/runs/<pd-tag>/kit/config.env)
#   ref-url     base URL of the reference server, e.g. http://127.0.0.1:8040 (serve.sh ... --port-offset 40)
# Options:
#   --pd-url URL        PD router base URL (default http://127.0.0.1:<ROUTER_PORT of the router config>)
#   --endpoint E        chat (default) | completions
#   --max-tokens N      default 256
#   --timeout S         per-request timeout, default 900
#   --docker            run pd_smoke.py inside the config's IMAGE (host has no python3); container yihou-dsv41-<tag>-smoke
#   --out DIR           default $DSV41_NFS_ROOT/runs/<pd-tag>/smoke-<UTC stamp>
# Exit: pd_smoke.py's code (0 PASS, 1 CHECK, 2 unreachable) or 3 if a failure pattern is found in the logs.
set -euo pipefail
KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJ=dsv41_vllm_pd_20261119
NFS_ROOT=${DSV41_NFS_ROOT:-/perf_apps/yihou/$PROJ}
LOCAL_ROOT=${DSV41_LOCAL_ROOT:-/data/yihou/$PROJ}
die() { echo "pd_smoke.sh: ERROR: $*" >&2; exit 2; }
[ $# -ge 2 ] || die "usage: pd_smoke.sh <pd-tag> <ref-url> [--pd-url URL] [--endpoint chat|completions] [--max-tokens N] [--timeout S] [--docker] [--out DIR]"
TAG=$1; REF=$2; shift 2
PDURL=""; EP=chat; MT=256; TO=900; DOCK=0; OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --pd-url) PDURL=${2:?}; shift ;;
    --endpoint) EP=${2:?}; shift ;;
    --max-tokens) MT=${2:?}; shift ;;
    --timeout) TO=${2:?}; shift ;;
    --docker) DOCK=1 ;;
    --out) OUT=${2:?}; shift ;;
    *) die "unknown option $1" ;;
  esac
  shift
done
[[ "$TAG" =~ ^[a-z0-9][a-z0-9._-]{0,52}$ ]] || die "unsafe tag '$TAG'"
[[ "$REF" =~ ^https?://[A-Za-z0-9._:-]+/?$ ]] || die "bad ref-url '$REF'"
[ -z "$PDURL" ] || [[ "$PDURL" =~ ^https?://[A-Za-z0-9._:-]+/?$ ]] || die "bad --pd-url '$PDURL'"
[[ "$MT" =~ ^[0-9]+$ ]] && [[ "$TO" =~ ^[0-9]+$ ]] || die "bad --max-tokens/--timeout"
case "$NFS_ROOT" in */yihou/*) ;; *) die "NFS root '$NFS_ROOT' does not contain /yihou/" ;; esac

RUN=$NFS_ROOT/runs/$TAG
CFG=$RUN/kit/config.env
[ -f "$CFG" ] || die "no $CFG (is <pd-tag> the router tag given to serve_pd.sh?)"
[ "$(stat -c %u "$CFG")" = "$(id -u)" ] && [ "$(( 0$(stat -c %a "$CFG") & 022 ))" = 0 ] \
  || die "unsafe $CFG (owner/mode); refusing to source"
cfgv() { bash -c 'source "$1" >/dev/null 2>&1; printf "%s" "${!2:-}"' _ "$CFG" "$1"; }
[ "$(cfgv PD_ROLE)" = router ] || die "$CFG is not a PD router role config (PD_ROLE=router)"
RPORT=$(cfgv ROUTER_PORT); IMAGE=$(cfgv IMAGE)
MURLS=$(cfgv METRICS_URLS); P_TAG=$(cfgv PD_P_TAG); D_TAG=$(cfgv PD_D_TAG)
PDURL=${PDURL:-http://127.0.0.1:$RPORT}
MP=${MURLS%%,*}; MD=${MURLS#*,}
OUT=${OUT:-$RUN/smoke-$(date -u +%Y%m%dT%H%M%SZ)}
case "$OUT" in */yihou/*) ;; *) die "out dir '$OUT' does not contain /yihou/" ;; esac
mkdir -p "$OUT"; chmod 1777 "$OUT" 2>/dev/null || true
L=$OUT/pd_smoke.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L"; }
say "pd_smoke: tag=$TAG pd=$PDURL ref=$REF metrics P=$MP D=$MD endpoint=$EP max_tokens=$MT out=$OUT"

# ---- 1+2: prompts, diff, metric deltas --------------------------------------------------------------------------------
PYARGS=(--pd "$PDURL" --ref "$REF" --metrics-p "$MP" --metrics-d "$MD" --endpoint "$EP" --max-tokens "$MT" --timeout "$TO")
GLOBAL_TO=$(( TO * 9 + 300 ))   # 4 prompts x 2 servers + scrapes; hard cap so the smoke can never hang
set +e
if [ "$DOCK" = 1 ]; then
  command -v docker >/dev/null || die "docker not found"
  timeout "$GLOBAL_TO" docker run --rm --name "yihou-dsv41-$TAG-smoke" --network host --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$KIT_DIR/pd_smoke.py:/yihou/pd_smoke.py:ro" -v "$OUT:/yihou/out" --entrypoint python3 "$IMAGE" \
    -I /yihou/pd_smoke.py "${PYARGS[@]}" --out /yihou/out 2>&1 | tee -a "$L"
  rc=${PIPESTATUS[0]}
else
  command -v python3 >/dev/null || die "no python3 on this host: rerun with --docker"
  timeout "$GLOBAL_TO" python3 -I "$KIT_DIR/pd_smoke.py" "${PYARGS[@]}" --out "$OUT" 2>&1 | tee -a "$L"
  rc=${PIPESTATUS[0]}
fi
set -e
say "pd_smoke.py rc=$rc (0 PASS, 1 CHECK, 2 unreachable, 124 global timeout)"

# ---- 3: logs ----------------------------------------------------------------------------------------------------------
logof() {  # logof <tag> <file>: NFS run dir first, node-local fallback (serve.sh entry falls back when NFS is not writable)
  local f
  for f in "$NFS_ROOT/runs/$1/$2" "$LOCAL_ROOT/$1/logs/$2"; do [ -f "$f" ] && { echo "$f"; return 0; }; done
  echo ""
}
RLOG=$(logof "$TAG" router.log); PLOG=$(logof "$P_TAG" server.log); DLOG=$(logof "$D_TAG" server.log)
PREW=$(logof "$TAG" router_prewait.log)
# Info lines (vLLM main 7a62ca06 mooncake_connector.py / router v0.1.14 vllm_pd_router.rs) - presence expected:
INFO_RE='Got Mooncake engine_ids|dp ranks found|Initializing Mooncake Transfer Engine|Mooncake Transfer Engine is using|Registering KV_Caches|KV Transfer metrics|ignored with a KV connector'
# Failure lines - any hit makes the smoke fail:
FAIL_RE='KV group count mismatch|local blocks\([0-9]+\) < remote blocks|pulling kv_caches for .* failed|Failed to connect to bootstrap|Bootstrap query to .* failed|failed to query Mooncake bootstrap|Failed to query Mooncake bootstrap|were pulled after their KV expired|Timeout waiting for P side ready|expired before sending|timed out after .* without being sent|KV lease of request .* expired|lost the KV layout|Missing transfer_id|Got invalid KVTransferParams|Sending to .* failed|Error in Mooncake sender|Error processing Mooncake xfer|Mooncake is not available|Traceback|EngineDeadError|RuntimeError'
lrc=0
for pair in "router:$RLOG" "prewait:$PREW" "prefill:$PLOG" "decode:$DLOG"; do
  n=${pair%%:*}; f=${pair#*:}
  if [ -z "$f" ]; then say "WARN no $n log found"; continue; fi
  say "== $n log $f"
  { grep -E "$INFO_RE" "$f" | tail -n 12 || true; } | tee -a "$L"
  if grep -qE "$FAIL_RE" "$f"; then
    say "FAIL patterns in $n log:"; grep -nE "$FAIL_RE" "$f" | tail -n 20 | tee -a "$L"; lrc=3
  fi
done
[ -n "$PREW" ] && grep -q 'ranks \[' "$PREW" && say "bootstrap ranks seen by pre-wait: $(grep -o 'ranks \[.*\]' "$PREW" | tail -n 1)"
final=$rc; [ "$rc" = 0 ] && [ "$lrc" != 0 ] && final=$lrc
say "pd_smoke: DONE rc=$final (outputs: $OUT/*.txt, $OUT/smoke.json)"
exit "$final"
