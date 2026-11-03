#!/usr/bin/env bash
# One fresh deployment of ARM_ENV with real MTP acceptance, checked in place. Prints one summary line per check
# ("<time> <label> <check>: ..."), which scripts/judge.py reads:
#   files     md5 of the 6 fix files in the decode container (fix / debug / orig) and of the 5 recipe files in the prefill
#   env       decode container env that decides the arm (ROCM700A, audit, KF switches)
#   start-up  "[dense-decode] DSA dual-graph enabled" lines and CUDA-graph capture times of the decode
#   smoke     engine/smoke.sh
#   probe     GSM8K probe (PROBE_N prompts, 64 in flight) through scripts/probe_stats.py; flagged samples -> .review.txt
#   longctx   long-context probe per LONGCTX_SPECS entry ("ctx_tokens:n", no shared prefix)
#   lm-eval   EVAL=1: InferenceX GSM8K (kit engine/eval.sh)
#   audit     arms with the variant audit: scripts/audit_stats.py on the decode log
# Runs on the prefill node; the decode node through `ssh $DECODE_SSH`. One retry when the stack does not come up.
# Usage: [PROBE_N=200] [LONGCTX_SPECS="16000:128 16000:128"] [EVAL=0] verify_suite.sh ARM_ENV
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PKG="${PKG:-$(cd "$here/.." && pwd)}"
arm="$1"
kit="$PKG/kit"
DEC_FILES="layers/attention/graph_variants.py model_executor/runner/decode_cuda_graph_runner.py managers/scheduler_components/dp_attn.py managers/schedule_batch.py model_executor/forward_batch_info.py batch_overlap/two_batch_overlap.py"
(
    set -a
    # shellcheck source=/dev/null
    . "$arm"
    set +a
    log_dir="$LOG_ROOT/verify"
    mkdir -p "$log_dir"
    prompts="${PROBE_PROMPTS:-$LOG_ROOT/gsm8k_probe_prompts.jsonl}"
    [[ -s "$prompts" ]] || gunzip -c "$PKG/data/gsm8k_probe_prompts.jsonl.gz" > "$prompts"
    name="$(basename "$arm" .env)"
    on() { if [[ "$1" == localhost ]]; then bash -c "$2"; else ssh -o BatchMode=yes "$1" "$2"; fi; }
    # busy GPUs (>= VRAM_IDLE_PCT % VRAM in use) among a comma-separated index list
    busy() { on "$1" "rocm-smi --showmeminfo vram --csv" | awk -F, -v l="$2" -v t="${VRAM_IDLE_PCT:-1}" 'BEGIN {n = split(l, g, ","); for (i = 1; i <= n; i++) w[g[i]] = 1}
        NR > 1 && $2 != "" {if ((NR - 2) in w && 100 * $3 / $2 >= t) b++} END {print b + 0}'; }
    for attempt in 1 2; do
        label="$name-$(date -u +%m%dT%H%M)"
        log="$log_dir/$label.log"
        until [[ "$(busy localhost "$PREFILL0_GPUS")" == 0 && "$(busy "$DECODE_SSH" "$DECODE_GPUS")" == 0 ]]; do sleep 60; done
        echo "$(date -u +%T) $label up"
        if (cd "$kit" && SIM_ACCEPT_LEN= RUN_ID="$label" bash engine/up.sh) >"$log" 2>&1; then
            break
        fi
        echo "$(date -u +%T) $label up failed (attempt $attempt): $(grep -aE 'stopped|Failed|Error|failed' "$log" | tail -1 | cut -c1-200)"
        (cd "$kit" && bash engine/down.sh) >>"$log" 2>&1
        [[ $attempt == 2 ]] && { echo "$(date -u +%T) $name failed twice"; exit 1; }
    done
    run="$WORK_DIR/runs/$label"
    SB=/sgl-workspace/sglang/python/sglang/srt
    dsum="$(on "$DECODE_SSH" "docker exec $CTR-decode sh -c 'cd $SB && md5sum $DEC_FILES'")"
    fixm=0 dbgm=0 orgm=0
    for f in $DEC_FILES; do
        s="$(awk -v f="$f" '$2 == f {print $1}' <<<"$dsum")"
        [[ "$s" == "$(md5sum < "$PKG/patches/fix/$f" | cut -d' ' -f1)" ]] && fixm=$((fixm + 1))
        [[ -f "$PKG/patches/debug/$f" && "$s" == "$(md5sum < "$PKG/patches/debug/$f" | cut -d' ' -f1)" ]] && dbgm=$((dbgm + 1))
        [[ "$s" == "$(awk -v f="$f" '$2 == f {print $1}' "$PKG/patches/BASE_MD5SUMS")" ]] && orgm=$((orgm + 1))
    done
    recm=0
    for pair in disaggregation/prefill.py:pp-chunk-wait/prefill.py mem_cache/unified_radix_cache.py:evict-headroom/unified_radix_cache.py \
        mem_cache/pool_host/mla.py:hicache-backup-quota/mla.py mem_cache/pool_host/dsa.py:hicache-backup-quota/dsa.py \
        models/deepseek_v2.py:deepseek_v2.py; do
        s="$(docker exec "$CTR-prefill0" md5sum "$SB/${pair%%:*}" | cut -d' ' -f1)"
        [[ "$s" == "$(md5sum < "$kit/patches/${pair#*:}" | cut -d' ' -f1)" ]] && recm=$((recm + 1))
    done
    echo "$(date -u +%T) $label files: decode fix=$fixm debug=$dbgm orig=$orgm of 6; prefill recipe=$recm of 5"
    echo "$(date -u +%T) $label env: $(on "$DECODE_SSH" "docker exec $CTR-decode env" | grep -E '^(SGLANG_USE_ROCM700A|SGLANG_DEBUG_GRAPH_VARIANT_AUDIT|YIHOU_KF_(SPARSE|INDEXER|MOE))=' | sort | tr '\n' ' ')"
    dlog="$(on "$DECODE_SSH" "docker logs $CTR-decode 2>&1")"
    echo "$(date -u +%T) $label start-up: dual_graph_lines=$(grep -ac 'DSA dual-graph enabled' <<<"$dlog") capture={$(grep -aoE 'cuda_graph=\{[^}]*\}' <<<"$dlog" | head -1 | sed 's/cuda_graph={//; s/}//')} spec_args=$(on "$DECODE_SSH" "docker inspect -f '{{json .Args}}' $CTR-decode" | grep -o -- '--speculative-[a-z-]*' | wc -l)"
    (cd "$kit" && bash engine/smoke.sh) >>"$log" 2>&1
    echo "$(date -u +%T) $label smoke: $(grep -a 'checks passed' "$log" | tail -1)"
    if [[ "${PROBE_N:-200}" != 0 ]]; then
        timeout 5400 python3 "$here/gsm8k_probe.py" "$prompts" "http://$PREFILL_NODE_IP:${ROUTER_PORT:-28300}" "$SERVED_MODEL" \
            "$log_dir/$label.probe.jsonl" --limit "${PROBE_N:-200}" --concurrency 64 >>"$log" 2>&1
        echo "$(date -u +%T) $label probe: $(python3 "$here/probe_stats.py" "$log_dir/$label.probe.jsonl" --review "$log_dir/$label.review.txt")"
    fi
    k=0
    for spec in ${LONGCTX_SPECS-16000:128 16000:128}; do
        k=$((k + 1))
        timeout 5400 python3 "$here/longctx_probe.py" "$prompts" "http://$PREFILL_NODE_IP:${ROUTER_PORT:-28300}" "$SERVED_MODEL" \
            "$log_dir/$label.longctx-$k-${spec%%:*}.jsonl" --ctx-tokens "${spec%%:*}" --n "${spec##*:}" --concurrency "${LONGCTX_CONC:-32}" >>"$log" 2>&1
        echo "$(date -u +%T) $label longctx $spec: $(grep -aE '^(needle|gsm8k): n=' "$log" | tail -2 | tr '\n' ' ')"
    done
    if [[ "${EVAL:-0}" == 1 ]]; then
        (cd "$kit" && bash engine/eval.sh) >>"$log" 2>&1
        res="$(find "$run/gsm8k" -name 'results*.json' -newer "$log_dir/$label.probe.jsonl" 2>/dev/null | head -1)"
        echo "$(date -u +%T) $label lm-eval: $( [[ -n "$res" ]] && python3 -c "
import json, sys
r = json.load(open(sys.argv[1]))['results']; t = next(iter(r.values()))
print('strict', t.get('exact_match,strict-match'), 'flexible', t.get('exact_match,flexible-extract'))" "$res" || echo missing)"
    fi
    (cd "$kit" && bash engine/down.sh) >>"$log" 2>&1
    if [[ "${DECODE_ENV_EXTRA:-}" == *SGLANG_DEBUG_GRAPH_VARIANT_AUDIT=1* ]]; then
        echo "$(date -u +%T) $label audit: $(python3 "$here/audit_stats.py" "$run/logs/$CTR-decode.log" | tr '\n' ' ')"
    fi
    echo "$(date -u +%T) $label down"
)
