#!/usr/bin/env bash
# Quick correctness of one or more site variants, one fresh deployment each with real acceptance: smoke.sh, the
# 200-prompt GSM8K probe at 64 in flight (template tags in the text, answers cut at max_tokens) and the long-context
# probe (no shared prefix, past the 2048-token sparse top-k threshold; LONGCTX_SPECS="" skips it); no lm-eval GSM8K.
# For bisecting garbled output; waits for both nodes' VRAM before each start.
# Usage: [LONGCTX_SPECS="16000:64"] quick_correct.sh KIT LOG_DIR SITE_ENV...
set -uo pipefail
kit="$1" log_dir="$2"
shift 2
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
probe_py="$here/../../mix-v0522-correctness-20261019/scripts/longctx_probe.py"
prompts="${PROBE_PROMPTS:-/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/bisect/hicache0-n200/runs/hicache0-n200/gsm8k/glm5.2-mxfp4/samples_gsm8k_2026-10-19T19-22-22.773074.jsonl}"
mkdir -p "$log_dir"
for site in "$@"; do
    (
        set -a
        # shellcheck source=/dev/null
        . "$site"
        set +a
        label="$(basename "$site" .env)-$(date -u +%m%dT%H%M)"
        log="$log_dir/$label.log"
        on() { if [[ "$1" == localhost ]]; then bash -c "$2"; else ssh -o BatchMode=yes "$1" "$2"; fi; }
        for n in localhost "$DECODE_SSH"; do
            until [[ -z "$(on "$n" "rocm-smi --showmeminfo vram --csv | awk -F, 'NR>1 && \$2 != \"\" {printf \"%d \", 100 * \$3 / \$2}'" | tr ' ' '\n' | awk '$1 >= 1')" ]]; do sleep 60; done
        done
        echo "$(date -u +%T) $label up"
        if (cd "$kit" && SIM_ACCEPT_LEN= RUN_ID="$label" bash engine/up.sh) >"$log" 2>&1; then
            (cd "$kit" && bash engine/smoke.sh) >>"$log" 2>&1
            echo "$(date -u +%T) $label smoke: $(grep -a 'checks passed' "$log" | tail -1); cold: $(grep -a '"long_prefix_cold"' "$log" | grep -o '"passed": [a-z]*')"
            timeout 3600 python3 "$here/../../mix-v0522-correctness-20261019/scripts/gsm8k_probe.py" "$prompts" \
                "http://$PREFILL_NODE_IP:${ROUTER_PORT:-28300}" "${SERVED_MODEL:-glm5.2-mxfp4}" "$log_dir/$label.probe.jsonl" \
                --limit 200 --concurrency 64 >>"$log" 2>&1
            echo "$(date -u +%T) $label probe: $(grep -a '^n=200 ' "$log" | tail -1)"
            for spec in ${LONGCTX_SPECS-16000:64}; do
                timeout 3600 python3 "$probe_py" "$prompts" "http://$PREFILL_NODE_IP:${ROUTER_PORT:-28300}" "${SERVED_MODEL:-glm5.2-mxfp4}" \
                    "$log_dir/$label.longctx-${spec%%:*}.jsonl" --ctx-tokens "${spec%%:*}" --n "${spec##*:}" --concurrency 32 >>"$log" 2>&1
                echo "$(date -u +%T) $label longctx $spec: $(grep -aE '^(needle|gsm8k): n=' "$log" | tail -2 | tr '\n' ' ')"
            done
        else
            echo "$(date -u +%T) $label up failed"
        fi
        (cd "$kit" && bash engine/down.sh) >>"$log" 2>&1
        echo "$(date -u +%T) $label down"
    )
done
