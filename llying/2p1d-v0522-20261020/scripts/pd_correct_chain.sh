#!/usr/bin/env bash
# Correctness run on a kit, after both nodes' GPUs have released their VRAM: a fresh deployment with
# real MTP acceptance -> smoke.sh -> eval.sh (GSM8K) -> down.sh. Before it records who else is on the
# nodes into LOG_DIR/correct.env-check for manual review; it does not judge them.
# Usage: pd_correct_chain.sh KIT SITE_ENV LOG_DIR RUN_ID
set -uo pipefail
kit="$1" site="$2" log_dir="$3" run_id="$4"
set -a
# shellcheck source=/dev/null
. "$site"
set +a
mkdir -p "$log_dir"
nodes=(localhost "$DECODE_SSH")
on() { if [[ "$1" == localhost ]]; then bash -c "$2"; else ssh -o BatchMode=yes "$1" "$2"; fi; }
vram() { on "$1" "rocm-smi --showmeminfo vram --csv | awk -F, 'NR>1 && \$2 != \"\" {printf \"%d \", 100 * \$3 / \$2}'"; }

for n in "${nodes[@]}"; do
    until [[ -z "$(vram "$n" | tr ' ' '\n' | awk '$1 >= 1')" ]]; do sleep 60; done
done
{
    for n in "${nodes[@]}"; do
        echo "===== $n $(date -u +%FT%TZ)"
        on "$n" "rocm-smi --showmeminfo vram --csv; rocm-smi --showpids; docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}'; uptime; ps -eo user,pcpu,etime,comm --sort=-pcpu | head -12"
    done
} >"$log_dir/$run_id.env-check" 2>&1
echo "$(date -u +%T) $run_id up"
(cd "$kit" && SIM_ACCEPT_LEN= RUN_ID="$run_id" bash engine/up.sh) >"$log_dir/$run_id.log" 2>&1 && {
    echo "$(date -u +%T) $run_id smoke"
    (cd "$kit" && bash engine/smoke.sh) >>"$log_dir/$run_id.log" 2>&1
    echo "$(date -u +%T) $run_id smoke exited $?"
    (cd "$kit" && bash engine/eval.sh) >>"$log_dir/$run_id.log" 2>&1
    echo "$(date -u +%T) $run_id eval exited $?"
    # PROBE_PROMPTS (an lm-eval GSM8K samples file): 200 prompts at 64 in flight through the router,
    # counting template tags in the text and answers cut at max_tokens, which lm-eval cannot see.
    if [[ -n "${PROBE_PROMPTS:-}" ]]; then
        timeout 5400 python3 "$(dirname "$0")/../../mix-v0522-correctness-20261019/scripts/gsm8k_probe.py" \
            "$PROBE_PROMPTS" "http://$PREFILL_NODE_IP:${ROUTER_PORT:-28300}" "${SERVED_MODEL:-glm5.2-mxfp4}" \
            "$log_dir/$run_id.probe.jsonl" --limit 200 --concurrency 64 >>"$log_dir/$run_id.log" 2>&1
        echo "$(date -u +%T) $run_id probe exited $?: $(tail -1 "$log_dir/$run_id.log")"
    fi
    # LONGCTX_SPECS ("CTX_TOKENS:N ..."): long prompts with no shared prefix, past the 2048 tokens above which the
    # DSA indexer selects a sparse top-k (GSM8K prompts never get there); needle + long-prefix GSM8K per spec.
    for spec in ${LONGCTX_SPECS:-}; do
        timeout 5400 python3 "$(dirname "$0")/../../mix-v0522-correctness-20261019/scripts/longctx_probe.py" \
            "$PROBE_PROMPTS" "http://$PREFILL_NODE_IP:${ROUTER_PORT:-28300}" "${SERVED_MODEL:-glm5.2-mxfp4}" \
            "$log_dir/$run_id.longctx-${spec%%:*}.jsonl" --ctx-tokens "${spec%%:*}" --n "${spec##*:}" \
            --concurrency 32 >>"$log_dir/$run_id.log" 2>&1
        echo "$(date -u +%T) $run_id longctx $spec exited $?: $(tail -2 "$log_dir/$run_id.log" | tr '\n' ' ')"
    done
}
(cd "$kit" && bash engine/down.sh) >>"$log_dir/$run_id.log" 2>&1
echo "$(date -u +%T) $run_id down"
