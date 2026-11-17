#!/usr/bin/env bash
# Watch the prefill for a silent hang: if vllm:prompt_tokens_total does not move for STALL s while requests wait,
# dump every Python stack in the container with py-spy (installed on first use) into OUT_DIR, once.
#   hang_watch.sh PREFILL_URL CONTAINER OUT_DIR [STALL=180] [DURATION=7200]
set -uo pipefail
url=$1 ctr=$2 out=$3 stall=${4:-180} dur=${5:-7200}
mkdir -p "$out"
read_m() { curl -sf -m 10 "$url/metrics" | awk '/^vllm:prompt_tokens_total/ {p += $2} /^vllm:num_requests_waiting\{/ {w += $2} END {printf "%d %d\n", p, w}'; }
t_end=$(( $(date +%s) + dur ))
last_p=-1 last_t=$(date +%s)
while (( $(date +%s) < t_end )); do
    read -r p w < <(read_m || echo "-1 -1")
    now=$(date +%s)
    if [[ "$p" != "$last_p" ]]; then last_p=$p last_t=$now; fi
    if (( p >= 0 && w > 0 && now - last_t >= stall )); then
        echo "$(date -u +%T) stall: prompt_tokens $p unchanged for $((now - last_t)) s with $w waiting; dumping" | tee "$out/stall.txt"
        docker exec "$ctr" bash -c 'pip install -q py-spy > /dev/null 2>&1; for pid in $(pgrep -f python); do echo "===== $pid $(tr "\0" " " < /proc/$pid/cmdline | cut -c1-150)"; py-spy dump --pid $pid 2>&1 | head -80; done' > "$out/pyspy.txt" 2>&1
        echo "$(date -u +%T) dumped to $out/pyspy.txt" | tee -a "$out/stall.txt"
        exit 0
    fi
    sleep 20
done
