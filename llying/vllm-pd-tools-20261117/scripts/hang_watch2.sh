#!/usr/bin/env bash
# Watch one vLLM engine for a silent hang and, once, dump its container's Python stacks plus kernel wait channels:
#   prefill: vllm:prompt_tokens_total unchanged for STALL s while requests wait;
#   decode:  vllm:generation_tokens_total unchanged for STALL s while requests run.
# Python processes are found by executable (EngineCore / workers rename themselves, so pgrep -f python misses them) and
# dumped with py-spy --native; each process also gets its threads' /proc wchan counts (kernel wait points, no root
# needed) and the dump ends with the host's kernel log tail. With QDIR set the dump runs as an item in $R/$QDIR (the
# container is on that queue's node; not waited for).
#   hang_watch2.sh URL CONTAINER OUT_DIR prefill|decode [STALL=180] [DURATION=9000]
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}
IMAGE=${IMAGE:-infera-vllm:v0340-0c5e5e89}
url=$1 ctr=$2 out=$3 kind=$4 stall=${5:-180} dur=${6:-9000}
mkdir -p "$out"
if [[ "$kind" == decode ]]; then
    progress='vllm:generation_tokens_total{' busy='vllm:num_requests_running{'
else
    progress='vllm:prompt_tokens_total{' busy='vllm:num_requests_waiting{'
fi
read_m() {
    curl -sf -m 10 "$url/metrics" | awk -v p="$progress" -v b="$busy" \
        'index($0, p) == 1 {x += $NF} index($0, b) == 1 {y += $NF} END {printf "%d %d\n", x, y}'
}
# shellcheck disable=SC2016
inner='pip install -q py-spy > /dev/null 2>&1
for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    case "$(readlink "$d/exe" 2>/dev/null)" in *python*) ;; *) continue ;; esac
    echo "===== $pid $(cat "$d/comm") $(tr "\0" " " < "$d/cmdline" | cut -c1-150)"
    echo "-- thread wchan:"; for t in "$d"/task/*; do cat "$t/wchan" 2>/dev/null; echo; done | sort | uniq -c | sort -rn | head -12
    timeout 90 py-spy dump --native --pid "$pid" 2>&1 | head -200
done'
dump_cmd="docker exec $ctr bash -c $(printf %q "$inner") > $out/pyspy-$kind.txt 2>&1; docker run --rm --cap-add SYSLOG --network none --entrypoint bash $IMAGE -c 'dmesg -T | tail -120' > $out/dmesg-$kind.txt 2>&1"
t_end=$(( $(date +%s) + dur ))
last_p=-1 last_t=$(date +%s)
while (( $(date +%s) < t_end )); do
    read -r p w < <(read_m || echo "-1 -1")
    now=$(date +%s)
    if [[ "$p" != "$last_p" ]]; then last_p=$p last_t=$now; fi
    if (( p >= 0 && w > 0 && now - last_t >= stall )); then
        echo "$(date -u +%T) $kind stall: progress $p unchanged for $((now - last_t)) s with $w busy; dumping" | tee -a "$out/stall.txt"
        if [[ -n "${QDIR:-}" ]]; then
            id="$R/$QDIR/$(date -u +%m%d-%H%M%S)-hangdump-$kind"
            { echo "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=$HOME USER=$USER R=$R"
              echo "$dump_cmd"; } > "$id.tmp" && mv "$id.tmp" "$id.sh"
            echo "$(date -u +%T) dump queued as $id" | tee -a "$out/stall.txt"
        else
            bash -c "$dump_cmd"
            echo "$(date -u +%T) dumped to $out/pyspy-$kind.txt" | tee -a "$out/stall.txt"
        fi
        exit 0
    fi
    sleep 20
done
