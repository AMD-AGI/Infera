#!/bin/bash
# Wakes the agent session (lines starting with AGENT_LOOP_WAKE_kfd619) on events of the two-node job 36242:
# state changes (start, preemption / requeue, end), each queue item that finishes, the queue running empty,
# and a heartbeat after 3 h without any event. One squeue call and one file read per minute.
unset ${!SLURM@}
K=/perf_apps/liyingli/bench_agentx/kfd619
LOG=$K/logs/holder.log
JOB=36242
HEARTBEAT=$((3 * 3600))
seen=$(wc -l < "$LOG")
prev=$(squeue -h -j $JOB -o %T 2>/dev/null); prev=${prev:-GONE}
last=$(date +%s)
wake() {  # EVENT PROMPT
    local e=${1//\"/\'}
    echo "AGENT_LOOP_WAKE_kfd619 {\"time\":\"$(date -u +%FT%TZ)\",\"event\":\"$e\",\"prompt\":\"$2\"}"
    last=$(date +%s)
}
echo "watching job $JOB (state $prev, holder.log line $seen)"
while true; do
    sleep 60
    st=$(squeue -h -j $JOB -o %T 2>/dev/null); st=${st:-GONE}
    if [ "$st" != "$prev" ]; then
        if [ "$st" = RUNNING ]; then
            wake "job $JOB $prev -> RUNNING on $(squeue -h -j $JOB -o %N)" \
                "作业 36242 开始了：确认 6.19 / 6.16 节点分配对、外来容器已清，看 81 编译是否在跑"
        else
            wake "job $JOB $prev -> $st" "作业 36242 状态变化：看是被抢占重排还是结束，队列还剩哪些，需要的话重排"
        fi
        prev=$st
    fi
    n=$(wc -l < "$LOG")
    if [ "$n" -gt "$seen" ]; then
        while read -r l; do
            left=$(ls "$K"/queue/*.sh 2>/dev/null | wc -l)
            wake "$l (queue left $left)" "条目跑完了：看输出、记进报告并提交；队列空了就把结果填进 PR 草稿，发给用户确认"
        done < <(sed -n "$((seen + 1)),${n}p" "$LOG" | grep -E ": done |NODE619=")
        seen=$n
    fi
    if [ $(( $(date +%s) - last )) -ge $HEARTBEAT ]; then
        wake "heartbeat, job $JOB $st" "心跳：检查作业 36242 和队列状态，节点没有空闲"
    fi
done
