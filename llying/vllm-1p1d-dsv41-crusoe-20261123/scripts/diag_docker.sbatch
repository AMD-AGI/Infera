#!/usr/bin/env bash
#SBATCH -J lyl-dsv41-diag
#SBATCH -p amd-spur-v2
#SBATCH -A amd-hyperloom
#SBATCH -q amd-hyperloom-qos
#SBATCH -N 1
#SBATCH --gres gpu:1
#SBATCH -c 2
#SBATCH -t 00:05:00
#SBATCH -D /tmp
#SBATCH -o /home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/diag-%j.out
# Why a shared (non-exclusive) job cannot reach docker: identity, groups, the socket, the API answer.
echo "host $(hostname -s) ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-}"
id
ls -l /var/run/docker.sock
curl -sS -m 15 -o /dev/null -w 'images/json http %{http_code}\n' --unix-socket /var/run/docker.sock http://localhost/images/json 2>&1
curl -sS -m 15 --unix-socket /var/run/docker.sock http://localhost/_ping 2>&1; echo
timeout 30 docker version --format '{{.Server.Version}}' 2>&1 | tail -2
cat /proc/self/status | grep -E '^(Groups|CapEff|NSpid)'
exit 0
