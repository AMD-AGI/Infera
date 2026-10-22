#!/usr/bin/env bash
# Node-side: wait until decode num_running_reqs >= MIN, then POST /start_profile (num_steps N, CPU+GPU).
D=http://10.245.144.125:29357; OUT=$1; MIN=${2:-25}; N=${3:-20}
for i in $(seq 1 120); do
  r=$(curl -s -m 10 $D/metrics | awk '/^sglang:num_running_reqs\{/{print int($2); exit}')
  echo "$(date -u +%T) running=$r"
  if [[ -n "$r" && "$r" -ge "$MIN" ]]; then
    curl -s -m 600 -X POST $D/start_profile -H 'Content-Type: application/json' \
      -d "{\"output_dir\":\"$OUT\",\"num_steps\":$N,\"activities\":[\"CPU\",\"GPU\"],\"record_shapes\":false}"
    echo; echo "$(date -u +%T) start_profile returned"; exit 0
  fi
  sleep 10
done; echo "never reached running>=$MIN"; exit 1
