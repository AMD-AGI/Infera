#!/usr/bin/env bash
set -Eeuo pipefail
root=/perf_apps/liyingli/bench_agentx/r1r4-4k-20260924
role=$1
set -a; source "${CONFIG:?}"; set +a
name=$CONTAINER_PREFIX-watchdog-$role
docker run -d --init --name "$name" --network host --user 100078:100078 --group-add "$(stat -c %g /var/run/docker.sock)" --entrypoint python3 \
 -v "$root:$root" -v /opt/slurm:/opt/slurm:ro -v /run/munge:/run/munge:ro \
 -v /usr/lib/x86_64-linux-gnu/libmunge.so.2:/usr/lib/x86_64-linux-gnu/libmunge.so.2:ro \
 -v /var/run/docker.sock:/var/run/docker.sock -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
 -e "SLURM_CONF=$root/config/slurm-client-dccs.conf" \
 infera-sglang:aus-0922-reqtrace "$root/scripts/container_guard.py" --root "$root" --role "$role"
