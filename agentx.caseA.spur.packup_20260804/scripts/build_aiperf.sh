#!/bin/bash
# Build the aiperf-agentx image on a held spur node. FOREGROUND inside spur exec
# on purpose: a backgrounded docker client dies at exec-namespace teardown.
set -eu
JOB="${1:?job}"
W=/shared_nfs/yihou_agentx_caseA
spur exec "$JOB" bash -c "
export DOCKER_CONFIG=/tmp/dockercfg; mkdir -p \$DOCKER_CONFIG
H=\$(hostname)
LOG=$W/imgbuild/build_\$H.log
docker build -t aiperf-agentx:v1.0 -f $W/imgbuild/Dockerfile $W/imgbuild > \"\$LOG\" 2>&1
rc=\$?
echo \"build rc=\$rc\" >> \"\$LOG\"
if [ \$rc -eq 0 ]; then echo ok > $W/imgbuild/status_\$H; else echo fail > $W/imgbuild/status_\$H; tail -20 \"\$LOG\"; fi
"
