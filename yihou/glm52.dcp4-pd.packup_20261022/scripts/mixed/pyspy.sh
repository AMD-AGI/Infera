#!/bin/bash
# pyspy.sh CONTAINER OUTFILE: py-spy dump of every sglang process in the container
C=$1; OUT=$2
docker exec $C bash -c 'which py-spy || pip install -q py-spy >/dev/null 2>&1; for p in $(pgrep -f "sglang::|launch_server"); do echo "=== PID $p $(cat /proc/$p/comm)"; py-spy dump --pid $p 2>&1 | head -60; done' > $OUT 2>&1
