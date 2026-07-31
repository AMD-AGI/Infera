#!/bin/bash
# PD router for the FINAL validation run, on the prefill node, in the `final`
# container. Fresh --port / --prometheus-port: a router whose circuit breaker is
# still open from an earlier run returns 503 in ~0.4 s, which reads exactly like
# a persisting backend failure. Distinct ports remove that confusion.
set -eu
export DOCKER_CONFIG=/tmp/dockercfg
W=/shared_nfs/yihou_exp3way
PJOB=17443; PIP=10.245.152.84; DIP=10.245.151.183; RP=8180; PP=29180

echo "router FINAL on $PIP:$RP  (prefill $PIP:30000 / decode $DIP:30001)"
spur exec "$PJOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
docker exec conv bash -c 'pkill -f sglang_router.launch_router 2>/dev/null; true'
sleep 2
docker exec -d conv bash -c 'python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://$PIP:30000 8998 \
  --decode  http://$DIP:30001 \
  --host 0.0.0.0 --port $RP --prometheus-port $PP \
  > $W/conv/router.log 2>&1'
" 2>&1 | grep -v libtinfow
echo "  -> $W/conv/router.log   endpoint http://$PIP:$RP"
