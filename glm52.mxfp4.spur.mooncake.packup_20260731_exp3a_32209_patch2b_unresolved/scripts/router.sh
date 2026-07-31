#!/bin/bash
# Start the PD router for one arm, on that arm's PREFILL node.
#
# Each arm gets its own --port and --prometheus-port. A router whose circuit
# breaker is still open from a previous run returns 503 in ~0.4 s, which looks
# exactly like a persisting server failure (PITFALLS P4) -- distinct ports per
# arm removes that confusion entirely.
#
# Usage: router.sh <arm e1|e2|e3>
set -eu
ARM="${1:?e1|e2|e3}"
export DOCKER_CONFIG=/tmp/dockercfg
W=/shared_nfs/yihou_exp3way

case "$ARM" in
e1) PJOB=14315; PIP=10.245.159.138; DIP=10.245.157.171; RP=8110; PP=29110 ;;
e2) PJOB=14317; PIP=10.245.152.243; DIP=10.245.155.111; RP=8120; PP=29120 ;;
e3) PJOB=14320; PIP=10.245.154.156; DIP=10.245.144.119; RP=8130; PP=29130 ;;
e3a) PJOB=14915; PIP=10.245.158.72; DIP=10.245.147.58; RP=8141; PP=29141 ;;
e3c) PJOB=17443; PIP=10.245.152.84; DIP=10.245.151.183; RP=8144; PP=29144 ;;
e3b) PJOB=14317; PIP=10.245.152.243; DIP=10.245.155.111; RP=8150; PP=29150 ;;
*)  echo "unknown arm $ARM" >&2; exit 1 ;;
esac

echo "router arm=$ARM on $PIP:$RP  (prefill $PIP:30000 / decode $DIP:30001)"
spur exec "$PJOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
docker exec dbg2 bash -c 'pkill -f sglang_router.launch_router 2>/dev/null; true'
sleep 2
docker exec -d dbg2 bash -c 'python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://$PIP:30000 8998 \
  --decode  http://$DIP:30001 \
  --host 0.0.0.0 --port $RP --prometheus-port $PP \
  > $W/$ARM/router.log 2>&1'
" 2>&1 | grep -v libtinfow
echo "  -> $W/$ARM/router.log"
