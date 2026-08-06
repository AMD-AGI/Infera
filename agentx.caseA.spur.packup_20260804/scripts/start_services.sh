#!/bin/bash
# etcd (prefill node only) + the infera-kvd daemon (both nodes).
#
# Each node's engine talks to its OWN local kvd socket -- these are two separate
# stores, not two views of one. Read the counters per node accordingly.
#
# Usage: start_services.sh <job> <prefill|decode> <my-ip>
set -eu
JOB="${1:?job}"
ROLE="${2:?prefill|decode}"
MY_IP="${3:?ens3 ip}"
CTR="${CTR:-agbench_mtp}"
KVD_SOCK=/tmp/kvd/kvd.sock

# --max-bytes is an ABSOLUTE cap, deliberately. sglang's --hicache-ratio default
# sizes the host pool off max_total_num_tokens and computed to 355 GB per DP rank
# once; a TB-scale pinned host allocation can wedge a spur node at kernel level.
spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
set -e
if [ '$ROLE' = prefill ]; then
  echo '===== etcd ====='
  docker rm -f ${CTR}_etcd >/dev/null 2>&1 || true
  docker run -d --name ${CTR}_etcd --network=host quay.io/coreos/etcd:v3.5.14 etcd \
    --advertise-client-urls http://$MY_IP:2379 --listen-client-urls http://0.0.0.0:2379 >/dev/null
  sleep 8
  curl -sf -m5 http://$MY_IP:2379/version >/dev/null && echo '  etcd up' || echo '  ETCD FAILED'
fi

echo '===== kvd daemon ====='
# Staged script file, NOT 'docker exec -d ... bash -lc'. The detached login-shell
# form exits and takes the child with it -- no process, no log, no error.
docker exec $CTR bash -c \"mkdir -p /tmp/kvd /tmp/kvd-long && rm -f $KVD_SOCK && printf '%s\n' '#!/bin/bash' 'exec python3 -m infera.kvd --socket $KVD_SOCK --max-bytes 64G --long-path /tmp/kvd-long --long-bytes 512G --log-level INFO' > /run_kvd.sh && chmod +x /run_kvd.sh\"
docker exec -d $CTR bash -c 'nohup /run_kvd.sh > /tmp/kvd.log 2>&1'
sleep 20
docker exec $CTR bash -c \"test -S $KVD_SOCK && echo '  kvd socket OK' || { echo '  KVD FAILED'; tail -20 /tmp/kvd.log; }\"
docker exec $CTR python3 -m infera.kvd.statctl --socket $KVD_SOCK 2>&1 | head -3
" 2>&1 | grep -v libtinfow
