#!/bin/bash
pkill -9 -f "infera.server" 2>/dev/null; sleep 3
exec python3 -m infera.server --host 0.0.0.0 --port 8100 \
  --discovery-backend etcd --etcd-endpoint 10.2.122.10:2379 \
  --request-transport http --kv-event-transport zmq \
  --router-policy round-robin --router-tokenizer-path /mnt/vast/xiaobo/models/GLM-5.2-MXFP4 \
  > /tmp/router.log 2>&1
