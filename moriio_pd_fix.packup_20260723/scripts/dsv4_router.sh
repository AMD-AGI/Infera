#!/bin/bash
set -u
IP=10.2.122.10; PREFIX=/infera/pd-dsv4/
MODEL=/mnt/vast/d_huggingface/models/DeepSeek-V4-Pro-fixed
LOG=/mnt/vast/c_huggingface/vllm_patch_verify/dsv4_router.log
docker exec glm_pd bash -lc "pkill -9 -f infera.server 2>/dev/null; sleep 1; : > $LOG; true"
docker exec glm_pd bash -c "cd /opt/infera && nohup python3 -m infera.server --host 0.0.0.0 --port 8000 --discovery-backend etcd --etcd-endpoint $IP:2379 --etcd-prefix $PREFIX --request-transport http --router-policy round-robin --router-tokenizer-path $MODEL > $LOG 2>&1 & echo launched pid=\$!"
sleep 8
echo "=== health ==="; docker exec glm_pd bash -lc "curl -s -o /dev/null -w '%{http_code}\n' http://$IP:8000/health"
echo "=== log tail ==="; tail -8 "$LOG" 2>/dev/null
