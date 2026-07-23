source /mnt/vast/c_huggingface/pd_env.sh
URL="http://$PREFILL_IP:$ROUTER_PORT"
echo "=== workers registered ==="
curl -s "$URL/v1/workers" 2>/dev/null | python3 -m json.tool 2>/dev/null | head -40 || curl -s "$URL/v1/workers"
echo ""
echo "=== smoke completion (KV must transfer P->D over mooncake RDMA) ==="
curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$INFERA_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one word.\"}],\"max_tokens\":16,\"temperature\":0}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('REPLY:', d['choices'][0]['message']['content'])" 2>&1
echo ""
echo "=== second: counting ==="
curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$INFERA_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 10.\"}],\"max_tokens\":48,\"temperature\":0}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('REPLY:', d['choices'][0]['message']['content'])" 2>&1
