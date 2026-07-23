source /mnt/vast/c_huggingface/pd_env.sh
URL="http://$PREFILL_IP:$ROUTER_PORT"
echo "=== workers ==="
curl -s "$URL/v1/workers" 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);[print(w['disagg_mode'],w['worker_id'],w['status']) for w in d['workers']]" 2>/dev/null
echo "=== smoke: capital ==="
curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$INFERA_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"capital of France, one word\"}],\"max_tokens\":16,\"temperature\":0}" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('REPLY:',d['choices'][0]['message']['content'] if 'choices' in d else d)" 2>&1
echo "=== smoke: counting ==="
curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$INFERA_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count 1 to 10.\"}],\"max_tokens\":48,\"temperature\":0}" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('REPLY:',d['choices'][0]['message']['content'] if 'choices' in d else d)" 2>&1
