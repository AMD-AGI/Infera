#!/usr/bin/env bash
# Positive evidence for each feature, INDEPENDENT of engine/smoke.sh.
# Runs on the JUMP HOST. Each check is one that would go red if the feature were
# silently absent.
set -u
P=chi2835; D=chi2879; PIP=10.2.122.78; DIP=10.2.122.10
CTR=glm52_pd
PLOG=/tmp/glm52_prefill.log; DLOG=/tmp/glm52_decode.log
J(){ timeout 90 ssh -o StrictHostKeyChecking=no "$1" "${@:2}"; }

echo "########## 1. PD pairing (router, never a leg's own port) ##########"
J $P "docker exec $CTR curl -s -m10 http://$PIP:8100/v1/workers" \
  | python3 -c "import sys,json
for w in json.load(sys.stdin)['workers']:
    print(f\"  {w['disagg_mode']:8s} {w['url']:26s} status={w['status']} dp_size={w['dp_size']}\")"

echo
echo "########## 2. prefill DPA OFF / decode DPA ON — LIVE cmdline, not the log ##########"
for pair in "$P:prefill" "$D:decode"; do
  h="${pair%%:*}"; role="${pair#*:}"
  echo "  -- $role on $h --"
  J $h "docker exec $CTR ps -eo args | grep '[l]aunch_server' | head -1" \
    | tr ' ' '\n' | grep -E '^--(dp-size|enable-dp-attention|ep-size|chunked-prefill-size|mem-fraction-static|speculative-algorithm|disable-custom-all-reduce)$|^(8|0\.[0-9]+|65536|EAGLE)$' \
    | paste -sd' ' | sed 's/^/     flags: /'
  echo -n "     scheduler_DP ranks: "
  J $h "docker exec $CTR ps -eo args | grep -oE 'scheduler_DP[0-9]+' | sort -u | wc -l"
done

echo
echo "########## 3. mooncake over RDMA (both must be 0) ##########"
for pair in "$P:$PLOG" "$D:$DLOG"; do
  h="${pair%%:*}"; f="${pair#*:}"
  echo -n "  $h MC_FORCE_TCP="
  J $h "docker exec $CTR bash -c 'strings $f | grep -c MC_FORCE_TCP || true'" | tr -d '\r'
  echo -n "  $h GID-is-NULL="
  J $h "docker exec $CTR bash -c 'strings $f | grep -c \"GID is NULL\" || true'" | tr -d '\r'
  echo -n "  $h in-container PORT_ACTIVE="
  J $h "docker exec $CTR bash -c 'ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE || echo 0'" | tr -d '\r'
done

echo
echo "########## 4. KV actually moved: transfer counters on the decode leg ##########"
J $D "docker exec $CTR bash -c 'strings $DLOG | grep -oE \"#transfer-req: [0-9]+\" | sort -u | tail -3; strings $DLOG | grep -icE \"KVTransferError|Traceback\" || true'"

echo
echo "########## 5. MTP accept-len distribution (healthy 2-3; a steady 4.00 is BAD) ##########"
J $D "docker exec $CTR bash -c 'strings $DLOG | grep -oE \"accept len: [0-9.]+\" | awk \"{print \\\$3}\" | sort -n | uniq -c | tail -12'"

echo
echo "########## 6. kvd — adapters and counters ##########"
echo -n "  prefill kvd adapters connected: "
J $P "docker exec $CTR bash -c 'strings $PLOG | grep -c \"kvd adapter connected\" || echo 0'" | tr -d '\r'
J $P "docker exec $CTR python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock"
echo -n "  decode kvd adapters (expect 0 by design): "
J $D "docker exec $CTR bash -c 'strings $DLOG | grep -c \"kvd adapter connected\" || echo 0'" | tr -d '\r'

echo
echo "########## 7. kv-aware router — policy + kv-event subscriptions ##########"
# The Rust router prints `router_policy: \"kv-aware\"` inside a Config{...} dump and
# colourises it, so smoke.sh's `grep -o 'router-policy=...'` never matches. Strip ANSI.
J $P "docker exec $CTR bash -c \"sed -r 's/\\x1B\\[[0-9;]*[mK]//g' /tmp/router.log | grep -oE 'router_policy: \\\"[a-z-]+\\\"' | head -1\""
J $P "docker exec $CTR bash -c \"sed -r 's/\\x1B\\[[0-9;]*[mK]//g' /tmp/router.log | grep -E 'kv events: subscribing|loaded tokenizer' | tail -4\""
