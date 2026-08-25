#!/usr/bin/env bash
# Decide whether the router Pod needs `hostNetwork: true` on THIS cluster.
#
# What is actually known
# ----------------------
# On tus1-p15 the agentic trace failed 44 of 448 requests against a router on the
# Pod network, all in later (long-context) turns, and 448/448 after the router
# moved to hostNetwork. That is the whole of the evidence. The MECHANISM was
# never established: the obvious MTU story does not survive checking, because on
# that cluster the host-to-Pod path runs over a 1450-MTU veth whose path MTU the
# kernel discovers correctly. Something about the CNI datapath under sustained
# long-body load was the difference, and "something" is as precise as it gets.
#
# So this script does not predict the problem from configuration. It tries to
# REPRODUCE it, and only a reproduction counts as an answer. Steps 2 and 3
# gather context that helps explain a failure once you have one; they never
# decide on their own. On a cluster where the recipe already works (a 1500-MTU
# MI300X cluster did), expect this to come back clean, and change nothing.
#
# Run it from wherever your benchmark client actually runs. The vantage point is
# most of the question.
#
# Usage:
#   check-router-hostnetwork.sh --router-url http://<ROUTER_POD_IP>:8000 [--pod-ip IP]
#
#   --router-url  a deployed router reached at its POD IP — not a Service VIP,
#                 not the node IP. Without this the script cannot conclude.
#   --pod-ip      any Pod IP, for the path-MTU probe. Defaults to the router's.
#
# Exit: 0 = Pod network held up, 2 = reproduce -> use hostNetwork, 1 = no verdict.

set -uo pipefail

POD_IP=""; ROUTER_URL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pod-ip)     POD_IP="$2"; shift 2 ;;
    --router-url) ROUTER_URL="$2"; shift 2 ;;
    -h|--help)    sed -n '2,31p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done
[[ -z "$POD_IP" && -n "$ROUTER_URL" ]] && POD_IP=$(sed -E 's#^https?://##; s#:.*##' <<<"$ROUTER_URL")

say()   { printf '%s\n' "$*"; }
head2() { printf '\n== %s\n' "$*"; }

REPRO=0        # 1 = a test actually reproduced a failure
CONCLUSIVE=0   # 1 = a test actually moved traffic

# ---------------------------------------------------------------- 1. vantage
head2 "1. Where this client runs"
if [[ -n "${KUBERNETES_SERVICE_HOST:-}" && -f /var/run/secrets/kubernetes.io/serviceaccount/namespace ]]; then
  say "   Inside a Pod: client and router share the overlay, so none of the"
  say "   host-to-Pod edge applies to your path."
  say ""
  say "VERDICT: hostNetwork not needed for this client."
  exit 0
fi
say "   Host netns. Traffic to a Pod IP crosses the CNI edge — the case where"
say "   the tus1-p15 failures appeared."

# ------------------------------------------------------- 2. context: the path
head2 "2. Context: MTU along the path (informational)"
say "   Reported because it explains SOME failures, not because a mismatch here"
say "   proves anything. Do not act on this section alone."

POD_MTU=""
say "   Pod veths in this netns (a veth pair shares its MTU, so this is the"
say "   MTU inside those pods) — one line per distinct value:"
while read -r n m; do
  say "     $n mtu=$m"
  [[ -z "$POD_MTU" ]] && POD_MTU="$m"
done < <(ip -o link show 2>/dev/null \
         | grep -oP '^\d+: \K(cali\w+|veth\w+|lxc\w+)(?=@)' \
         | while read -r i; do echo "$i $(cat "/sys/class/net/$i/mtu" 2>/dev/null)"; done \
         | sort -u -k2,2 | head -5)
[[ -z "$POD_MTU" ]] && say "     (none — no pods on this node)"

say "   Overlay devices:"
while read -r i; do
  say "     $i mtu=$(cat "/sys/class/net/$i/mtu" 2>/dev/null)"
done < <(ip -o link show 2>/dev/null \
         | grep -oP '^\d+: \K(flannel\.\d+|vxlan\.calico|cilium_vxlan|cni\d+)(?=[:@])')

if [[ -n "$POD_IP" ]]; then
  dev=$(ip route get "$POD_IP" 2>/dev/null | grep -oP 'dev \K\S+' | head -1)
  if [[ -n "$dev" ]]; then
    dmtu=$(cat "/sys/class/net/$dev/mtu" 2>/dev/null)
    say "   Route to $POD_IP: dev $dev mtu=$dmtu"
    case "$dev" in
      cali*|veth*|lxc*|cni*) say "     -> that pod is LOCAL to this node; no overlay encap on the path." ;;
      *)                     say "     -> that pod is REMOTE; the overlay is on the path." ;;
    esac
  fi
fi

# ------------------------------------------- 3. context: conntrack + PMTU
head2 "3. Context: conntrack headroom and path MTU (informational)"
if [[ -r /proc/sys/net/netfilter/nf_conntrack_count ]]; then
  cc=$(cat /proc/sys/net/netfilter/nf_conntrack_count)
  cm=$(cat /proc/sys/net/netfilter/nf_conntrack_max 2>/dev/null || echo 0)
  say "   conntrack: $cc / $cm in use"
  if (( cm > 0 && cc * 10 > cm * 8 )); then
    say "   NOTE: above 80%. Pod-network traffic is conntracked; hostNetwork"
    say "   traffic largely is not, which is one way the two arms can differ."
  fi
else
  say "   conntrack: counters not readable here."
fi
if dmesg 2>/dev/null | grep -q "nf_conntrack: table full"; then
  say "   dmesg reports conntrack table full — a strong candidate cause."
fi

if [[ -n "$POD_IP" ]]; then
  PMTU=0
  for probe in 1400 1450 4000 8000 8900 8972; do
    ping -c1 -W2 -M do -s $((probe - 28)) "$POD_IP" >/dev/null 2>&1 && PMTU=$probe || break
  done
  if (( PMTU == 0 )); then
    say "   DF ping to $POD_IP: no reply even when small (ICMP likely filtered)."
  else
    say "   DF ping to $POD_IP: largest packet through = $PMTU bytes."
    rdev=$(ip route get "$POD_IP" 2>/dev/null | grep -oP 'dev \K\S+' | head -1)
    rmtu=$(cat "/sys/class/net/$rdev/mtu" 2>/dev/null || echo 0)
    if (( rmtu > 0 && rmtu > PMTU + 100 )); then
      say "   The route claims $rmtu but only $PMTU survives: a real black hole."
    else
      say "   Consistent with the route's own MTU — PMTU discovery is working."
    fi
  fi
fi

# ------------------------------------------- 4. the only step that concludes
head2 "4. Reproduction: sustained large bodies at the router's Pod IP"
if [[ -z "$ROUTER_URL" ]]; then
  say "   SKIPPED — and without it this script has no verdict. Re-run with"
  say "   --router-url http://<ROUTER_POD_IP>:8000 once the router is up."
else
  # Deliberately malformed JSON padded to size: the server has to read the whole
  # Content-Length before it can fail to parse, which is the transfer we want.
  # Any HTTP status means the body landed, and that is a pass — we are testing
  # the network, not the API.
  tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
  post_once() {
    curl -sS -o /dev/null -w '%{http_code}' --max-time 30 \
         -H 'Content-Type: application/json' \
         --data-binary @"$tmp" "$ROUTER_URL/v1/chat/completions" 2>/dev/null || echo 000
  }

  say "   a) single body, increasing size"
  for kb in 16 64 256 512 1024; do
    python3 -c "import sys; sys.stdout.write('{\"pad\":\"' + 'A'*($kb*1024))" > "$tmp"
    code=$(post_once); CONCLUSIVE=1
    if [[ "$code" == "000" ]]; then
      say "      ${kb} KB: NO RESPONSE — connection opened, payload never completed."
      REPRO=1; break
    fi
    say "      ${kb} KB: HTTP $code (delivered)"
  done

  # One-at-a-time rarely failed on tus1-p15 either; the failures showed up under
  # sustained concurrency, which is what actually stresses the datapath.
  if (( REPRO == 0 )); then
    say "   b) 16 concurrent 300 KB bodies x 3 rounds (the shape that failed)"
    python3 -c "import sys; sys.stdout.write('{\"pad\":\"' + 'A'*(300*1024))" > "$tmp"
    fails=0
    codes=$(mktemp)
    for _ in 1 2 3; do
      : > "$codes"
      for _ in $(seq 16); do post_once >> "$codes" & done
      wait
      n=$(grep -c '^000$' "$codes"); n=${n:-0}
      fails=$(( fails + n ))
    done
    rm -f "$codes"
    CONCLUSIVE=1
    if (( fails > 0 )); then
      say "      $fails of 48 requests got no response. Reproduced."
      REPRO=1
    else
      say "      48 of 48 completed."
    fi
  fi
fi

# ---------------------------------------------------------------- 5. verdict
head2 "5. Verdict"
if (( REPRO == 1 )); then
  say "   Reproduced a failure the client cannot distinguish from a dead router."
  say "   Set hostNetwork: true (plus dnsPolicy: ClusterFirstWithHostNet) on the"
  say "   router service in your deploy.yaml and re-run this script to confirm."
  say "   It costs port 8000 on that node and puts the router outside"
  say "   NetworkPolicy, so if you own the CNI, finding the real cause with"
  say "   sections 2-3 in hand is the better outcome."
  exit 2
fi
if (( CONCLUSIVE == 0 )); then
  say "   No verdict: nothing above moved traffic. Re-run with --router-url."
  exit 1
fi
say "   The Pod network carried every large body under concurrency. Leave the"
say "   shipped defaults alone — this cluster does not need hostNetwork."
exit 0
