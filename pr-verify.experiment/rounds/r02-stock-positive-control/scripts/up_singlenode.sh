#!/usr/bin/env bash
# what: bring up a GLM-5.2 1P1D deployment with BOTH legs on this one node.
# why : the 2-node path is blocked (n01-33 has no rail route to us), and the #33970 race
#       is thread-vs-stream, not host-vs-host — proven in r01 by a mooncake loopback MVP.
# how : two containers, disjoint GPUs (0-3 / 4-7), distinct rail IPs and HCAs, so the two
#       legs talk over real RDMA verbs exactly as they would across a rack.
#
# ARM=stock   -> `git checkout` the three PR-B files in both containers (unpatched)
# ARM=patched -> leave the image as built (patch applied)
# The image's sglang is a git checkout, so the arms differ by exactly those three files.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT="/home/yihou/dev/git.16-19/infera.patch.record.system/examples/sglang_1p1d_glm5.2"

ARM="${ARM:?ARM=stock|patched}"
export INFERA_IMAGE="${INFERA_IMAGE:-infera-local:sglang-prverify-20260819}"
export MODEL_MOUNT="${MODEL_MOUNT:-/data/models}"
export MODEL="${MODEL:-/data/models/GLM-5.2-MXFP4}"
export TOKENIZER="${TOKENIZER:-$MODEL}"
export SERVED="${SERVED:-glm5.2-mxfp4}"

# Two legs on one box: distinct rail IPs so each leg binds its own NIC and HCA, mirroring
# a real 2-node split. Verified in r01: cross-HCA RDMA loopback runs at 348 Gb/s.
P_IP="${P_IP:-192.168.1.14}"; P_IB="${P_IB:-ionic_0}"; P_CTR="${P_CTR:-glm52_p}"
D_IP="${D_IP:-192.168.2.14}"; D_IB="${D_IB:-ionic_1}"; D_CTR="${D_CTR:-glm52_d}"
export ETCD_IP="$P_IP"

# r01: this fabric exposes only gid[0] (link-local) and gid[1] (RoCEv2 IPv4). Index 3
# does not exist and makes mooncake fail with "GID is NULL ... No available RNIC".
# NOTE ib_write_bw -x 3 works anyway — perftest indexes GIDs differently, so a passing
# perftest run does NOT validate this value.
export MC_GID_INDEX="${MC_GID_INDEX:-1}"
export RDMAV_FORK_SAFE=1

export TP="${TP:-4}"
export CHUNK="${CHUNK:-131072}"
export CTX="${CTX:-262144}"
# KV events and kvd are orthogonal to the KV-transfer race and each would add a
# host-port collision between the two legs. Off, so the run has fewer moving parts.
export KVAWARE=0 KVD=0
export ROUTER_POLICY="${ROUTER_POLICY:-round-robin}"
export DPA="${DPA:-1}"

source "$KIT/common.sh"

log "ARM=$ARM  tp=$TP chunk=$CHUNK dpa=$DPA  P=$P_IP/$P_IB(gpu0-3)  D=$D_IP/$D_IB(gpu4-7)"

# ---- containers -----------------------------------------------------------------------
for spec in "$P_CTR" "$D_CTR"; do
  CTR="$spec" start_container || die "container $spec failed"
done

# ---- arm selection --------------------------------------------------------------------
# One variable between the arms: the three files PR #33970 touches.
PR_FILES="python/sglang/srt/disaggregation/common/utils.py \
python/sglang/srt/disaggregation/mooncake/conn.py \
python/sglang/srt/disaggregation/prefill.py"
for c in "$P_CTR" "$D_CTR"; do
  if [ "$ARM" = "stock" ]; then
    docker exec "$c" bash -c "cd /sgl-workspace/sglang && git checkout -- $PR_FILES" \
      || die "could not revert PR-B files in $c"
  fi
  n=$(docker exec "$c" bash -c \
    "grep -c wait_event /sgl-workspace/sglang/python/sglang/srt/disaggregation/mooncake/conn.py" \
    | tr -d '\r\n')
  log "  $c: 'wait_event' occurs ${n}x in mooncake/conn.py (stock=0, patched=9)"
  case "$ARM:$n" in
    stock:0|patched:9) ;;
    *) die "$c is not in the '$ARM' arm (wait_event=${n}) — refusing to run a mislabelled experiment" ;;
  esac
done

# ---- etcd -----------------------------------------------------------------------------
start_etcd "$ETCD_IP" glm52-etcd || die "etcd failed"

# ---- legs -----------------------------------------------------------------------------
CTR="$P_CTR" ROLE=prefill MY_IP="$P_IP" RDMA_IB_DEVICES="$P_IB" \
  GPUS="0,1,2,3" PORT=30000 MTP=0 LOG=/tmp/glm52_prefill.log \
  bash "$KIT/engine/leg.sh" || die "prefill launch failed"

CTR="$D_CTR" ROLE=decode MY_IP="$D_IP" RDMA_IB_DEVICES="$D_IB" \
  GPUS="4,5,6,7" PORT=30001 MTP=0 LOG=/tmp/glm52_decode.log \
  bash "$KIT/engine/leg.sh" || die "decode launch failed"

log "both legs launching; weights are 408 GB from local disk — expect several minutes"
CTR="$P_CTR" wait_health "http://$P_IP:30000/health" 160 15 || die "prefill never served"
CTR="$D_CTR" wait_health "http://$D_IP:30001/health" 160 15 || die "decode never served"

# ---- router ---------------------------------------------------------------------------
CTR="$P_CTR" start_router "$P_IP" || die "router failed"

log "UP. router=http://$P_IP:$ROUTER_PORT  arm=$ARM"
