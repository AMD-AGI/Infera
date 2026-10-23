#!/usr/bin/env bash
# rdmactr (task dcpx 2026-10-22): one-line-per-rail snapshot of ionic RDMA hw counters (read-only sysfs).
# Usage: rdmactr.yihou.sh [label]   -> "<utc> <host> <label> <dev> k=v ..."
K="rx_rdma_ucast_bytes tx_rdma_ucast_bytes resp_rx_outouf_seq resp_rx_dup_request req_rx_pkt_seq_err tx_rdma_retx_pkts tx_rdma_ack_timeout req_tx_retry_excd_err rx_rdma_ecn_pkts rx_rdma_cnp_pkts tx_rdma_cnp_pkts"
for d in ionic_0 ionic_1 ionic_2 ionic_3; do
  l="$(date -u +%FT%TZ) $(hostname -s) ${1:-} $d"
  for k in $K; do l+=" $k=$(cat /sys/class/infiniband/$d/ports/1/hw_counters/$k 2>/dev/null)"; done
  echo "$l"
done
