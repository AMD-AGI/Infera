#!/bin/bash
# One line per ionic NIC: name tx_rdma_ucast_bytes rx_rdma_ucast_bytes tx_rdma_ack_timeout tx_rdma_retx_pkts (cumulative).
for d in /sys/class/infiniband/ionic_*; do
    c=$d/ports/1/hw_counters
    echo "$(basename $d) $(cat $c/tx_rdma_ucast_bytes 2>/dev/null || echo 0) $(cat $c/rx_rdma_ucast_bytes 2>/dev/null || echo 0) $(cat $c/tx_rdma_ack_timeout 2>/dev/null || echo 0) $(cat $c/tx_rdma_retx_pkts 2>/dev/null || echo 0)"
done
