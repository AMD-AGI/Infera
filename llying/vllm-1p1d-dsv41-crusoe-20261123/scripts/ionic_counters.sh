#!/bin/bash
# Snapshot of the ionic RDMA / port counters that showed NIC-internal drops and ACK timeouts on crsuse2 before
# (llying/cross-rank-rca-20260921): one line per non-zero counter, "<dev> <counter> <value>".
for d in /sys/class/infiniband/ionic_*; do
    n=$(basename $d)
    for f in $d/ports/1/hw_counters/* $d/ports/1/counters/*; do
        [ -f "$f" ] || continue
        v=$(cat "$f" 2>/dev/null) || continue
        [ "$v" != 0 ] && echo "$n $(basename $f) $v"
    done
done
