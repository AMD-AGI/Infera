#!/usr/bin/env bash
# node_probe (task llrepro 2026-10-24, ON a node): read-only. Usage: node_probe.yihou.sh <outprefix> <seconds>
# Writes <outprefix>.<host>.rdma (8 rails t0/t1), .cpu (per-core busy % over the window), .gpu (rocm-smi every 10 s), .topH.
o=$1; s=${2:-120}; h=$(hostname -s)
ctr(){ for d in ionic_0 ionic_1 ionic_2 ionic_3 ionic_4 ionic_5 ionic_6 ionic_7; do l="$(date -u +%s) $d"; for k in tx_rdma_ucast_bytes rx_rdma_ucast_bytes tx_rdma_retx_pkts tx_rdma_ack_timeout; do l+=" $k=$(cat /sys/class/infiniband/$d/ports/1/hw_counters/$k 2>/dev/null)"; done; echo "$l"; done; }
cpu(){ awk '/^cpu[0-9]/{print $1, $2+$3+$4+$7+$8, $2+$3+$4+$5+$6+$7+$8}' /proc/stat; }
ctr > $o.$h.rdma; c0=$(cpu)
for i in $(seq 1 $((s/10))); do echo "== $(date -u +%T)" >> $o.$h.gpu; rocm-smi --showuse --showpower --json 2>/dev/null >> $o.$h.gpu; echo >> $o.$h.gpu; sleep 10; done
ctr >> $o.$h.rdma; c1=$(cpu)
join <(echo "$c0") <(echo "$c1") | awk '{b=($4-$2)/(($5-$3)+1e-9)*100; print $1, b}' > $o.$h.cpu
top -H -b -n 2 -d 5 | awk '/^top -/{k++} k==2 && $9+0>30 {print $1, $9, $12}' | head -40 > $o.$h.topH
echo done > $o.$h.done
