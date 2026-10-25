#!/usr/bin/env bash
# node_probe1s (task llrepro 2026-10-24, ON a node): read-only. Usage: node_probe1s.yihou.sh <outprefix> <seconds>
# 1 s samples of ionic_0..7 tx/rx RDMA bytes + retx/ack-timeout (.rdma1s), per-core CPU over the window (.cpu),
# rocm-smi every 10 s (.gpu), top -H (.topH).
o=$1; s=${2:-120}; h=$(hostname -s)
cpu(){ awk '/^cpu[0-9]/{print $1, $2+$3+$4+$7+$8, $2+$3+$4+$5+$6+$7+$8}' /proc/stat; }
c0=$(cpu)
( for i in $(seq 1 $((s/10))); do echo "== $(date -u +%T)"; rocm-smi --showuse --showpower --json 2>/dev/null; echo; sleep 10; done ) > $o.$h.gpu &
for i in $(seq 1 $s); do
  t=$(date -u +%s.%N)
  for d in ionic_0 ionic_1 ionic_2 ionic_3 ionic_4 ionic_5 ionic_6 ionic_7; do
    p=/sys/class/infiniband/$d/ports/1/hw_counters
    echo "$t $d $(cat $p/tx_rdma_ucast_bytes $p/rx_rdma_ucast_bytes $p/tx_rdma_retx_pkts $p/tx_rdma_ack_timeout 2>/dev/null | tr '\n' ' ')"
  done
  sleep 1
done > $o.$h.rdma1s
c1=$(cpu)
join <(echo "$c0") <(echo "$c1") | awk '{b=($4-$2)/(($5-$3)+1e-9)*100; print $1, b}' > $o.$h.cpu
top -H -b -n 2 -d 5 | awk '/^top -/{k++} k==2 && $9+0>30 {print $1, $9, $12}' | head -40 > $o.$h.topH
wait; echo done > $o.$h.done
