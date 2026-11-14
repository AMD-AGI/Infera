#!/bin/bash
# DRIVER619-20261113: is the 6.19 node's RDMA itself slower? Host-memory ib_write_bw (1 MiB, 5 s, GID index 1) on each rail
# ionic_0..7, n04-29 <-> n04-33 both directions; reference rdma-bw-n04-33_n10-29.txt (392 Gb/s per rail, 11-10).
# srun gets -w and --exclude both: a bare -w <node> landed on the other node twice on 11-13.
OUT=$K/out/rdma-bw-cross-$(date -u +%m%dT%H%M)
mkdir -p "$OUT"
ip_of() { getent hosts "$1" | awk '{print $1}' | head -1; }
on() { local n=$1; shift; local x=$NODE616; [ "$n" = "$NODE616" ] && x=$NODE619; srun --jobid=$JOB --overlap -N1 -n1 -w $n --exclude=$x "$@"; }
port=18515
for dir in "$NODE619 $NODE616" "$NODE616 $NODE619"; do
    read -r s c <<<"$dir"
    sip=$(on $s bash -c "ip -4 -o addr show dev fenic | awk '{print \$4}' | cut -d/ -f1")
    for i in 0 1 2 3 4 5 6 7; do
        port=$((port + 1))
        on $s timeout 30 ib_write_bw -d ionic_$i -x 1 -s 1048576 -D 5 --report_gbits -F -p $port > "$OUT/srv-$s-ionic_$i.log" 2>&1 &
        sleep 2
        r=$(on $c timeout 30 ib_write_bw -d ionic_$i -x 1 -s 1048576 -D 5 --report_gbits -F -p $port $sip 2>&1 | awk '$1 == 1048576 {print $4}')
        wait
        echo "ionic_$i ${c%%.*} -> ${s%%.*} (server): ${r:-failed} Gb/s" | tee -a "$OUT/summary.txt"
    done
done
