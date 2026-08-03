#!/bin/bash
strings /shared_nfs/yihou_agbench_mtp/logs/armB_decode.log | grep -oE 'accept len: [0-9.]+' | awk '{print $3}' | python3 -c "
import sys
v=[float(x) for x in sys.stdin]
v.sort()
if not v: print('no accept len lines'); raise SystemExit
n=len(v)
P=lambda p: v[min(n-1,int(n*p/100))]
print(f'n={n} mean={sum(v)/n:.3f} min={v[0]:.2f} p25={P(25):.2f} p50={P(50):.2f} p75={P(75):.2f} p90={P(90):.2f} max={v[-1]:.2f}')
print(f'exactly 4.00: {sum(1 for x in v if x>=3.995)} ({100*sum(1 for x in v if x>=3.995)/n:.1f}%)')
"
