# What PP(k-1) main-thread event ends when PP(k) recv_requests ends? Usage: v3.yihou.py <dir>
import json, sys, collections
D=sys.argv[1]; S=[json.load(open(f'{D}/ex2.pp{i}.json')) for i in range(4)]
print('PM names', sorted(set(p['n'] for p in S[2]['PM'])))
for k in (1,3):
    rr=sorted([p for p in S[k]['PM'] if p['n']=='recv_requests' and p['d']>15e3],key=lambda p:p['t'])
    for p in rr[:10]:
        e=p['t']+p['d']
        near=[(q['n'],round((q['t']+q['d']-e)/1e3,2),round(q['d']/1e3,1)) for q in S[k-1]['PM'] if abs(q['t']+q['d']-e)<3e3 or abs(q['t']-e)<3e3]
        nc=[(g['x'],round((g['t']+g['d']-e)/1e3,2),round(g['d']/1e3,1)) for g in S[k-1]['G'] if g['k']=='nccl' and abs(g['t']+g['d']-e)<3e3]
        print('PP%d recv_requests dur %.1f: PP%d events ending/starting within 3ms: %s | nccl ending: %s'%(k,p['d']/1e3,k-1,near[:8],nc[:4]))
