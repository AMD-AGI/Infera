# Verify: is PP send kernel duration transfer time or wait for receiver's recv? Usage: sendmatch.yihou.py <dir>
import json, sys
D=sys.argv[1]; S=[json.load(open(f'{D}/ex2.pp{i}.json')) for i in range(4)]
def batches(s):
    st=s['steps']; bl=[]; i=0
    while i<len(st):
        e=st[i]; dr=None
        if i+1<len(st) and 'DRAFT' in st[i+1]['n'] and st[i+1]['t']-e['t']<1e6: dr=st[i+1]; i+=1
        bl.append((e['t'],max(e['t']+e['d'],dr['t']+dr['d']) if dr else e['t']+e['d'])); i+=1
    return bl
B=[batches(s) for s in S]
for j in [1,2,7,8,9,10,11,12,13,14,15]:
    for k in range(3):
        end=B[k][j][1]; nxt=B[k+1][j][0]
        snd=sorted([g for g in S[k]['G'] if g['k']=='nccl' and g['x']=='pp_send' and g['d']>500 and end-5e3<=g['t']<=nxt],key=lambda g:g['t'])
        rcv=sorted([g for g in S[k+1]['G'] if g['k']=='nccl' and g['d']>500 and end-120e3<=g['t']<=nxt+1e3],key=lambda g:g['t'])
        cp=[p for p in S[k+1]['PM'] if p['n']=='_pp_recv_proxy_tensors' and end-5e3<=p['t']<=nxt]
        T=end
        s0=snd[0] if snd else None
        print('b%02d PP%d->PP%d sendEnd-of-step %.1f nextStart %.1f | sends %s | recvs on next %s | next cpu recv_proxy start %s'%(j,k,k+1,0.0,(nxt-T)/1e3,
          [(round((g['t']-T)/1e3,1),round(g['d']/1e3,1)) for g in snd[:3]],
          [(g['x'],round((g['t']-T)/1e3,1),round(g['d']/1e3,1)) for g in rcv[:8]],
          [round((p['t']-T)/1e3,1) for p in cp]))
# tiny sends at step end: count and duration on PP0
import statistics
for k in range(4):
    for j in [10,13]:
        end=B[k][j][1]
        t=[g for g in S[k]['G'] if g['k']=='nccl' and abs(g['t']-end)<1e3]
        print('PP%d b%d nccl kernels within 1ms of step end:'%(k,j),[(g['x'],g['s'],round((g['t']-end)/1e3,2),g['d']) for g in t])
