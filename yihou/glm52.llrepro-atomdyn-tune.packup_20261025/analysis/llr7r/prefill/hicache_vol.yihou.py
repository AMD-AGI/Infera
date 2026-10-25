# HiCache backup/load volume from cpu_op shapes. Usage: hicache_vol.yihou.py <dir>
import json, sys, collections
for k in range(4):
    s=json.load(open(f'{sys.argv[1]}/ex2.pp{k}.json'))
    st=sorted(s['steps'],key=lambda z:z['t']); lo=st[0]['t']-5e5
    gaps=[(st[i-1]['t']+st[i-1]['d'],st[i]['t']) for i in range(1,len(st)) if st[i]['t']-(st[i-1]['t']+st[i-1]['d'])>3e6]
    hi=st[-1]['t']+st[-1]['d']
    acc=collections.defaultdict(lambda:[0,0,0])
    for h in s['hs']:
        if not(lo<=h['t']<=hi) or any(a<h['t']<b for a,b in gaps): continue
        a=h['a']; dims=a.get('Input Dims',[]); ci=a.get('Concrete Inputs',[])
        if 'lf_pf' in h['n']:   # backup all layers: [layers],[N,L,1,item],[idx],[idx]
            n=dims[2][0]; L=dims[1][1]; item=dims[1][3]; key=('backup D2H lf_pf item%d L%d'%(item,L))
            acc[key][0]+=1; acc[key][1]+=n; acc[key][2]+=n*L*item
        elif 'pf_lf' in h['n']: # load per layer: [N,L,1,item] host, [M,item] dev, [idx]
            n=dims[2][0]; item=dims[1][1]; key='load H2D per-layer pf_lf item%d'%item
            acc[key][0]+=1; acc[key][1]+=n; acc[key][2]+=n*item
    kt=collections.defaultdict(float)
    for g in s['G']:
        if g['k']=='hicache_kernel' and lo<=g['t']<=hi and not any(a<g['t']<b for a,b in gaps): kt[(g['x'],g['n'][:40])]+=g['d']/1e3
    print('PP%d'%k)
    for key,v in acc.items(): print('   %-40s calls %4d  index-rows %8d  bytes %.3f GB'%(key,*v[:2],v[2]/1e9))
    for key,v in kt.items(): print('   kernel %-60s %.1f ms'%(key,v))
