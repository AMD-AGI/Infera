# Verify misc prefill claims from ex2 json. Usage: v2.yihou.py <dir>
import json, sys, collections
D=sys.argv[1]; S=[json.load(open(f'{D}/ex2.pp{i}.json')) for i in range(4)]
CLEAN=[1,2,7,8,9,10,11,12,13,14,15]
def union(iv):
    iv=sorted(iv); tot=0; cs=ce=None
    for a,b in iv:
        if b<=a: continue
        if cs is None: cs,ce=a,b
        elif a<=ce: ce=max(ce,b)
        else: tot+=ce-cs; cs,ce=a,b
    if cs is not None: tot+=ce-cs
    return tot
def batches(s):
    st=s['steps']; bl=[]; i=0
    while i<len(st):
        e=st[i]; dr=None
        if i+1<len(st) and 'DRAFT' in st[i+1]['n'] and st[i+1]['t']-e['t']<1e6: dr=st[i+1]; i+=1
        bl.append((e['t'],max(e['t']+e['d'],dr['t']+dr['d']) if dr else e['t']+e['d'])); i+=1
    return bl
B=[batches(s) for s in S]
for k in range(4):
    s=S[k]; bl=B[k]
    segs=[]; lo=bl[0][0]
    for j in range(1,len(bl)):
        if bl[j][0]-bl[j-1][1]>3e6: segs.append((lo,bl[j-1][1])); lo=bl[j][0]
    segs.append((lo,bl[-1][1])); act=sum(b-a for a,b in segs)
    iv=collections.defaultdict(list)
    for p in s['PO']:
        for a,b in segs: iv[p['n']].append((max(a,p['t']),min(b,p['t']+p['d'])))
    print('PP%d active %.0f ms segs %d; '%(k,act/1e3,len(segs))+'; '.join('%s union %.0f ms (%.1f%%)'%(n,union(v)/1e3,100*union(v)/act) for n,v in iv.items()))
    rb=[(p['t'],p['d']) for p in s['PM'] if p['n']=='run_batch' and p['d']>2e5]
    print('   run_batch >200ms:',[round(d/1e3,1) for t,d in rb])
    # busy over clean cycles with corrections
    T=0; acc=collections.Counter()
    for j in CLEAN:
        a,b=bl[j-1][1],bl[j][1]; T+=b-a
        G=[g for g in s['G'] if g['t']<b and g['t']+g['d']>a]
        cl=lambda f: union([(max(g['t'],a),min(g['t']+g['d'],b)) for g in G if f(g)])
        acc['compute_incl_hicache']+=cl(lambda g: g['k']!='nccl' and not g['k'].startswith('memcpy') and g['k']!='memset')
        acc['compute_excl_hicache']+=cl(lambda g: g['k'] not in('nccl','memset','hicache_kernel') and not g['k'].startswith('memcpy'))
        acc['any']+=cl(lambda g: True)
        acc['any_excl_memcpy>1ms']+=cl(lambda g: not (g['k'].startswith('memcpy') and g['d']>1e3))
        acc['any_excl_memcpy>1ms_and_nccl>2ms']+=cl(lambda g: not (g['k'].startswith('memcpy') and g['d']>1e3) and not (g['k']=='nccl' and g['d']>2e3))
        acc['hicache_kernel']+=cl(lambda g: g['k']=='hicache_kernel')
    print('   clean cycles mean %.1f ms: '%(T/1e3/len(CLEAN))+', '.join('%s %.1f%%'%(x,100*v/T) for x,v in acc.items()))
    # hicache backup kernel: streams, time outside steps
    hb=[g for g in s['G'] if g['k']=='hicache_kernel' and g['x']=='hicache_backup' and any(a<=g['t']<=b for a,b in segs)]
    ins=sum(union([(max(g['t'],x),min(g['t']+g['d'],y)) for x,y in bl]) for g in hb)
    print('   hicache_backup kernels n=%d streams %s total %.1f ms, outside steps %.1f ms'%(len(hb),dict(collections.Counter(g['s'] for g in hb)),sum(g['d'] for g in hb)/1e3,(sum(g['d'] for g in hb)-ins)/1e3))
# PP3 recv_requests end vs PP2 recv_requests end (clean batches)
for k in (1,2,3):
    rr=sorted([p for p in S[k]['PM'] if p['n']=='recv_requests' and p['d']>5e3],key=lambda p:p['t'])
    rp=sorted([p for p in S[k-1]['PM'] if p['n']=='recv_requests'],key=lambda p:p['t'])
    out=[]
    for p in rr:
        e=p['t']+p['d']
        prev=[q for q in rp if q['t']+q['d']<=e+0.5e3]
        if prev:
            q=prev[-1]; out.append((round(p['d']/1e3,1),round((e-(q['t']+q['d']))/1e3,2)))
    print('PP%d recv_requests>5ms: (dur, end minus latest PP%d recv_requests end) ms'%(k,k-1),out[:25])
