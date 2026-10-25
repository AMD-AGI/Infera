# GPU busy over clean-batch cycles only: cycle(j) = [end(j-1), end(j)] per stage. Usage: cleanbusy.yihou.py <dir>
import json, sys
CLEAN=[1,2,7,8,9,10,11,12,13,14,15]
def union(iv):
    iv=sorted(iv); tot=0; cs=ce=None
    for a,b in iv:
        if cs is None: cs,ce=a,b
        elif a<=ce: ce=max(ce,b)
        else: tot+=ce-cs; cs,ce=a,b
    if cs is not None: tot+=ce-cs
    return tot
for k in range(4):
    s=json.load(open(f'{sys.argv[1]}/ex2.pp{k}.json')); st=s['steps']; bl=[]; i=0
    while i<len(st):
        e=st[i]; end=e['t']+e['d']
        if i+1<len(st) and 'DRAFT' in st[i+1]['n']: end=max(end,st[i+1]['t']+st[i+1]['d']); i+=1
        bl.append((e['t'],end)); i+=1
    T=0; acc={'compute':0,'nccl':0,'any':0,'memcpy':0}
    for j in CLEAN:
        a,b=bl[j-1][1],bl[j][1]; T+=b-a
        cl=lambda f: union([(max(g['t'],a),min(g['t']+g['d'],b)) for g in s['G'] if f(g) and g['t']<b and g['t']+g['d']>a])
        acc['compute']+=cl(lambda g: g['k']!='nccl' and not g['k'].startswith('memcpy') and g['k']!='memset')
        acc['nccl']+=cl(lambda g: g['k']=='nccl'); acc['memcpy']+=cl(lambda g: g['k'].startswith('memcpy')); acc['any']+=cl(lambda g: True)
    print('PP%d clean cycles %d, mean cycle %.1f ms: '%(k,len(CLEAN),T/1e3/len(CLEAN))+', '.join('%s %.1f%%'%(x,100*v/T) for x,v in acc.items()))
