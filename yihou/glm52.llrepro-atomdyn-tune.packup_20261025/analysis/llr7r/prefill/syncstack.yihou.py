# Python call stacks of main-thread 'synchronize' calls > 2 ms that END in an inter-step gap. Usage: syncstack.yihou.py <trace> <ex2.json>
import gzip, json, sys, collections
d=json.load(gzip.open(sys.argv[1])); base=d['baseTimeNanoseconds']/1e3
ex=json.load(open(sys.argv[2])); tid=ex['main_tid']
st=sorted(ex['steps'],key=lambda z:z['t'])
gaps=[(st[i-1]['t']+st[i-1]['d'],st[i]['t']) for i in range(1,len(st)) if 0<st[i]['t']-(st[i-1]['t']+st[i-1]['d'])<3e6]
py=sorted([(e['ts']+base,e['ts']+base+e['dur'],e['name']) for e in d['traceEvents'] if e.get('ph')=='X' and e.get('cat')=='python_function' and e['tid']==tid],key=lambda x:(x[0],-x[1]))
syncs=[p for p in py if p[2].endswith('): synchronize') and p[1]-p[0]>2e3]
cnt=collections.Counter(); tot=collections.Counter()
for s in syncs:
    ing=any(a<s[1]<=b+1e3 for a,b in gaps); inside=any(a<=s[0]<=b for a,b in gaps)
    stack=[p[2].split('/sglang/')[-1] for p in py if p[0]<=s[0] and p[1]>=s[1] and p is not s and '/sglang/' in p[2]][-4:]
    k=('ends_in_gap' if ing else 'ends_in_step', ' < '.join(reversed(stack)))
    cnt[k]+=1; tot[k]+=s[1]-s[0]
for k,v in sorted(tot.items(),key=lambda x:-x[1]): print('%8.1f ms n=%3d %s'%(v/1e3,cnt[k],k))
