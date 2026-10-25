# Independent raw parse of one prefill PP trace. Usage: raw.yihou.py <trace.json.gz>
import gzip, json, sys, collections, re, datetime
d=json.load(gzip.open(sys.argv[1])); base=d.get('baseTimeNanoseconds',0)/1e3
ev=[e for e in d['traceEvents'] if e.get('ph')=='X']; del d
utc=lambda t: datetime.datetime.utcfromtimestamp((base+t)/1e6).strftime('%H:%M:%S.%f')[:-3]
K=[e for e in ev if e.get('cat')=='kernel']
M=[e for e in ev if e.get('cat') in ('gpu_memcpy','gpu_memset')]
gs=sorted([e for e in ev if e.get('cat')=='gpu_user_annotation' and e['name'].startswith('step[')],key=lambda e:e['ts'])
print('n gpu step annotations',len(gs)); 
for i,e in enumerate(gs): print(i,utc(e['ts']),round(e['dur']/1e3,1),e['name'][:60])
# group target+draft into batch
B=[];i=0
while i<len(gs):
    a=gs[i]['ts']; b=a+gs[i]['dur']; nm=gs[i]['name']
    if i+1<len(gs) and 'DRAFT' in gs[i+1]['name'] and gs[i+1]['ts']-a<1e6: b=max(b,gs[i+1]['ts']+gs[i+1]['dur']); i+=1
    B.append((a,b,nm)); i+=1
print('batches',len(B))
CLEAN=[1,2,7,8,9,10,11,12,13,14,15]
# max kernel per batch
for j,(a,b,nm) in enumerate(B):
    mk=max([e['dur'] for e in K if a<=e['ts']<b] or [0])
    print('b%02d dur %.1f maxkernel %.1f ms toks %s'%(j,(b-a)/1e3,mk/1e3,re.search(r'toks=(\d+)',nm).group(1)))
# kernel names inside clean batch windows (by start time), excluding nccl
tot=collections.Counter(); cnt=collections.Counter(); strm=collections.defaultdict(collections.Counter)
for j in CLEAN:
    a,b,_=B[j]
    for e in K:
        if a<=e['ts']<b:
            n=re.sub(r'\d+','#',e['name'])[:110]; tot[n]+=e['dur']; cnt[n]+=1; strm[n][e['args'].get('stream')]+=1
T=sum(tot.values())
print('total kernel ms per clean batch (all streams, by start inside step) %.2f'%(T/1e3/len(CLEAN)))
for n,v in tot.most_common(80): print('%8.2f ms/b n/b=%6.1f %s %s'%(v/1e3/len(CLEAN),cnt[n]/len(CLEAN),dict(strm[n]),n))
