import gzip, json, sys, collections
f=sys.argv[1]
d=json.load(gzip.open(f))
ev=d['traceEvents']
X=[e for e in ev if e.get('ph')=='X']
t0=min(e['ts'] for e in X)
print('t0',t0,'baseTimeNanoseconds',d.get('baseTimeNanoseconds'))
# sample args
seen=set()
for e in X:
    c=e.get('cat')
    if c in ('kernel','gpu_memcpy','gpu_user_annotation','cuda_runtime') and c not in seen:
        seen.add(c); print('SAMPLE',json.dumps(e)[:800])
# streams
sc=collections.defaultdict(float); sn=collections.Counter()
for e in X:
    if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset'):
        k=(e['pid'],e['tid']); sc[k]+=e['dur']; sn[k]+=1
for k in sc: print('STREAM',k,sn[k],'%.1f ms'%(sc[k]/1e3))
# memcpy by stream, with bytes
mb=collections.defaultdict(lambda:[0,0.0,0])
for e in X:
    if e.get('cat')=='gpu_memcpy':
        a=e.get('args',{}); k=(e['name'],e['tid'])
        mb[k][0]+=1; mb[k][1]+=e['dur']; mb[k][2]+=a.get('bytes',0) or 0
for k,v in mb.items(): print('MEMCPY',k,v[0],'%.1f ms'%(v[1]/1e3),'%.2f GB'%(v[2]/1e9))
# timeline of annotations
L=[]
for e in X:
    c=e.get('cat'); n=e['name']
    if c=='gpu_user_annotation' and (n.startswith('step[') or n.startswith('nccl') or n=='gloo:send'):
        L.append((e['ts'],e['dur'],'G',n,e['tid']))
    if c=='user_annotation' and (n.startswith('step[') or n in('recv_res_dict_from_prev_stage','send_res_dict_to_next_stage','send_proxy_dict_to_next_stage','scheduler.recv_requests','scheduler.process_queue','scheduler.run_batch','scheduler.process_batch_result','gloo:recv') or (n=='gloo:send' and e['dur']>5000)):
        L.append((e['ts'],e['dur'],'C',n,e['tid']))
L.sort()
for ts,du,k,n,tid in L: print('%10.2f %9.2f %s %-6s %s'%((ts-t0)/1e3,du/1e3,k,tid,n))
tmax=max(e['ts']+e['dur'] for e in X); print('span ms',(tmax-t0)/1e3)
