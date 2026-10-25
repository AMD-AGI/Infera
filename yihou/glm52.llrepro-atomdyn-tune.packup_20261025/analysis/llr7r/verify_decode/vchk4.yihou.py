# verifier part 4 (yihou): dump one step's verify-window GPU events with their CPU launch context.
import gzip,json,sys,collections,bisect
d=json.load(gzip.open(sys.argv[1])); K=int(sys.argv[2]); ev=d['traceEvents']; X=[e for e in ev if e.get('ph')=='X']
G=sorted([e for e in X if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')],key=lambda e:e['ts'])
byc=collections.defaultdict(list)
for e in G: byc[e['args'].get('correlation')].append(e)
rt=[e for e in X if e.get('cat')=='cuda_runtime']; rtc={e['args'].get('correlation'):e for e in rt}
ua=sorted([e for e in X if e.get('cat')=='user_annotation'],key=lambda e:e['ts'])
gl=sorted([e for e in rt if e['name']=='hipGraphLaunch'],key=lambda e:e['ts'])
seq=[(e,len(byc.get(e['args'].get('correlation'),[]))) for e in gl]
ts=[e['ts'] for e in G]
st=[i for i in range(len(seq)-2) if seq[i][1]>100 and seq[i+1][1]==0]
i=st[K]
dk=byc[seq[i][0]['args']['correlation']]; ek=byc[seq[i+2][0]['args']['correlation']]
d1=max(k['ts']+k['dur'] for k in dk); e0=min(k['ts'] for k in ek)
vl=seq[i+1][0]
print('verify launch at %.3f ms rel draft end, dur %.3f, tid %s, stream arg %s'%((vl['ts']-d1)/1e3,vl['dur']/1e3,vl['tid'],vl['args']))
print('draft graph launch args',seq[i][0]['args'],'kernel stream',dk[0]['args'].get('stream'),dk[0]['tid'])
w=G[bisect.bisect_right(ts,d1):bisect.bisect_left(ts,e0)]
def ann(t):
    a=[u['name'] for u in ua if u['ts']<=t<=u['ts']+u['dur']]; return a[-2:]
for k in w:
    r=rtc.get(k['args'].get('correlation'))
    print('%8.3f %7.3f tid%s st%s  %-50s | launch %s at %.3f rel d1  %s'%((k['ts']-d1)/1e3,k['dur']/1e3,k['tid'],k['args'].get('stream'),k['name'][:50],r['name'] if r else None,(r['ts']-d1)/1e3 if r else 0,ann(r['ts']) if r else ''))
print('draft_extend start rel d1 %.3f'%((e0-d1)/1e3))
