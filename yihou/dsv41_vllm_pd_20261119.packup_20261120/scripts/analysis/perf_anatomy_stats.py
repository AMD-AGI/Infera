import re, sys, statistics as st
# Parse vLLM RustFrontend log_stats lines; summarize the steady window (middle 60% of the active span).
pat = re.compile(r'(\d\d-\d\d \d\d:\d\d:\d\d).*?Avg prompt tput: ([\d.]+) toks/s, Avg generation tput: ([\d.]+) toks/s, Reqs Running: (\d+), Waiting: (\d+)(?:, Deferred: (\d+) reqs)?.*?GPU KV cache used: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%(?:, External prefix cache hit rate: ([\d.]+)%)?')
rows=[]
for line in open(sys.argv[1], errors='replace'):
    if 'Reqs Running' not in line: continue
    line = re.sub(r'\x1b\[[0-9;]*m','',line)
    m = pat.search(line)
    if not m: continue
    t,pt,gt,run,wait,defr,kv,hit,ext = m.groups()
    rows.append((t,float(pt),float(gt),int(run),int(wait),int(defr or 0),float(kv),float(hit),float(ext) if ext else None))
col=int(sys.argv[2]) if len(sys.argv)>2 else 2
act=[i for i,r in enumerate(rows) if r[col]>100]
if not act: print('no active'); sys.exit()
a,b=act[0],act[-1]; n=b-a
w=rows[a+int(0.2*n):a+int(0.8*n)]
def s(i): 
    v=[r[i] for r in w if r[i] is not None]
    return (round(st.median(v),1), round(st.mean(v),1)) if v else None
print(sys.argv[1].split('/')[-1][:60], 'lines',len(rows),'window',w[0][0],'->',w[-1][0],'n',len(w))
for name,i in [('prompt_tput',1),('gen_tput',2),('running',3),('waiting',4),('deferred',5),('kv_used%',6),('prefix_hit%',7),('ext_hit%',8)]:
    print(f'  {name:12s} median/mean', s(i))
