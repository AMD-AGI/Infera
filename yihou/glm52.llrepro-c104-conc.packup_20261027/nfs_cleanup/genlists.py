# Builds candidate list files from the read-only walk. Deletes nothing.
import glob, re, time, collections, os
A='/shared_nfs/yihou_glm52/du_audit.yihou'; O=A+'/list'; R='/shared_nfs/yihou_glm52/'
now=time.time(); G=1<<30
ent={}  # path -> (type, alloc, mtime, owner)
for f in glob.glob(O+'/*.tsv'):
    for line in open(f, errors='replace'):
        p=line.rstrip('\n').split('\t',5)
        if len(p)<6: continue
        ent[p[5].rstrip('/')]=(p[0],int(p[2])*512,float(p[3]),p[4])
PROT_PREFIX=[R+'llrepro_20261023/'+x for x in ['kit','work/cache','prep_next.yihou','mxdraft.yihou','acc.yihou','mlacfg.yihou','ec_port.yihou','pk11.yihou','pk12.yihou']]+[R+'kf_e2e_20261023',R+'v0521cap2_20261013/hist.yihou.py']
def protected(p):
    if any(p==x or p.startswith(x+'/') for x in PROT_PREFIX): return True
    if re.search(r'/(InferenceX|src|harness|overlay|overlays|patches|kit)(/|$)',p): return True
    m=re.match(re.escape(R)+r'llrepro_20261023/work/runs/[^/]+',p)
    if m and any(k.startswith(m.group(0)) and now-v[2]<86400 for k,v in sub(m.group(0))): return True
    return False
# children index
kids=collections.defaultdict(list)
for k in ent: kids[os.path.dirname(k)].append(k)
def sub(p):
    st=[p]; out=[]
    while st:
        q=st.pop(); out.append((q,ent.get(q,('?',0,0,'?'))))
        st.extend(kids.get(q,[]))
    return out
def summ(p):
    s=sub(p); own=set(v[3] for _,v in s); return own, sum(v[1] for _,v in s if v[0]=='f')
roots=[]
for k,v in ent.items():
    if v[0]!='d': continue
    if re.fullmatch(re.escape(R)+r'llying_dcp_20261022/work/[^/]+/cache',k): roots.append(k)
    if re.fullmatch(re.escape(R)+r'v0521dbg_20261010/cache/\d+',k): roots.append(k)
    if re.fullmatch(re.escape(R)+r'dcp_pd_20261021/pd/cache/[^/]+',k): roots.append(k)
    if re.fullmatch(re.escape(R)+r'dcp_pd_20261021/cache/(v3-045|v3-060|v3-075|046|054|151|268b|agentx-hf)',k): roots.append(k)
    if re.fullmatch(re.escape(R)+r'agentx44_20261009/cache/102(-v0521)?',k): roots.append(k)
    if re.fullmatch(re.escape(R)+r'llrepro_20261023/work/runs/[^/]+/agentx/tmp/aiperf_mmap_[0-9a-f]+',k): roots.append(k)
s1=[];s2=[];s4=[];left=[]
def emit(p, owner):
    tgt = s4 if owner=='root' else (s1 if re.search(r'(dataset_mmap|aiperf_mmap_)',p) else s2)
    tgt.append(p)
def split(p):
    if protected(p): left.append(('protected',p)); return
    own,_=summ(p)
    if len(own)==1: emit(p, next(iter(own))); return
    for c in sorted(kids.get(p,[])):
        if ent[c][0]=='d' or ent[c][1]>=64*(1<<20): split(c)
        # small mixed-owner files directly in a mixed dir are left in place
for r in sorted(roots): split(r)
# step1: pull dataset_mmap subdirs out of yihou-owned s2 roots for clarity (s2 entries that contain dataset_mmap)
def size(p): return summ(p)[1]
def wr(name, lst):
    with open(f'{A}/{name}','w') as f:
        for p in lst: f.write(p+'\n')
    print(f'{name}: {len(lst)} paths, {sum(size(p) for p in lst)/G:.1f} GB')
wr('del_step1.yihou.list', s1); wr('del_step2.yihou.list', s2); wr('del_step4_root.yihou.list', s4)
# step3 compress list
s3=[]
for k,v in ent.items():
    if v[0]!='f' or v[3]!='yihou' or now-v[2]<86400 or protected(k): continue
    if re.search(r'(server_metrics_export\.json|profile_export_aiperf_timeslices\.(csv|json)|profile_export\.jsonl)$',k) or (re.search(r'\.(log|out|err)$',k) and v[1]>=50*(1<<20)):
        if any(k.startswith(x+'/') for x in s1+s2+s4): continue
        s3.append(k)
wr('gzip_step3.yihou.list', sorted(s3))
# root-owned non-regenerable items, for the user
keep_root=[k for k,v in ent.items() if v[0]=='f' and v[3]=='root' and re.search(r'(\.trace\.json(\.gz)?|\.pt)$',k)]
print('root traces/.pt kept:', len(keep_root), round(sum(ent[k][1] for k in keep_root)/G,1),'GB')
print('left (protected/mixed):', left[:20])
print('step4 sample:'); [print('  ',p, round(size(p)/G,2)) for p in s4[:15]]
print('step2 sample:'); [print('  ',p, round(size(p)/G,2)) for p in s2[:15]]
