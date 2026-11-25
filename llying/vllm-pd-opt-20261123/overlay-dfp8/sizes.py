import json,re,struct,os,collections,sys
d=sys.argv[1]
c=json.load(open(d+'/config.json'));ex=c['quantization_config']['exclude']
pat=collections.Counter(re.sub(r'layers\.\d+','layers.N',e) for e in ex)
print(pat); print([e for e in ex if re.search(r'layers\.(7[89])\.',e)][:50])
print(sorted(set(int(m.group(1)) for e in ex for m in [re.search(r'layers\.(\d+)\.self_attn\.indexer',e)] if m)))
idx=json.load(open(d+'/model.safetensors.index.json'))['weight_map']
files=sorted(set(idx.values()))
agg=collections.defaultdict(int); dt=collections.Counter()
for f in files:
  with open(d+'/'+f,'rb') as fh:
    n=struct.unpack('<Q',fh.read(8))[0]; h=json.loads(fh.read(n))
  for k,v in h.items():
    if k=='__metadata__':continue
    sz=v['data_offsets'][1]-v['data_offsets'][0]
    m=re.search(r'layers\.(\d+)\.',k); L=int(m.group(1)) if m else -1
    mtp = L>=78
    key=re.sub(r'layers\.\d+','layers.N',k)
    key=re.sub(r'experts\.\d+','experts.E',key)
    agg[(mtp,key,v['dtype'],tuple(v['shape']) if 'experts.E' not in key else ())]+=sz
for (mtp,k,t,s),sz in sorted(agg.items(),key=lambda x:-x[1]):
  if sz>5e7: print(mtp,k,t,s,round(sz/2**30,3))
tot=collections.defaultdict(float)
for (mtp,k,t,s),sz in agg.items(): tot[(mtp,t)]+=sz/2**30
print(dict(tot))
