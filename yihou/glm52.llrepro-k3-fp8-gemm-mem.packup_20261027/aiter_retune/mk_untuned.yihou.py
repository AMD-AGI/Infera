# Build aiter tuner input CSVs from a decode engine log (aiter "not found tuned config" + fused_moe "2stage default" lines).
import re,sys,os
log,out=sys.argv[1],sys.argv[2]
g=re.compile(r"shape is M:(\d+), N:(\d+), K:(\d+) dtype='([^']+)' otype='([^']+)' bias=(\w+), scaleAB=(\w+), bpreshuffle=(\w+), not found tuned config")
m=re.compile(r"using 2stage default for \('gfx950', 256, (\d+), (\d+), (\d+), (\d+), (\d+), <ActivationType\.(\w+): \d+>, '([^']+)', '([^']+)', '([^']+)', '([^']+)', (\w+), (\w+)\)")
G=set();M=set()
for l in open(log,errors="ignore"):
    a=g.search(l)
    if a: G.add((int(a[1]),int(a[2]),int(a[3]),a[6],a[4],a[5],a[7],a[8]))
    b=m.search(l)
    if b: M.add((int(b[1]),b[2],b[3],b[4],b[5],"ActivationType."+b[6],b[7],b[8],b[9],b[10],str(int(b[11]=="True")),str(int(b[12]=="True"))))
with open(os.path.join(out,"decode_bf16_untuned_gemm.csv"),"w") as f:
    f.write("M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle\n")
    for r in sorted(G,key=lambda r:(r[1],r[2],r[0])): f.write(",".join(map(str,r))+"\n")
with open(os.path.join(out,"decode_bf16_untuned_fmoe.csv"),"w") as f:
    f.write("token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1\n")
    for r in sorted(M,key=lambda r:r[0]): f.write(",".join(map(str,r))+"\n")
print(len(G),"gemm rows;",len(M),"fmoe rows")
