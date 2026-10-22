import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
def flat(d,p=""):
    out={}
    for k,v in d.items():
        if isinstance(v,dict): out.update(flat(v,p+k+"."))
        else: out[p+k]=v
    return out
fa,fb=flat(a),flat(b)
print("# key-level diff: %s (model config.json) -> %s (overlay)" % (sys.argv[1], sys.argv[3]))
for k in sorted(set(fa)|set(fb)):
    va=fa.get(k,"<absent>"); vb=fb.get(k,"<absent>")
    if va!=vb:
        sa=json.dumps(va); sb=json.dumps(vb)
        if len(sa)>200: sa=sa[:200]+"...(%d chars)"%len(sa)
        if len(sb)>200: sb=sb[:200]+"...(%d chars)"%len(sb)
        print(k, ":", sa, "->", sb)
