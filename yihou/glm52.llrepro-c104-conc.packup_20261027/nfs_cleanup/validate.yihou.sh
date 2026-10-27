#!/bin/bash
# validate.yihou.sh <listfile> : DRY RUN ONLY. Validates every line; prints OK/INVALID; deletes nothing.
set -u
L="$1"
A=/shared_nfs/yihou_glm52/du_audit.yihou
[ -f "$L" ] || { echo "no list $L"; exit 2; }
# mounted sources of running containers on 135 and 136
{ docker ps -q | xargs -r docker inspect -f '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}'
  ssh -n -o BatchMode=yes crsuse2-m2m-136 "docker ps -q | xargs -r docker inspect -f '{{range .Mounts}}{{.Source}}{{\"\n\"}}{{end}}'"; } 2>/dev/null | grep '^/shared_nfs/yihou_glm52' | sort -u > $A/mounts.now.txt
python3 - "$L" $A/mounts.now.txt <<'PY'
import sys, os, re
L, M = sys.argv[1], sys.argv[2]
R='/shared_nfs/yihou_glm52/'
prot=[R+'llrepro_20261023/'+x for x in ['kit','work/cache','prep_next.yihou','mxdraft.yihou','acc.yihou','mlacfg.yihou','ec_port.yihou','pk11.yihou','pk12.yihou']]+[R+'kf_e2e_20261023',R+'v0521cap2_20261013/hist.yihou.py',R+'agentx44_20261009/cache/InferenceX',R+'dcp_pd_20261021/cache/InferenceX',R+'v0521dbg_20261010/cache/InferenceX',R+'v0521dbg_20261010/harness',R+'dcp_pd_20261021/pd/harness',R+'dcp_pd_20261021/src']
mounts=[l.strip().rstrip('/') for l in open(M) if l.strip()]
bad=0; n=0
for i,l in enumerate(open(L),1):
    p=l.rstrip('\n'); n+=1; why=None
    if not p: why='empty'
    elif not p.startswith(R): why='prefix'
    elif len([c for c in p.split('/') if c])<5: why='<5 components'
    elif '..' in p.split('/') or '*' in p: why='dotdot/glob'
    elif not os.path.lexists(p): why='missing'
    elif any(p==x or x.startswith(p+'/') or p.startswith(x+'/') for x in prot): why='protected'
    elif re.search(r'/(InferenceX|harness|overlay|patches|kit)(/|$)|\.tar(\.zst)?$|\.site\.env$',p): why='protected-pattern'
    elif any(p==m or p.startswith(m+'/') or m.startswith(p+'/') for m in mounts): why='mounted'
    if why: bad+=1; print(f'INVALID line {i}: {why}: {p}') if bad<=20 else None
print(f'{L}: {n} lines, {bad} invalid, mounts={len(mounts)} -> {"ABORT" if bad else "OK (dry run, nothing deleted)"}')
PY
