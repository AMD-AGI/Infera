# Create streaming-sync_back copies of run.v3 / pipeline.v3 / chain_cap2l (new files only; originals untouched).
H = "/shared_nfs/yihou_glm52/v0521dbg_20261010/harness"
import os
for f in ("run.v3s.yihou.sh", "pipeline.v3s.yihou.sh", "chain_cap2ls.yihou.sh"):
    assert not os.path.exists(f"{H}/{f}"), f
r = open(f"{H}/run.v3.yihou.sh").read()
old = '''  b64=$("${SSH[@]}" -n "$NODE" "cd $(q "$NODE_RD") 2>/dev/null || exit 0; \\
    tar --warning=no-file-changed -cf - $(q "${what[@]}") 2>/dev/null | base64 -w0") || return 1
  [[ -n "$b64" ]] || { echo "sync_back: nothing under $NODE_RD on $NODE" >&2; return 0; }
  base64 -d <<< "$b64" | tar -C "$RD" -xf -
'''
new = '''  # run.v3s (2026-10-15): stream ssh tar -> local tar; the original held the whole archive base64 in a bash variable,
  # which hung for 50 min under login-host memory-cgroup throttling (D state, mem_cgroup_handle_over_high).
  "${SSH[@]}" -n "$NODE" "cd $(q "$NODE_RD") 2>/dev/null || exit 0; \\
    tar --warning=no-file-changed -cf - $(q "${what[@]}") 2>/dev/null" | tar -C "$RD" -xf - || return 1
'''
assert r.count(old) == 1, "sync_back body not found"
r = r.replace(old, new).replace("local b64\n", "")
open(f"{H}/run.v3s.yihou.sh", "w").write(r)
p = open(f"{H}/pipeline.v3.yihou.sh").read()
assert p.count('D="bash $H/run.v3.yihou.sh"') == 1
open(f"{H}/pipeline.v3s.yihou.sh", "w").write(p.replace('D="bash $H/run.v3.yihou.sh"', 'D="bash $H/run.v3s.yihou.sh"'))
c = open(f"{H}/chain_cap2l.yihou.sh").read()
assert c.count('D="bash $H/run.v3.yihou.sh"') == 1 and c.count("bash $H/pipeline.v3.yihou.sh") == 1
c = c.replace('D="bash $H/run.v3.yihou.sh"', 'D="bash $H/run.v3s.yihou.sh"').replace("bash $H/pipeline.v3.yihou.sh", "bash $H/pipeline.v3s.yihou.sh")
c = c.replace("#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n# (chain_cap2ls: copy of chain_cap2l using run.v3s/pipeline.v3s = streaming sync_back, 2026-10-15)\n", 1)
open(f"{H}/chain_cap2ls.yihou.sh", "w").write(c)
print("ok")
