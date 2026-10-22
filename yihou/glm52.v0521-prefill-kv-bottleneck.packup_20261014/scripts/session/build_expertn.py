import os, shutil, sys
N, S = sys.argv[1], sys.argv[2]
T = N + "/mounts/expertn"
os.makedirs(T, exist_ok=True)
shutil.copyfile(N + "/yihou_expertn.py", T + "/yihou_expertn.py")
src = open(S + "/model_loader/loader.py").read()
anchor = "            use_safetensors = resolved_source.use_safetensors\n"
assert src.count(anchor) == 1
hunk = open(N + "/loader_hunk.py").read()
open(T + "/loader.yihou-expertn.py", "w").write(src.replace(anchor, anchor + hunk))
for f in (T + "/loader.yihou-expertn.py", T + "/yihou_expertn.py"):
    compile(open(f).read(), f, "exec")
print("COMPILE_OK", T)
