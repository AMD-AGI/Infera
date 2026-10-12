import re, subprocess
exec(open("step_regress.yihou.py").read().split("pat = re.compile")[0])
p = re.compile(r"T(\d\d:\d\d:\d\d)\.\d+Z \[[^\]]*DP\d TP\d\] Decode batch, #running-req: (\d+),.*?token usage: ([\d.]+),.*?#prealloc-req: (\d+), #transfer-req: (\d+), #retracted-req: (\d+).*?#queue-req: (\d+)")
for k, (sd, bd) in runs.items():
    s, e = window(f"{R}/{bd}/benchmark.log")
    out = subprocess.run(["strings", f"{R}/{sd}/server-logs/decode-0.a1.log"], capture_output=True, text=True).stdout
    rows = [m.groups() for m in p.finditer(out) if s <= m.group(1) <= e]
    n = len(rows); f = lambda i: sum(int(r[i]) > 0 for r in rows) / n * 100
    print(f"{k}: n={n} prealloc>0 {f(3):.1f}% transfer>0 {f(4):.1f}% retracted>0 {f(5):.1f}% queue>0 {f(6):.1f}% maxusage {max(float(r[2]) for r in rows):.2f} mean_transfer {sum(int(r[4]) for r in rows)/n:.2f}")
