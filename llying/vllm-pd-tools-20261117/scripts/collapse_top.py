"""Top inclusive / self frames of a py-spy --format raw profile.   collapse_top.py RAW [N=30]"""
import collections, sys
n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
incl, leaf, total = collections.Counter(), collections.Counter(), 0
for line in open(sys.argv[1]):
    stack, _, c = line.rstrip().rpartition(" ")
    if not c.isdigit():
        continue
    c = int(c); total += c
    frames = stack.split(";")
    leaf[frames[-1]] += c
    for f in set(frames):
        incl[f] += c
print("samples", total)
print("-- inclusive")
for f, c in incl.most_common(n):
    print(f"{100 * c / total:5.1f}% {f[:160]}")
print("-- self")
for f, c in leaf.most_common(15):
    print(f"{100 * c / total:5.1f}% {f[:160]}")
