#!/usr/bin/env python3
"""argdiff (task llrepro 2026-10-24): diff the sglang server_args line of two server logs (ours vs llying kit).
usage: python3 -I argdiff.yihou.py <log A> <log B> [labelA labelB]"""
import re, sys


def args(path):
    with open(path, errors='replace') as f:
        for line in f:
            i = line.find('server_args=ServerArgs(')
            j = line.find('server_args={')
            if i >= 0 or j >= 0:
                body = line[(i if i >= 0 else j):]
                # key=value (ServerArgs repr) or 'key': value (dict repr)
                kv = re.findall(r"'?(\w+)'?\s*[=:]\s*('(?:[^'\\]|\\.)*'|\[[^\]]*\]|\{[^{}]*\}|[^,)}]+)", body)
                return {k: v.strip() for k, v in kv}
    return {}


a, b = args(sys.argv[1]), args(sys.argv[2])
la, lb = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ('A', 'B')
print(f'# {la}: {len(a)} keys, {lb}: {len(b)} keys')
for k in sorted(set(a) | set(b)):
    if a.get(k) != b.get(k):
        print(f'{k:45} {la}={str(a.get(k))[:70]:72} {lb}={str(b.get(k))[:70]}')
