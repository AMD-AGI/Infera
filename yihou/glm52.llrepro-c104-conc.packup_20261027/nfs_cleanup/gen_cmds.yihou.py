#!/usr/bin/env python3
"""gen_cmds (task NFS cleanup 2026-10-27): turn a validated path list into a shell script of LITERAL per-path commands
(no variable delete targets). usage: gen_cmds.yihou.py <mode del|gzip> <list> <out.sh>"""
import os, shlex, sys
mode, lst, out = sys.argv[1:4]
R = '/shared_nfs/yihou_glm52/'
lines = ['#!/usr/bin/env bash', f'# generated from {lst} ({mode}); literal named paths only', 'n=0']
for p in (l.rstrip('\n') for l in open(lst)):
    if not p.startswith(R) or len([c for c in p.split('/') if c]) < 5 or '..' in p.split('/'):
        sys.exit(f'bad path, abort: {p!r}')
    q = shlex.quote(p)
    if mode == 'del':
        lines.append(f'[ -e {q} ] && nice -n 10 ionice -c3 rm -r -- {q} && n=$((n+1))')
    else:
        g = shlex.quote(p + '.gz')
        lines.append(f'[ -f {q} ] && [ ! -e {g} ] && [ -z "$(find {q} -mmin -1440)" ] && nice -n 10 ionice -c3 gzip -1 -c -- {q} > {g} '
                     f'&& gzip -t -- {g} && touch -r {q} {g} && rm -- {q} && n=$((n+1))')
lines.append('echo "done n=$n"')
open(out, 'w').write('\n'.join(lines) + '\n')
print(out, len(lines) - 4, 'commands')
