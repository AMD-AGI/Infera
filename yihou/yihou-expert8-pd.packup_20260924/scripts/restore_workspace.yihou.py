#!/usr/bin/env python3
"""Restore verbatim payload to a NEW shared workspace, relocating only workspace paths."""
import argparse
import json
from pathlib import Path
import shutil

ORIGINAL = '/home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd/results/yihou-expert8-pd'
PACK = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    target = args.workspace.resolve()
    if not args.workspace.is_absolute() or 'yihou' not in str(target):
        parser.error('use an absolute shared workspace path containing yihou')
    if target.exists() or target.is_symlink():
        parser.error('refusing to overwrite any existing workspace')
    if any(c.isspace() for c in str(target)):
        parser.error('workspace path must not contain whitespace')
    shutil.copytree(PACK / 'payload', target)
    shutil.copytree(PACK / 'patches', target / 'patches')
    for path in target.rglob('*'):
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if ORIGINAL in text:
            path.write_text(text.replace(ORIGINAL, str(target)))
    for subdir in ['model','tmp/router','cache/router','rounds/002-bringup','rounds/003-fixed-c32']:
        (target / subdir).mkdir(parents=True, exist_ok=True)
    (target / 'restore-record.json').write_text(json.dumps({
        'packup': str(PACK), 'original_workspace': ORIGINAL,
        'restored_workspace': str(target),
        'change': 'workspace path substitution only; no model or GPU actions performed'
    }, indent=2)+'\n')
    print(target)

if __name__ == '__main__':
    main()
