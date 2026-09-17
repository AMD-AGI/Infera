#!/usr/bin/env python3
"""NEW packaging helper: relocate verbatim launchers into a fresh workspace.

Offline only: does not invoke Docker, Spur, a GPU runtime, or a shell script.
The original scripts and all packaged evidence remain untouched.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

KIT = Path(__file__).resolve().parents[1]
OLD = '/shared_nfs/yihou/playground/glm52_fake_tp4ep4_10k500_20260910-0452'
OLD_NAME = 'g52-fake-short-tp4ep4'
PROTECTED = [Path(OLD),
             Path('/shared_nfs/yihou/playground/glm52_decode_fake_20260909_1016'),
             Path('/shared_nfs/yihou/packups/glm52_fake_decode_c16.packup_20260910-040449'),
             KIT]


def prepare(destination, name, port):
    raw = Path(destination).expanduser()
    if raw.exists() or raw.is_symlink():
        raise ValueError('Destination must not exist (including symlinks)')
    dst = raw.resolve()
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', str(dst)):
        raise ValueError('Use an absolute shell-safe path: letters/digits/_/./-/slash')
    if any(dst == root.resolve() or root.resolve() in dst.parents for root in PROTECTED):
        raise ValueError('Destination must be outside the kit and original experiment trees')
    if not dst.parent.is_dir():
        raise ValueError('Destination parent must already exist')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,90}', name) or name == OLD_NAME:
        raise ValueError('Use a fresh Docker-safe container name, not the historical name')
    if not 1024 <= port <= 65535:
        raise ValueError('Port must be 1024..65535')
    dst.mkdir(exist_ok=False)
    for directory in ['scripts/original', 'scripts/tests', 'source/baseline',
                      'reference', 'state', 'rounds', 'cache', 'patches']:
        (dst / directory).mkdir(parents=True, exist_ok=True)
    for source in (KIT / 'source').rglob('*.py'):
        target = dst / source.relative_to(KIT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for source in (KIT / 'scripts/tests').glob('*.py'):
        shutil.copy2(source, dst / 'scripts/tests' / source.name)
    for source in (KIT / 'patches').glob('*.patch'):
        shutil.copy2(source, dst / 'patches' / source.name)
    for filename in ['synthetic_prompts.json', 'Dockerfile']:
        shutil.copy2(KIT / 'reference' / filename, dst / 'reference' / filename)
    substitutions = [(OLD, str(dst)), (OLD_NAME, name), ('31816', str(port))]
    records = []
    for source in sorted((KIT / 'scripts/original').iterdir()):
        data = source.read_bytes()
        shutil.copy2(source, dst / 'scripts/original' / source.name)
        text = data.decode()
        for old, new in substitutions:
            text = text.replace(old, new)
        if OLD in text or OLD_NAME in text:
            raise ValueError('Historical runtime destination survived substitution')
        target = dst / 'scripts' / source.name
        target.write_text(text)
        target.chmod(source.stat().st_mode & 0o777)
        records.append({'source': str(source.relative_to(KIT)),
                        'generated': str(target.relative_to(dst)),
                        'original_sha256': hashlib.sha256(data).hexdigest(),
                        'generated_sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    (dst / 'state/preparation.json').write_text(json.dumps({
        'created_at': datetime.now(timezone.utc).isoformat(),
        'helper': 'NEW packaging relocation helper; not used by the measured run',
        'kit': str(KIT), 'workspace': str(dst), 'container_name': name, 'port': port,
        'substitutions': substitutions, 'scripts': records,
        'external_model': '/shared_nfs/models/GLM-5.2-MXFP4',
        'cache_policy': 'Empty: no compiled cache or weights copied',
    }, indent=2) + '\n')
    print(json.dumps({'workspace': str(dst), 'container': name, 'port': port}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination')
    parser.add_argument('--name', default=None)
    parser.add_argument('--port', type=int, default=31816)
    args = parser.parse_args()
    name = args.name or ('g52-replay-' + hashlib.sha256(str(Path(args.destination).resolve()).encode()).hexdigest()[:12])
    try:
        prepare(args.destination, name, args.port)
    except ValueError as exc:
        parser.error(str(exc))
