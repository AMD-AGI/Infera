#!/usr/bin/env python3
"""NEW standard-library CPU audit; no Docker, Spur, GPU or git operations.

Temporary test files live under this kit's provenance directory and are removed.
By default verify the final manifest; --before-manifest is for initial assembly.
"""
import argparse
import ast
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from recompute_metrics import compute

KIT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def command(args, expected=0, **kwargs):
    proc = subprocess.run(args, capture_output=True, text=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, **kwargs)
    if proc.returncode != expected:
        raise AssertionError(f'{args}: expected {expected}, got {proc.returncode}\n{proc.stdout}\n{proc.stderr}')
    return proc


def audit(before_manifest=False, verify_sources=False):
    results = {'audited_at': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
               'no_gpu_docker_spur_git_commands': True}
    manifest = json.loads((KIT / 'provenance/source_manifest.json').read_text())
    for row in manifest['records']:
        data = (KIT / row['destination']).read_bytes()
        assert sha(data) == row['destination_sha256'], row['destination']
        if row['compression'] == 'gzip':
            data = gzip.decompress(data)
        assert sha(data) == row['source_sha256'] and len(data) == row['source_bytes']
        if verify_sources:
            source = Path(row['source'])
            assert sha(source.read_bytes()) == row['source_sha256']
            assert source.stat().st_mtime_ns == row['source_mtime_ns']
    results['provenance_records_exact'] = len(manifest['records'])
    results['original_sources_content_and_mtime_unchanged'] = verify_sources
    if verify_sources:
        source = Path(manifest['source'])
        actual = {str(p) for p in source.rglob('*') if p.is_file() and 'cache' not in p.relative_to(source).parts}
        included = {r['source'] for r in manifest['records'] if str(source) + '/' in r['source']}
        assert actual == included
        results['all_noncache_source_files_preserved'] = len(actual)
    python_files, shell_files = [], []
    for file in sorted(KIT.rglob('*')):
        if not file.is_file():
            continue
        if file.suffix == '.py':
            ast.parse(file.read_text(), filename=str(file))
            python_files.append(str(file.relative_to(KIT)))
        elif file.suffix == '.sh':
            command(['bash', '-n', str(file)])
            shell_files.append(str(file.relative_to(KIT)))
    results['python_syntax_files'] = len(python_files)
    results['bash_syntax_files'] = len(shell_files)
    with tempfile.TemporaryDirectory(prefix='audit-', dir=KIT / 'provenance') as temporary:
        scratch = Path(temporary)
        patches = [
            ('fake_proposal.patch', 'eagle_disaggregation.py', 'eagle_disaggregation_fake_fix.py', 'python/sglang/srt/speculative/eagle_disaggregation.py'),
            ('fake_index_elision.patch', 'kv_cache_configurator_original.py', 'kv_cache_configurator_fake_fix.py', 'python/sglang/srt/mem_cache/kv_cache_configurator.py'),
            ('fake_topk_domain.patch', 'dsa_utils.py', 'dsa_utils_fake_fix.py', 'python/sglang/srt/layers/attention/dsa/utils.py'),
        ]
        for patch_name, original, modified, target in patches:
            path = scratch / 'patch' / target
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(KIT / 'source/baseline' / original, path)
            command(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(KIT / 'patches' / patch_name)], cwd=scratch / 'patch')
            assert path.read_bytes() == (KIT / 'source' / modified).read_bytes()
        results['three_patches_apply_zero_fuzz_and_match_exactly'] = True
        for test, original, modified in [
            ('test_index_elision.py', 'kv_cache_configurator_original.py', 'kv_cache_configurator_fake_fix.py'),
            ('test_fake_topk_domain.py', 'dsa_utils.py', 'dsa_utils_fake_fix.py'),
        ]:
            command([sys.executable, str(KIT / 'scripts/tests' / test), str(KIT / 'source/baseline' / original)], expected=1)
            command([sys.executable, str(KIT / 'scripts/tests' / test), str(KIT / 'source' / modified)])
        results['guard_red_green'] = {'index_elision_cases': 72, 'topk_domain_cases': 48}
        # Relocate the helper's kit itself, then build outside that cold copy.
        cold = scratch / 'cold-kit'
        cold.mkdir()
        for directory in ['scripts/original', 'scripts/tests', 'source', 'reference', 'patches']:
            shutil.copytree(KIT / directory, cold / directory)
        shutil.copy2(KIT / 'scripts/prepare_workspace.py', cold / 'scripts/prepare_workspace.py')
        helper = cold / 'scripts/prepare_workspace.py'
        fresh = scratch / 'fresh-run'
        command([sys.executable, str(helper), str(fresh), '--name', 'audit-new-container', '--port', '32816'])
        preparation = json.loads((fresh / 'state/preparation.json').read_text())
        for row in preparation['scripts']:
            original = (cold / row['source']).read_text()
            expected = original
            for old, new in preparation['substitutions']:
                expected = expected.replace(old, new)
            actual = (fresh / row['generated']).read_text()
            assert actual == expected
            assert '/shared_nfs/yihou/playground/glm52_fake_tp4ep4_10k500_20260910-0452' not in actual
            file = fresh / row['generated']
            if file.suffix == '.sh':
                command(['bash', '-n', str(file)])
            else:
                ast.parse(actual)
        assert not list((fresh / 'cache').iterdir())
        command([sys.executable, str(helper), str(fresh)], expected=2)
        command([sys.executable, str(helper), str(cold / 'forbidden')], expected=2)
        command([sys.executable, str(helper), '/shared_nfs/yihou/playground/glm52_fake_tp4ep4_10k500_20260910-0452/forbidden-audit'], expected=2)
        assert not (cold / 'forbidden').exists()
        results['cold_relocation_fresh_only_and_protected_root_rejection'] = True
    results['metrics'] = compute(KIT)
    # No real secrets expected. Use conservative recognizable credential patterns;
    # arbitrary generated benchmark text is retained unchanged.
    patterns = [r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{50,}',
                r'AKIA[0-9A-Z]{16}', r'(?i)bearer\s+[A-Za-z0-9_.-]{24,}']
    hits = []
    for row in manifest['records']:
        data = (KIT / row['destination']).read_bytes()
        if row['compression'] == 'gzip':
            data = gzip.decompress(data)
        text = data.decode(errors='replace')
        if any(re.search(pattern, text) for pattern in patterns):
            hits.append(row['destination'])
    assert not hits, hits
    results['recognizable_secret_pattern_hits'] = hits
    for point in ['c16', 'c32']:
        config = json.loads((KIT / 'rounds' / point / 'container-inspect.json').read_text())[0]['Config']
        suspicious = [entry.split('=', 1)[0] for entry in config['Env']
                      if re.search(r'(?i)(password|secret|api.?key|credential|auth.?token)', entry.split('=', 1)[0])]
        assert not suspicious, suspicious
    results['proposal_runtime_test'] = 'NOT RERUN: local Python lacks Torch/SGLang; verbatim 4-case CPU test supplied for exact image. Patch-byte and syntax checks passed.'
    if before_manifest:
        results['package_checksums'] = 'Deferred until manifest freeze; rerun without --before-manifest'
    else:
        entries = {}
        for line in (KIT / 'MANIFEST.sha256').read_text().splitlines():
            digest, name = line.split('  ', 1)
            entries[name] = digest
            assert sha((KIT / name).read_bytes()) == digest, name
        actual = {str(p.relative_to(KIT)) for p in KIT.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256' and '__pycache__' not in p.parts}
        assert actual == set(entries), actual.symmetric_difference(entries)
        results['package_checksums'] = {'verified_files': len(entries), 'extra_or_missing': 0}
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-manifest', action='store_true')
    parser.add_argument('--verify-sources', action='store_true', help='Optional read-only comparison against original source paths')
    args = parser.parse_args()
    print(json.dumps(audit(args.before_manifest, args.verify_sources), indent=2))
