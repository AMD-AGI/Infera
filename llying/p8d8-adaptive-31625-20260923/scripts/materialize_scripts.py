#!/usr/bin/env python3
"""Reconstruct the used runtime scripts in a new directory from tracked sources."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);a=p.parse_args();repo=Path(__file__).resolve().parents[3]
if a.output.exists():raise SystemExit('Refuse: output exists; use a new directory')
plan=[]
for entry in json.loads(a.manifest.read_text())['files']:
 if entry.get('exclude_reason'):continue
 sources=entry['repository_sources'];assert sources
 source=repo/sources[0];dest=Path(entry['runtime_relative_path']);assert not dest.is_absolute() and '..' not in dest.parts
 if hashlib.sha256(source.read_bytes()).hexdigest()!=entry['sha256']:raise SystemExit(f'Source drift: {source}; restore the recorded commit or deliberately refresh the manifest')
 plan.append((source,a.output/dest))
for source,dest in plan:
 dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
print(f'Materialized {len(plan)} verified scripts at {a.output}; no services launched')
