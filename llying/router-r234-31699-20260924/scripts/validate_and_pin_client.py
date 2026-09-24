#!/usr/bin/env python3
"""Compare equivalent pinning stages, preserving the reference dependency lock."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
out=Path(sys.argv[1]);root=Path(os.environ['TRACE_RUNTIME']);constraints=Path(os.environ['AGENTX_CLIENT_CONSTRAINTS'])
assert constraints.is_absolute() and constraints.is_file() and '\n' not in str(constraints)
p=out/'runtime.env';lines=p.read_text().splitlines();assert not any(x.startswith('UV_CONSTRAINT=') for x in lines)
digest=hashlib.sha256(constraints.read_bytes()).hexdigest()
base=Path(os.environ['BASELINE_RUN'])/'c80'
base_pins=[x.split('=',1)[1] for x in (base/'runtime.env').read_text().splitlines() if x.startswith('UV_CONSTRAINT=')]
assert len(base_pins)<=1, 'Duplicate reference dependency constraint'
def pin():p.write_text('\n'.join(lines)+f'\nUV_CONSTRAINT={constraints}\n')
if base_pins:
    # Newer reference runs have already passed this wrapper and contain the pin.
    recorded=json.loads((base/'client-dependency-constraints.json').read_text())
    assert base_pins==[str(constraints)] and recorded['path']==str(constraints), 'Reference dependency constraint path changed'
    assert recorded['sha256']==digest, 'Reference dependency constraint contents changed'
    pin()
subprocess.run([sys.executable,str(root/'scripts/validate_chunk8k_runtime.py'),str(out)],check=True)
if not base_pins:pin()
(out/'client-dependency-constraints.json').write_text(json.dumps({'path':str(constraints),'sha256':digest,'scope':'uv dependency installation before benchmark; no request/scheduling changes'},indent=2)+'\n')
print('Pinned pre-benchmark dependency resolution to recorded A0 versions')
