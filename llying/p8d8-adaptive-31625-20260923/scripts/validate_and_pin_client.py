#!/usr/bin/env python3
"""Run the existing runtime validator, then constrain pre-benchmark dependency resolution."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
out=Path(sys.argv[1]);root=Path(os.environ['TRACE_RUNTIME']);constraints=Path(os.environ['AGENTX_CLIENT_CONSTRAINTS'])
assert constraints.is_absolute() and constraints.is_file() and '\n' not in str(constraints)
subprocess.run([sys.executable,str(root/'scripts/validate_chunk8k_runtime.py'),str(out)],check=True)
p=out/'runtime.env';lines=p.read_text().splitlines();assert not any(x.startswith('UV_CONSTRAINT=') for x in lines)
p.write_text('\n'.join(lines)+f'\nUV_CONSTRAINT={constraints}\n')
(out/'client-dependency-constraints.json').write_text(json.dumps({'path':str(constraints),'sha256':hashlib.sha256(constraints.read_bytes()).hexdigest(),'scope':'uv dependency installation before benchmark; no request/scheduling changes'},indent=2)+'\n')
print('Pinned pre-benchmark dependency resolution to recorded A0 versions')
