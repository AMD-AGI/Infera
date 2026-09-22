#!/usr/bin/env python3
"""Copy installed AIPerf into a client-local overlay and add correlation rid.

The pinned checkout remains read-only. Only the JSON metadata rid is changed;
request text, scheduling, token budgets and benchmark records are unchanged.
"""
import ast
import hashlib
import shutil
from pathlib import Path
import aiperf

source = Path(aiperf.__file__).parent
target = Path('/tmp/aus-client-overlay/aiperf')
shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__'))
path = target/'transports/aiohttp_transport.py'
text = path.read_text()
old = '            url = self.build_url(request_info)\n'
assert text.count(old)==1
text = text.replace(old,
    '            # AUS diagnostic correlation only; preserve the workload.\n'
    '            if request_info.x_request_id:\n'
    '                payload = orjson.loads(payload) if isinstance(payload, bytes) else dict(payload)\n'
    '                payload["rid"] = request_info.x_request_id\n' + old)
ast.parse(text)
path.write_text(text)
print('AUS client correlation overlay:', hashlib.sha256(text.encode()).hexdigest(), flush=True)
