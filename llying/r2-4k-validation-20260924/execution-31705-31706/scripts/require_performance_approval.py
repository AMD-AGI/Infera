#!/usr/bin/env python3
"""Reject performance runs unless the user-approved config and binary still match."""
import hashlib,json,os
from pathlib import Path
root=Path(os.environ['TRACE_RUNTIME']);approval=root/'config/performance-approval.json'
if not approval.exists():raise SystemExit('Performance blocked: present the full config to the user and obtain explicit approval first.')
data=json.loads(approval.read_text());entry=data.get('approved_runs',{}).get(os.environ['RUN_ID'])
if not data.get('user_confirmation') or not entry:raise SystemExit('This run has no recorded user configuration approval.')
paths=[Path(os.environ['CONFIG']),root/'config/common.sh',root/'config/performance.sh',root/'config/baseline-template.sh',root/'config/topology.tsv',root/'config/client-constraints.txt']
actual={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
if actual!=entry.get('config_sha256'):raise SystemExit('Configuration changed since user approval.')
binary=Path(os.environ['ROUTER_BINARY_OVERRIDE'])
if hashlib.sha256(binary.read_bytes()).hexdigest()!=entry.get('router_sha256'):raise SystemExit('Router binary changed since user approval.')
print('USER_APPROVED_PERFORMANCE_CONFIG_VERIFIED')
