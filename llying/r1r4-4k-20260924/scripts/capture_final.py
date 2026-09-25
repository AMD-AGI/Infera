#!/usr/bin/env python3
"""Capture once after replay; later offline steps reuse the saved identities."""
import datetime,json,os,urllib.request
from pathlib import Path
r=Path(os.environ['RUN']);out=r/'final-server-info';out.mkdir(exist_ok=True)
for role,port in [('prefill',29001),('decode',29002)]:
 target=out/(role+'.json')
 if target.exists():
  json.loads(target.read_text())
  continue
 with urllib.request.urlopen(f"http://{os.environ[role.upper()+'_IP']}:{port}/get_server_info",timeout=20) as f:d=json.load(f)
 tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(d)+'\n');tmp.replace(target)
 (out/(role+'-captured-at.txt')).write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n')
