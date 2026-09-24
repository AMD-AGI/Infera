#!/usr/bin/env python3
import json,os,urllib.request
from pathlib import Path
r=Path(os.environ['RUN']);out=r/'final-server-info';out.mkdir(exist_ok=True)
for role,port in [('prefill',29001),('decode',29002)]:
 with urllib.request.urlopen(f"http://{os.environ[role.upper()+'_IP']}:{port}/get_server_info",timeout=20) as f:d=json.load(f)
 (out/(role+'.json')).write_text(json.dumps(d)+'\n')
