#!/usr/bin/env python3
"""Capture only this case's diagnostic suffix from reused P/D processes."""
import base64,fcntl,json,os,shlex,subprocess,tarfile,time
from pathlib import Path
r=Path(os.environ['RUN']);prefix=os.environ['CONTAINER_PREFIX'];opts=shlex.split(os.environ['SSH_OPTS'])
lock=(r/'.capture.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
def sealed():
 p=r/'STATUS'
 return p.exists() and 'COMPLETE_REVIEW_PENDING' in p.read_text()
if sealed():raise SystemExit(0)
live_path=r/'live-containers.json'
live=json.loads(live_path.read_text()) if live_path.exists() else {}
cursor_path=r/'snapshot/diagnostic-cursors.json'
cursors=json.loads(cursor_path.read_text()) if cursor_path.exists() else {}
start_path=r/'snapshot/capture-start-epoch.txt'
since=start_path.read_text().strip() if start_path.exists() else '0'
PROBE=r'''
import base64,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]);cursors=json.loads(base64.b64decode(sys.argv[2]))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for p in sorted(root.glob('*.jsonl')):
  st=p.stat();entry=cursors.get(p.name);start=entry['size'] if entry else 0
  if entry and (st.st_ino!=entry['inode'] or st.st_size<start):raise RuntimeError(f'diagnostic file rotated or truncated: {p}')
  with p.open('rb') as f:f.seek(start);data=f.read(st.st_size-start)
  if data and not data.endswith(b'\n'):data=data[:data.rfind(b'\n')+1]
  info=tarfile.TarInfo(p.name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
'''
(r/'server-logs').mkdir(parents=True,exist_ok=True)
for role in ('prefill','decode'):
 node=os.environ[f'{role.upper()}_NODE']
 srcid=os.environ.get(f'DIAG_SOURCE_{role.upper()}',os.environ.get('SERVICE_RUN_ID',os.environ['RUN_ID']))
 source=f'/tmp/aus-diag-{srcid}/{role}'
 encoded=base64.b64encode(json.dumps(cursors.get(role,{})).encode()).decode()
 cmd=['ssh',*opts,node,shlex.join(['python3','-',source,encoded])]
 dest=r/'diagnostics'/role;dest.mkdir(parents=True,exist_ok=True)
 with subprocess.Popen(['timeout','60s',*cmd],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as p:
  p.stdin.write(PROBE.encode());p.stdin.close()
  with tarfile.open(fileobj=p.stdout,mode='r|') as archive:
   for entry in archive:
    if not entry.isfile() or Path(entry.name).name!=entry.name or not entry.name.endswith('.jsonl'):raise RuntimeError('Unsafe diagnostic member')
    data=archive.extractfile(entry).read();tmp=dest/(entry.name+f'.{os.getpid()}.tmp');tmp.write_bytes(data)
    if sealed():tmp.unlink()
    else:tmp.replace(dest/entry.name)
  err=p.stderr.read().decode();code=p.wait()
  if code:raise RuntimeError(err)
 tmp=r/f'server-logs/{role}.{os.getpid()}.tmp'
 with tmp.open('w') as f:
  subprocess.run(['ssh',*opts,node,shlex.join(['docker','logs','--timestamps','--since',since,live.get(role,{}).get('Id',f'{prefix}-{role}-0')])],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60)
 if sealed():tmp.unlink()
 else:tmp.replace(r/f'server-logs/{role}.log')
router_info=r/'snapshot/router-mode-validation.json'
router_id=json.loads(router_info.read_text())['container']['Id'] if router_info.exists() else prefix+'-router'
tmp=r/f'server-logs/router.{os.getpid()}.tmp'
with tmp.open('w') as f:
 subprocess.run(['docker','logs','--timestamps',router_id],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60)
if sealed():tmp.unlink()
else:
 tmp.replace(r/'server-logs/router.log')
 (r/'last-capture.txt').write_text(str(time.time())+'\n')
