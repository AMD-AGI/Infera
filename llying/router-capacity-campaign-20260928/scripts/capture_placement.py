"""Capture each worker separately so local DP ranks and container PIDs cannot collide."""
import fcntl,json,os,shlex,subprocess,tarfile
from pathlib import Path

E=os.environ;RUN=Path(E['RUN']);rows=json.loads((RUN/'placement-resolved.json').read_text())
lock=(RUN/'.capture.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
since=int((RUN/'snapshot/capture-start-epoch.txt').read_text())
SSH=['ssh',*shlex.split(E['SSH_OPTS'])]
PROBE='''import io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]);start=int(sys.argv[2])*1000000000
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as out:
 for p in sorted(root.glob('*.jsonl')):
  chunks=[]
  for line in p.open('rb'):
   if not line.endswith(b'\\n'):break
   row=json.loads(line)
   if row.get('wall_ns',0)>=start:chunks.append(line)
  data=b''.join(chunks);item=tarfile.TarInfo(p.name);item.size=len(data)
  out.addfile(item,io.BytesIO(data))
'''

def sealed():
    p=RUN/'STATUS'
    return p.exists() and 'COMPLETE_REVIEW_PENDING' in p.read_text()

if sealed():raise SystemExit(0)
(RUN/'server-logs').mkdir(exist_ok=True)
for w in rows:
    dest=RUN/'diagnostics'/w['role'];dest.mkdir(parents=True,exist_ok=True)
    cmd=SSH+[w['node'],shlex.join(['python3','-c',PROBE,w['diag_dir'],str(since)])]
    with subprocess.Popen(['timeout','90s',*cmd],stdout=subprocess.PIPE,stderr=subprocess.PIPE) as child:
        with tarfile.open(fileobj=child.stdout,mode='r|') as archive:
            for entry in archive:
                if not entry.isfile() or Path(entry.name).name!=entry.name:raise ValueError('invalid diagnostic member')
                data=[]
                for line in archive.extractfile(entry):
                    event=json.loads(line);event['source_worker']=w['instance'];event['source_node']=w['node'];data.append(json.dumps(event,separators=(',',':'))+'\n')
                target=dest/(w['instance']+'-'+entry.name);tmp=target.with_suffix('.tmp');tmp.write_text(''.join(data))
                if sealed():tmp.unlink()
                else:tmp.replace(target)
        error=child.stderr.read().decode();code=child.wait(timeout=60)
        if code:raise RuntimeError(error)
    target=RUN/'server-logs'/(w['instance']+'.log');tmp=target.with_suffix('.tmp')
    with tmp.open('w') as f:
        subprocess.run(SSH+[w['node'],shlex.join(['docker','logs','--timestamps','--since',str(since),w['container_id']])],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=90)
    if sealed():tmp.unlink()
    else:tmp.replace(target)
with (RUN/'server-logs/router.log').open('w') as f:
    subprocess.run(SSH+[E['PREFILL_NODE'],shlex.join(['docker','logs','--timestamps',E['CONTAINER_PREFIX']+'-router'])],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=90)
