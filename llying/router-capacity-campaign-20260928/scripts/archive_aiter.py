"""Archive completed libraries from the unchanged base ABI for third-node preparation."""
import hashlib,json,tarfile,time
from pathlib import Path
root=Path('/tmp/aiter-jit-100078/aec8fd47d0beca6300a5fcb581bce44b0273008d0633d01f4c4be269fa1080a0')
out=Path('/perf_apps/liyingli/bench_agentx/router-capacity-20260928/artifacts')
files=[p for p in root.iterdir() if p.is_file() and p.suffix in ('.so','.json','.csv') and time.time()-p.stat().st_mtime>60]
manifest={}
with tarfile.open(out/'aiter-prefill-cache.tar','w',dereference=True) as archive:
    for p in sorted(files):
        archive.add(p,arcname=p.name);manifest[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(out/'aiter-prefill-cache.json').write_text(json.dumps({'base_image':'4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb','source_image':'aec8fd47d0beca6300a5fcb581bce44b0273008d0633d01f4c4be269fa1080a0','files':manifest},indent=2)+'\n')
print('Archived',len(files),'files,',sum(v['bytes'] for v in manifest.values()),'bytes')
