#!/usr/bin/env python3
"""Preserve completed AITER libraries across preemption; never copy build locks."""
import argparse,hashlib,io,json,os,tarfile,time
from pathlib import Path
IMAGE='4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb'
def snapshot(cache,bundle):
 assert cache.name==IMAGE,'Cache must be keyed by the verified image ID'
 if bundle.exists():raise ValueError('Refuse to overwrite an existing bundle')
 entries=[];skipped=[]
 for p in sorted(cache.glob('*.so')):
  lock=cache/'build'/('lock_'+p.stem)
  if p.is_symlink() or lock.exists():skipped.append({'name':p.name,'reason':'symlink_or_build_lock'});continue
  before=p.stat();data=p.read_bytes();after=p.stat()
  if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns) or lock.exists():skipped.append({'name':p.name,'reason':'changed_during_read'});continue
  if not data.startswith(b'\x7fELF'):skipped.append({'name':p.name,'reason':'not_elf'});continue
  entries.append((p.name,data))
 if not entries:raise ValueError('No completed libraries to preserve')
 manifest={'image_id':IMAGE,'source_cache':str(cache),'created_epoch':time.time(),'libraries':{n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in entries},'skipped':skipped}
 bundle.parent.mkdir(parents=True,exist_ok=True);tmp=bundle.with_name(bundle.name+f'.{os.getpid()}.tmp')
 with tarfile.open(tmp,'w') as tar:
  for name,data in [('manifest.json',json.dumps(manifest,indent=2).encode()),*entries]:
   info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;tar.addfile(info,io.BytesIO(data))
 tmp.replace(bundle);return manifest

def install(bundle,cache):
 assert cache.name==IMAGE
 if cache.exists():raise ValueError('Use a fresh per-case cache directory; never overwrite live libraries')
 with tarfile.open(bundle,'r') as tar:
  names=tar.getnames();manifest=json.load(tar.extractfile('manifest.json'));assert manifest['image_id']==IMAGE
  if len(names)!=len(set(names)) or set(names)!={'manifest.json',*manifest['libraries']}:raise ValueError('Unexpected or duplicate bundle members')
  contents=[]
  for name,meta in manifest['libraries'].items():
   if Path(name).name!=name or not name.endswith('.so'):raise ValueError('Unsafe library name')
   member=tar.getmember(name)
   if not member.isfile():raise ValueError('Library member must be a regular file')
   data=tar.extractfile(member).read()
   if len(data)!=meta['bytes'] or hashlib.sha256(data).hexdigest()!=meta['sha256'] or not data.startswith(b'\x7fELF'):raise ValueError('Library content mismatch')
   contents.append((name,data))
 cache.mkdir(parents=True)
 for name,data in contents:(cache/name).write_bytes(data)
 (cache/'seed-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return {'installed':len(contents),'cache':str(cache),'bundle':str(bundle)}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['snapshot','install']);p.add_argument('cache',type=Path);p.add_argument('bundle',type=Path);a=p.parse_args();print(json.dumps(snapshot(a.cache,a.bundle) if a.action=='snapshot' else install(a.bundle,a.cache)))
if __name__=='__main__':main()
