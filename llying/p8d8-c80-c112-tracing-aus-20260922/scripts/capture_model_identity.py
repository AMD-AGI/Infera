#!/usr/bin/env python3
"""Record model/tokenizer metadata before a reusable baseline starts."""
import hashlib
import json
import os
from pathlib import Path
import sys

root=Path(os.environ['MODEL'])
result={}
for name in ['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json']:
 p=root/name
 if not p.exists():
  if name!='model.safetensors.index.json':raise SystemExit(f'Missing required metadata: {p}')
  continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 result[name]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
print('Model metadata recorded:',sys.argv[1])
