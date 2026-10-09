#!/usr/bin/env python3
"""Read safetensors JSON headers only; account synthetic checkpoint bytes."""
import collections
import json
import re
import struct
from pathlib import Path

W=Path(__file__).resolve().parents[1]
ROOT=Path('/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4')
OUT=W/'rounds/001-component/expert8'
layer_re=re.compile(r'^model\.layers\.(\d+)\.')
expert_re=re.compile(r'\.mlp\.experts\.(\d+)\.')
categories=collections.defaultdict(lambda:dict(original_bytes=0,retained_bytes=0,discarded_bytes=0,tensors=0,retained_tensors=0))
shards=0
for path in sorted(ROOT.glob('*.safetensors')):
 with path.open('rb') as f:
  size=struct.unpack('<Q',f.read(8))[0]
  header=json.loads(f.read(size))
 shards+=1
 for name,meta in header.items():
  if name=='__metadata__':continue
  size=meta['data_offsets'][1]-meta['data_offsets'][0]
  layer=layer_re.match(name);expert=expert_re.search(name)
  is_draft=layer is not None and int(layer.group(1))>=78
  if is_draft:category='excluded_draft'
  elif expert:category='target_routed_moe'
  elif '.mlp.gate.' in name:category='target_router_gate'
  elif '.mlp.shared_experts.' in name:category='target_shared_moe'
  elif '.self_attn.' in name:category='target_attention'
  elif '.mlp.' in name:category='target_dense_mlp'
  else:category='target_embedding_norm_head_other'
  keep=not is_draft and (not expert or int(expert.group(1))<8)
  retained=size if keep else 0
  if keep and name.endswith(('.mlp.gate.weight','.mlp.gate.e_score_correction_bias')):
   assert meta['shape'][0]==256,(name,meta)
   assert size%256==0
   retained=size//256*8
  c=categories[category];c['original_bytes']+=size;c['retained_bytes']+=retained
  c['discarded_bytes']+=size-retained;c['tensors']+=1;c['retained_tensors']+=int(keep)

routed=categories['target_routed_moe'];gate=categories['target_router_gate']
original=sum(v['original_bytes'] for v in categories.values())
retained=sum(v['retained_bytes'] for v in categories.values())
report={'method':'Safetensors header data_offsets and shapes only; no tensor materialization',
 'checkpoint':str(ROOT),'shards':shards,'categories':dict(categories),
 'total_original_bytes_including_draft':original,'total_retained_target_checkpoint_bytes':retained,
 'total_discarded_bytes_including_draft':original-retained,
 'theoretical_ep4_comparison':{
  'routed_original_per_rank_bytes':routed['original_bytes']/4,
  'routed_retained_per_rank_bytes':routed['retained_bytes']/4,
  'routed_saved_per_rank_bytes':routed['discarded_bytes']/4,
  'router_gate_saved_per_rank_bytes_replicated':gate['discarded_bytes'],
  'routed_plus_gate_saved_per_rank_bytes':routed['discarded_bytes']/4+gate['discarded_bytes'],
  'note':'Original256EP4 vs synthetic8EP4 target only; routed experts evenly partitioned, router gate replicated. Excludes runtime padding, repacking, dtype conversions, temporary buffers and graph overhead. Shared/dense/attention remain unchanged.'}}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'memory-accounting.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
