"""Focused tests for physical expert filtering and all-eight GPU routing."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from sglang.srt.models import yihou_expert8 as e8

W=Path(__file__).resolve().parents[1]
OUT=W/'rounds/001-component/expert8'
OUT.mkdir(parents=True,exist_ok=True)
config=SimpleNamespace(n_routed_experts=8,num_experts_per_tok=8,n_group=1,topk_group=1,scoring_func='sigmoid',yihou_synthetic_expert8=True)
assert e8.enabled(config)
assert not e8.enabled(SimpleNamespace())
weights={
 'model.layers.3.mlp.gate.weight':torch.arange(256*4,dtype=torch.float32).reshape(256,4),
 'model.layers.3.mlp.gate.e_score_correction_bias':torch.arange(256,dtype=torch.float32),
 'model.layers.3.mlp.experts.0.gate_proj.weight':torch.ones(3,4),
 'model.layers.3.mlp.experts.7.gate_proj.weight':torch.ones(3,4)*7,
 'model.layers.3.mlp.experts.8.gate_proj.weight':torch.ones(3,4)*8,
 'model.layers.3.mlp.experts.255.gate_proj.weight':torch.ones(3,4)*255,
 'model.layers.3.mlp.shared_experts.gate_proj.weight':torch.ones(3,4)*9,
 'model.layers.78.mlp.experts.0.gate_proj.weight':torch.ones(3,4)*78,
}
f=OUT/'tiny-checkpoint.yihou.safetensors'
save_file(weights,str(f))
seen=[]
real_open=e8.safe_open
class Tracked:
 def __init__(self,*a,**k):
  assert k['device']=='cpu';self.handle=real_open(*a,**k)
 def __enter__(self):self.handle.__enter__();return self
 def __exit__(self,*a):return self.handle.__exit__(*a)
 def keys(self):return self.handle.keys()
 def get_tensor(self,name):seen.append(name);return self.handle.get_tensor(name)
with patch.object(e8,'safe_open',Tracked):
 loaded=dict(e8.weights_iterator([str(f)],78))
assert set(seen)==set(loaded)
assert all(e8.keep_weight(n,78) for n in seen)
assert len(loaded)==5
assert torch.equal(loaded['model.layers.3.mlp.gate.weight'],weights['model.layers.3.mlp.gate.weight'][:8])
assert loaded['model.layers.3.mlp.gate.e_score_correction_bias'].shape==(8,)

for n in [0,1,32,128]:
 logits=torch.randn(n,8,device='cuda')
 h=torch.empty(n,4,device='cuda')
 w,ids=e8.route_all_eight(h,logits,8,True)
 assert ids.shape==(n,8) and ids.dtype==torch.int32
 assert torch.equal(ids,torch.arange(8,device='cuda',dtype=torch.int32).expand(n,8))
 assert torch.isfinite(w).all()
 assert torch.allclose(w.sum(-1),torch.ones(n,device='cuda'),atol=1e-6)
 reference=logits.float().sigmoid();reference/=reference.sum(-1,keepdim=True)
 assert torch.allclose(w,reference,atol=1e-6)

logits=torch.randn(32,8,device='cuda');h=torch.empty(32,4,device='cuda')
stream=torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
 for _ in range(3):e8.route_all_eight(h,logits,8,True)
torch.cuda.current_stream().wait_stream(stream)
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):graph_w,graph_ids=e8.route_all_eight(h,logits,8,True)
logits.normal_();graph.replay();torch.cuda.synchronize()
expected,_=e8.route_all_eight(h,logits,8,True)
assert torch.allclose(graph_w,expected)
assert torch.equal(graph_ids,torch.arange(8,device='cuda',dtype=torch.int32).expand(32,8))
report={'checkpoint_filter_before_get_tensor':True,'gate_slices_match':True,'retained_tensor_count':len(loaded),'gpu_eager_sizes':[0,1,32,128],'cuda_graph_replay':True,'routed_ids':list(range(8))}
(OUT/'tests.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
