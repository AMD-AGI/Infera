import torch
from vllm.v1.kv_offload.cpu import gpu_worker as w
torch.cuda.init()
t, p = w._exact_pinned_host_tensor(5000, 1 << 20)
print("shape", tuple(t.shape), "pinned", t.is_pinned(), "fns", w._hip_host_alloc_fns() is not None)
g = torch.randint(-100, 100, (1 << 20,), dtype=torch.int8, device="cuda")
t[4999].copy_(g); torch.cuda.synchronize()
print("round trip", bool(torch.equal(t[4999].to("cuda"), g)))
del t
print("free rc", w._hip_host_alloc_fns()[1](p))
