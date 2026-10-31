"""Decode DP-attention MoE collectives on 4 GPUs, GPU kernel time from the profiler (eager, unregistered aiter buffers: includes the copy-in):
gather of each rank's rows [max_len, 6144] bf16 into [4*max_len, 6144] (RCCL all_gather vs aiter custom all-gather)
and the combine back (aiter all-reduce + local slice copy, as the 10-30 trace shows; vs aiter / RCCL reduce-scatter).
  torchrun --nproc-per-node 4 dp_comm_bench.py"""
import os, torch, torch.distributed as dist
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank); dev = torch.device("cuda", rank)
dist.init_process_group("nccl", device_id=dev)
gloo = dist.new_group(backend="gloo")
ca = CustomAllreduce(gloo, dev, max_size=512 * 1024 * 1024)
H, REP = 6144, 20


def graph_time(fn, warm=5, iters=50):
    """GPU kernel time per call (us), from the profiler, eager launches; max over ranks."""
    for _ in range(warm): fn()
    torch.cuda.synchronize(); dist.barrier()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters): fn()
        torch.cuda.synchronize()
    us = sum(e.device_time_total for e in prof.key_averages()) / iters
    t = torch.tensor([us], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


for ml in (8, 16, 32, 48, 64, 96, 128):
    inp = torch.randn(ml, H, dtype=torch.bfloat16, device=dev)
    glob = torch.empty(world * ml, H, dtype=torch.bfloat16, device=dev)
    loc = torch.empty(ml, H, dtype=torch.bfloat16, device=dev)
    r = {}
    r["ag_rccl"] = graph_time(lambda: dist.all_gather_into_tensor(glob, inp))
    r["ag_aiter"] = graph_time(lambda: ca.all_gather_unreg(inp, out=glob)) if ca.should_custom_ag(inp) else float("nan")
    def ar_slice():
        out = ca.all_reduce(glob, registered_input=False)
        loc.copy_(out[rank * ml:(rank + 1) * ml])
    r["ar_aiter+copy"] = graph_time(ar_slice)
    r["rs_aiter"] = graph_time(lambda: ca.reduce_scatter(glob, loc, registered=False))
    r["rs_rccl"] = graph_time(lambda: dist.reduce_scatter_tensor(loc, glob))
    # correctness of the aiter paths against RCCL
    ref_g = torch.empty_like(glob); dist.all_gather_into_tensor(ref_g, inp)
    ca_g = ca.all_gather_unreg(inp, out=torch.empty_like(glob))
    ref_l = torch.empty_like(loc); dist.reduce_scatter_tensor(ref_l, glob.copy_(ref_g))
    rs_l = torch.empty_like(loc); ca.reduce_scatter(glob, rs_l, registered=False)
    ok = torch.equal(ref_g, ca_g) and torch.allclose(ref_l.float(), rs_l.float(), rtol=2e-2, atol=2e-2)
    if rank == 0:
        mb = world * ml * H * 2 / 2**20
        print(f"max_len {ml:4d} (global {mb:5.2f} MiB): " + "  ".join(f"{k} {v:6.1f}us" for k, v in r.items()) + f"  match {ok}", flush=True)
dist.destroy_process_group()
