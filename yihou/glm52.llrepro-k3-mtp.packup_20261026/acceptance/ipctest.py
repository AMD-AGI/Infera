import os, torch, torch.distributed as dist, torch.multiprocessing as mp
def f(r, n):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29911")
    dist.init_process_group("nccl", rank=r, world_size=n); torch.cuda.set_device(r)
    x = torch.ones(1 << 20, device="cuda") * (r + 1); dist.all_reduce(x)
    print(r, "allreduce ok", x[0].item(), flush=True); dist.destroy_process_group()
if __name__ == "__main__":
    mp.spawn(f, args=(4,), nprocs=4)
