# R054 microbench: GPU-side latency for a cross-stream event dependency on an otherwise idle GPU.
# Stream A: spin ~SPIN_MS, then record ev_a_done. Stream B (enqueued BEFORE A finishes): wait_event(ev_a_done),
# record ev_b_start, tiny kernel. Latency = elapsed(ev_a_done -> ev_b_start) on the GPU clock.
# Env: GPU_MAX_HW_QUEUES (set by caller), XS_POOL=1 creates torch's pool streams first (like sglang),
#      XS_SAME=1 puts A and B on the same torch stream object pair? no: uses two streams from the pool.
import os, statistics as st, time
import torch

dev = torch.device("cuda:0")
torch.cuda.set_device(dev)
SPIN_MS = float(os.environ.get("XS_SPIN_MS", "5"))
REPS = int(os.environ.get("XS_REPS", "30"))
if os.environ.get("XS_POOL", "1") == "1":
    pool = [torch.cuda.Stream() for _ in range(8)]  # touches torch's stream pool (32 per priority, lazily)
    a, b = pool[0], pool[1]
else:
    a, b = torch.cuda.Stream(), torch.cuda.Stream()
x = torch.zeros(1 << 20, device=dev)
# calibrate spin cycles
torch.cuda.synchronize()
t0 = time.time(); torch.cuda._sleep(int(1e7)); torch.cuda.synchronize(); per_cycle = (time.time() - t0) / 1e7
cycles = int(SPIN_MS / 1e3 / per_cycle)
lat = []; lat_same = []
for i in range(REPS + 3):
    ea = torch.cuda.Event(enable_timing=True); eb = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(a):
        torch.cuda._sleep(cycles); x.add_(1); ea.record(a)
    with torch.cuda.stream(b):
        b.wait_event(ea); eb.record(b); x.mul_(1.0)
    torch.cuda.synchronize()
    if i >= 3:
        lat.append(ea.elapsed_time(eb) * 1e3)
    # same-stream reference: spin then record two events on A
    e1 = torch.cuda.Event(enable_timing=True); e2 = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(a):
        torch.cuda._sleep(cycles); x.add_(1); e1.record(a); x.mul_(1.0); e2.record(a)
    torch.cuda.synchronize()
    if i >= 3:
        lat_same.append(e1.elapsed_time(e2) * 1e3)
print(f"RESULT hwq={os.environ.get('GPU_MAX_HW_QUEUES','<unset>')} pool={os.environ.get('XS_POOL','1')} spin_ms={SPIN_MS} "
      f"xstream_wait_us p50={st.median(lat):.0f} p90={sorted(lat)[int(0.9*len(lat))]:.0f} max={max(lat):.0f} | "
      f"same_stream_us p50={st.median(lat_same):.0f} max={max(lat_same):.0f}")
