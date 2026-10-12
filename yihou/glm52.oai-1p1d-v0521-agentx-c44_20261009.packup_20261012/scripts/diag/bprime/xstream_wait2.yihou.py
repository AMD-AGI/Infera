# R055 microbench: GPU-side start latency of a stream that waits on another stream's event, measured on the GPU
# clock (hip events with timing): lat = elapsed(ev_a_done -> ev_b_start).
# Modes (all run in one process; env decides GPU_MAX_HW_QUEUES etc.):
#   idle   : A spins SPIN ms then records ev_a; B (enqueued while A runs) waits ev_a, records ev_b, tiny kernel.
#   third  : as idle, plus stream C running a long spin (LONG ms) that overlaps the whole A/B window.
#   decode : decode pattern. A = spin + pinned HtoD (64 KiB) + tiny kernel, record ev_a; B = wait ev_a, record ev_b,
#            tiny kernel, then a LATE spin and ev_late; the host thread blocks in ev_late.synchronize() (=
#            hipEventSynchronize on a later event) during the dependency wait.
# Streams come from torch's pool (like sglang). Prints one RESULT line per mode.
import os, statistics as st, time
import torch

dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
SPIN = float(os.environ.get("XS_SPIN_MS", "5")); LONG = float(os.environ.get("XS_LONG_MS", "30"))
LATE = float(os.environ.get("XS_LATE_MS", "20")); REPS = int(os.environ.get("XS_REPS", "25"))
pool = [torch.cuda.Stream() for _ in range(8)]
a, b, c = pool[0], pool[1], pool[2]
x = torch.zeros(1 << 16, device=dev); y = torch.zeros(1 << 16, device=dev)
h = torch.ones(1 << 14, dtype=torch.float32).pin_memory(); hd = torch.empty(1 << 14, device=dev)
torch.cuda.synchronize()
t0 = time.time(); torch.cuda._sleep(int(1e7)); torch.cuda.synchronize(); per_cycle = (time.time() - t0) / 1e7
cyc = lambda ms: int(ms / 1e3 / per_cycle)
print(f"streams a={a.cuda_stream:#x} b={b.cuda_stream:#x} c={c.cuda_stream:#x} cycles/ms={cyc(1)}")


def run(mode):
    lat, host_wait = [], []
    for i in range(REPS + 3):
        ea = torch.cuda.Event(enable_timing=True); eb = torch.cuda.Event(enable_timing=True)
        el = torch.cuda.Event()
        if mode == "third":
            with torch.cuda.stream(c):
                torch.cuda._sleep(cyc(LONG)); y.add_(1)
        with torch.cuda.stream(a):
            torch.cuda._sleep(cyc(SPIN))
            if mode == "decode":
                hd.copy_(h, non_blocking=True)
            x.add_(1); ea.record(a)
        with torch.cuda.stream(b):
            b.wait_event(ea); eb.record(b); x.mul_(1.0)
            if mode == "decode":
                torch.cuda._sleep(cyc(LATE)); el.record(b)
        if mode == "decode":
            t = time.time(); el.synchronize(); host_wait.append((time.time() - t) * 1e3)
        torch.cuda.synchronize()
        if i >= 3:
            lat.append(ea.elapsed_time(eb) * 1e3)
    s = sorted(lat)
    extra = f" host_sync_ms p50={st.median(host_wait):.1f}" if host_wait else ""
    print(f"RESULT mode={mode} hwq={os.environ.get('GPU_MAX_HW_QUEUES', '<4>')} intr={os.environ.get('HSA_ENABLE_INTERRUPT', '<1>')} "
          f"cpuwait={os.environ.get('ROC_CPU_WAIT_FOR_SIGNAL', '<1>')} actwait={os.environ.get('ROC_ACTIVE_WAIT_TIMEOUT', '<0>')} "
          f"lat_us p50={s[len(s)//2]:.0f} p90={s[int(0.9*len(s))]:.0f} max={s[-1]:.0f} min={s[0]:.0f}{extra}", flush=True)


for m in os.environ.get("XS_MODES", "idle,third,decode").split(","):
    run(m)
