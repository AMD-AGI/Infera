import subprocess
import threading
import time

import torch

dev = torch.device("cuda")
a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev)
samples = []
stop = False


def sampler():
    while not stop:
        out = subprocess.run(
            ["amd-smi", "metric", "-g", "0", "-c", "-p", "--csv"], capture_output=True, text=True
        ).stdout
        samples.append(out)
        time.sleep(0.5)


th = threading.Thread(target=sampler)
for _ in range(20):
    a @ a
torch.cuda.synchronize()
th.start()
t0 = time.time()
n = 0
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record()
while time.time() - t0 < 12:
    for _ in range(50):
        a @ a
    n += 50
    torch.cuda.synchronize()
e.record()
torch.cuda.synchronize()
stop = True
th.join()
ms = s.elapsed_time(e) / n
print(f"sustained 8192^3: {2 * 8192**3 / ms / 1e9:.0f} TFLOP/s over {n} GEMMs")
print(samples[len(samples) // 2][:3000])
