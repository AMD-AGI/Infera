Fresh isolated S10 measurements at `aed9bc6` on one allocated MI355X GPU (PyTorch 2.9.1 / ROCm 7.2), compared with upstream main `8a455f1`. Actual GPU regression also passed: 3 tests + 16 subtests.

This measures only the metadata-cache/kernel wrapper, with row indices already on device: 20 layers, 576 bytes/row, 20 alternating-order batches of 10 calls after warmup; wall time includes the final GPU synchronization and Python/metadata costs. Outputs were byte-exact before timing.

| Gather rows | Main median us/call | Cached median us/call |
| --- | ---: | ---: |
| 1 | 39.57 | 17.15 |
| 777 | 39.09 | 17.07 |
| 8192 | 130.42 | 107.52 |

This is one-device synthetic gather evidence, not serving throughput, not a measure of stream priority or pinned-index transfer, and not a PP-ordering/accuracy test. No MTP simulation is involved in this microbenchmark. The PR remains draft for the remaining distributed ordering, CUDA, integration-performance and upstream-CI checks.

<details><summary>Reproducer (run from this PR checkout)</summary>

Export the baseline next to the script as `s10-baseline-gather.py`:

```bash
git show 8a455f1:python/sglang/kernels/ops/kvcache/pd_dcp_gather.py > s10-baseline-gather.py
PYTHONPATH=python python s10-metadata-bench.py
```

`s10-metadata-bench.py`:

```python
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch
from sglang.kernels.ops.kvcache import pd_dcp_gather as fixed

spec = importlib.util.spec_from_file_location('s10_baseline', Path(__file__).with_name('s10-baseline-gather.py'))
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
rows = 16384
widths = [576] * 20
sources = [torch.randint(0, 256, (rows, width), dtype=torch.uint8, device='cuda') for width in widths]
pointers = [x.data_ptr() for x in sources]
cache = fixed.MlaPackMetadataCache()
results = []
for count in (1, 777, 8192):
    indices = torch.arange(count, dtype=torch.int64, device='cuda') * 2
    old = torch.empty(count * sum(widths), dtype=torch.uint8, device='cuda')
    new = torch.empty_like(old)
    def before():
        baseline.copy_mla_rows_into_pack(pointers, indices, old, widths)
    def after():
        fixed.copy_mla_rows_into_pack(pointers, indices, new, widths, metadata_cache=cache)
    before(); after(); torch.cuda.synchronize()
    torch.testing.assert_close(old, new, rtol=0, atol=0)
    samples = {'before': [], 'after': []}
    for fn in (before, after):
        for _ in range(10): fn()
    for repeat in range(20):
        order = [('before', before), ('after', after)]
        if repeat % 2: order.reverse()
        for label, fn in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10): fn()
            torch.cuda.synchronize()
            samples[label].append((time.perf_counter() - start) * 1e6 / 10)
    results.append({'rows': count, 'layers': 20, 'bytes_per_row': 576, 'median_wall_us_per_call': {k: statistics.median(v) for k,v in samples.items()}, 'samples': samples})
print(json.dumps({'torch': torch.__version__, 'hip': torch.version.hip, 'gpu': torch.cuda.get_device_name(), 'results': results}, indent=2))

```
</details>
