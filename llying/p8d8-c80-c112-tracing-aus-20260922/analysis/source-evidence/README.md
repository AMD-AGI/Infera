# Source evidence for the 2026-09-23 offline RCA

These are selected source files read from the saved Docker image archive:

`/perf_apps/liyingli/bench_agentx/p8d8-tracing-aus-20260922/diagnostic-image.tar`

`manifest.json` records the image config, containing layer and SHA256 of every
saved source file. Main source selection searched layers newest first; the
metrics reporter and memory pool were subsequently read from the layer containing
scheduler.py. The `missing` list records initial optional filenames not found;
`managers/scheduler_components/metrics_reporter.py` is the actual reporter path.
This is a selected evidence snapshot, not an installable SGLang checkout.

The running experiment additionally bind-mounted `aus_diag.py`; its executed
copy remains in RUN/live-diag-helper.py. The image's timing/admission code and
RUN's effective server configuration are used together in the report.

To verify saved files from this directory:

```python
import hashlib, json
from pathlib import Path
for entry in json.loads(Path('manifest.json').read_text())['files']:
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256']
```

The router discussion references the Infera repository sources, not a recovered
router binary/source match. This provenance limitation is explicit in the report.
