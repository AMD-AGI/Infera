#!/usr/bin/env python3
"""Pin the existing Weka dataset loader to the baseline revision in the overlay."""
import ast
import hashlib
from pathlib import Path
p = Path('/tmp/aus-client-overlay/aiperf/dataset/loader/semianalysis_cc_traces_weka.py')
s = p.read_text()
old = '    tag: ClassVar[str] = "SemiAnalysisCCTracesWeka"\n'
assert s.count(old) == 1
revision = '23f152f6f0f9399a85901b89a6458def0ef16729'
s = s.replace(old, old + f'    hf_revision = "{revision}"\n')
ast.parse(s)
p.write_text(s)
print('Pinned baseline Weka dataset:', revision, 'loader_sha256:', hashlib.sha256(s.encode()).hexdigest(), flush=True)
