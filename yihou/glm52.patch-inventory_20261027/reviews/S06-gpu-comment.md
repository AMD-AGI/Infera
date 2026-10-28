Hardware follow-up at head `43dec26b032c08f4e9eee0c772db9f44d3afc275`: the registered, unmocked onload/forward regression now passes on one allocated MI355X GPU with PyTorch `2.9.1+rocm7.2.0.git7e1940d4` / HIP `7.2.26015`.

```bash
PYTHONPATH=python python3 -m pytest -q -p no:cacheprovider test/registered/model_loading/test_offloader_shuffled_weights.py
```

Result: **1 passed, 3 subtests passed**. It exercises real pinned-host offload, H2D onload on an alternate stream, forward execution, true/false/absent tag states, two onload cycles and restoration of the original host parameter. This is not a full-model Aiter/MXFP4 accuracy or performance run, and no CUDA hardware run is claimed.

The PR remains draft pending maintainer-enabled upstream CI/review. The earlier CPU result remains 3 passed + 2 subtests, with two expected failures on unpatched main. Please enable `run-ci` for the registered AMD/CUDA regression when appropriate.
