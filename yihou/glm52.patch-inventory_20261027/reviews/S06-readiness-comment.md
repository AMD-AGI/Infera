Updated to `43dec26b032c08f4e9eee0c772db9f44d3afc275`, rebased on main `449617bc94a5d4ae23ad62d0da2f842e711ca983`.

- Added a CPU regression through the real offloader forward wrapper: the consumer sees `is_shuffled`, and the original host parameter is restored across repeated forwards.
- CPU result: 3 passed, 2 subtests. Restoring unpatched main makes both positive tests fail on the missing tag; the absent/false controls still pass.
- Added an AMD/CUDA-registered regression for real pinned-memory offload, asynchronous onload, forward values/metadata, and parameter restoration. That test has no device/stream mocks, but it has only been collected/skipped on the CPU host so far; the isolated GPU requests did not obtain a node during this validation session.
- All local hooks passed and the [new-head repository lint](https://github.com/sgl-project/sglang/actions/runs/40471207945) passed.

The [AMD gate](https://github.com/sgl-project/sglang/actions/runs/40471208261/job/122078486436) still reports the missing `run-ci` label, so skipped downstream checks are not hardware passes. Could a maintainer enable CI and run `PYTHONPATH=python python test/registered/model_loading/test_offloader_shuffled_weights.py` on AMD? The body includes the exact commands and validation limits; no performance claim is made.
