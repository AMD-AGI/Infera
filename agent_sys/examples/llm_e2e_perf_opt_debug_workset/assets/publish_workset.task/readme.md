# publish_workset (debug harness copy)

Identical in behaviour to the real package's leaf: it copies one workset
directory into the output handoff and writes the `code` type's README.

`entry.sh` here is a shim that execs the **real** package's `publish.py`, so the
logic under test is the one that runs in production. See the comment in
`entry.sh` for why this is not a symlink.

Requires `--var real_package=` and `--var workset_dir=`.
