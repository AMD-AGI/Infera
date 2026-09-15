# Patches

**No patch to the benchmark code was required.** `bench/profile_decode.py`, `bench/topology.py` and
`bench/batch_state.py` ran unmodified; `evidence/code_snapshot/code_hashes.sha256` is identical
across all 41 attempts. (`evidence/code_snapshot/code.diff` is non-empty, but every hunk belongs to
`compare_server.py` and `comparisons/` — dirty files left over from a previous task, not on this
sweep's code path.)

## `run_decode_env_yihou.diff`

**What.** A copy of the shared harness `scripts/run_decode.sh` with exactly one addition: entries in
a `DOCKER_ENV` variable are injected into the `docker exec` as `-e NAME=VALUE`, validated against
`^[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_,:.=-]+$`, and recorded per run in `docker_env_yihou.txt`.

**Why.** The C=160 capture OOM pointed at allocator fragmentation, so the run had to be repeated
with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The container was created without that
variable and `run_decode.sh` had no way to pass one.

**How applied.** Not applied — it is a **separate file**, `scripts/run_decode_env_yihou.sh`. The
shared `scripts/run_decode.sh` was deliberately left untouched for two reasons: it is part of the
audited harness referenced by a published packup, and bash reads a script incrementally, so editing
it while a sweep is mid-flight can corrupt a running point. The diff is included here so the
difference from the audited original is auditable in one glance.

**Context.** Everything else — the allocation check, the node denylist, the image-digest and owner
re-assertion, the code/hash/git snapshot, `launch_status.json` — is unchanged. Only the two `_xs`
points used it; see `notes.md` for what it did and did not achieve.
