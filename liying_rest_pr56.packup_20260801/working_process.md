# Backfilling liying's remaining PR56 patches (non-gfx942)

**Goal.** The merged branch `yihou.dev.glm52.merged.experiment` deliberately took
only the two PR56 commits our experiment exercised. Close the remaining gap for
everything that is **not gfx942**, validating each by experiment rather than by
argument, patching the built image in place first.

**Ground truth** (per CLAUDE.md): the built image `infera/engine-sglang:merged`
is live on chi2879 + chi2867, and `glm52.kvd.kvaware.mtp.pd.dp.kv.event.all.commited.finial/`
has the verified scripts + the passing numbers to 对拍 against.

## What is actually left

PR56 `llying/glm5p2_fp8_fixes` @ 7 commits, refetched 2026-08-01 — unchanged,
still OPEN. Mapping each against our branch:

| PR56 commit | status on our branch | verdict |
|---|---|---|
| `0360af5` early-send image layer | taken as `2181136` | done |
| `01b0534` bigram decode | **Python half taken** as `6e6fdb7` — `git diff 6e6fdb7 01b0534 -- <py files>` is **empty** | **Rust half missing** |
| `6121189` gfx942 image layer | out | out of scope (gfx942) |
| `b2150c3` gfx942 v0.5.16 base | out | out of scope (gfx942) |
| `1ebdc7e` gfx942 arch detection | out | out of scope (gfx942) |
| `0bb23c7` `INFERA_SGLANG_READY_TIMEOUT` | out | **candidate** |
| `d63e48b` NodePort-range skip | out | **candidate**, conflicts with our `826619b` |

So three work items:

- **P1** — `rust/router/src/kv_event.rs` bigram decode + its inline test.
  The one *real unfixed bug* on the branch: a `--router-backend rust`
  deployment with MTP still silently degrades to load balancing.
- **P2** — `INFERA_SGLANG_READY_TIMEOUT` in `infera/engine/sglang/worker.py`.
- **P3** — NodePort-range skip in `infera/common/net.py`. Textually conflicts
  with our `826619b` (randomised scan start); both edit `free_tcp_port_block`.
  Must be merged by hand, keeping both properties.

## Rounds

| round | dir | purpose | outcome |
|---|---|---|---|
| 1 | `rounds/r1_local_patch/` | write P1/P2/P3, local unit tests | **PASS** — 21/21 local |
| 2 | `rounds/r2_image_patch/` | patch the live container, verify in bytecode + cargo | **PASS** — 22/22 pytest, 10/10 cargo incl. the new bigram test |
| 3 | `rounds/r3_rust_ab/` | live A/B: unpatched vs patched Rust router | **VACUOUS** — see below |
| 4 | `rounds/r4_rust_control/` | real-ZMQ bigram test, run unpatched then patched | **PASS** — 0 → 2 hits |

### Round 1 — writing the three patches

- **P1** applied verbatim from `01b0534` (`as_u32_any` + the inline
  `decodes_sglang_bigram_batch_under_mtp` test). No conflict: our branch never
  touched `kv_event.rs`.
- **P2** applied verbatim from `0bb23c7` + its env-reference row. Liying's commit
  ships **no test**; wrote `tests/engine/sglang/test_ready_timeout.py`.
- **P3** hand-merged. Our `826619b` randomises the scan start; liying's
  `d63e48b` skips the NodePort window. Both live in the same loop, so the merge
  keeps `itertools.chain(randomised, exhaustive)` and adds the `reserved` skip
  as the first statement of the loop body. Liying's separate
  `tests/unit/common/test_net_ports.py` was **folded into our existing
  `test_net_port_block.py`** rather than added as a second file — same subject,
  same function, and a second file would only re-conflict later.

Local: `21 passed` across the three touched suites.

### Round 2 — in the live container (chi2879 `merged_run`)

Verified in **bytecode**, not source, per `notes.md` §3.

| check | result |
|---|---|
| `net::_reserved_nodeport_range` in `net*.pyc` | OK |
| `worker::INFERA_SGLANG_READY_TIMEOUT` in `worker*.pyc` | OK |
| 40× `free_tcp_port_block(8)`: none in 30000-32767, >1 distinct base | OK |
| `INFERA_NODEPORT_RANGE=none` → guard off; `=not-a-range` → default kept | OK |
| pytest (net + ready-timeout + bigram) | **22 passed** |
| `cargo test -p infera-router kv_event` | **10 passed**, incl. `decodes_sglang_bigram_batch_under_mtp` |

Three tooling traps hit and fixed, all previously documented for this stack:

1. **`docker exec` with a heredoc silently no-ops.** Without `-i` there is no
   stdin, so `docker exec CTR bash -s <<EOF` runs an empty script and *exits 0*.
   The step read as a pass while doing nothing. Fix: stage the inner script as a
   **file** and `docker exec CTR bash /tmp/inner.sh`.
2. **`cargo` needs `LIBCLANG_PATH`** (onig_sys/bindgen) — the Dockerfile sets it
   by searching `/opt/rocm*`; an ad-hoc `cargo test` does not. Here it resolves
   to `/opt/rocm-7.2.0/lib/llvm/lib`.
3. **The image has no `pytest-asyncio`.** `@pytest.mark.asyncio` is then an
   *unknown mark*: the coroutine is never awaited and the test reports **pass**
   without executing. Four pre-existing `test_kv_event_e2e.py` async tests fail
   there for this reason alone (they pass locally, 21/21, where the plugin
   exists) — unrelated to these patches. My new test was rewritten to call
   `asyncio.run` from sync test functions so it actually executes in the image
   it guards.

### Round 3 — the live Rust-router A/B, and why it proved nothing

Ran the shipped `/usr/local/bin/infera-router` and the freshly built patched
binary in turn against the live PD pair on port 8199, same prompt sent twice,
reading policy.rs's per-pick `cache_hits`:

| leg | md5 | picks |
|---|---|---|
| before (unpatched) | `f0990a44…` | `cache_hits=0,0,`**`51`**`,0` / `request_blocks=51` |
| after (patched) | `4743821b…` | `cache_hits=0,0,0,0` |

**The unpatched binary read 51 hits, so the comparison is vacuous.** Cause: the
live *prefill* leg runs without `--speculative-algorithm` — MTP is on the decode
leg, and `is_eagle` only switches the radix key to bigrams on a leg that has it.
The wire was carrying plain ints, which both binaries decode correctly.

(The "after" leg reading 0 across the board is a second-order artifact of the
same setup: each `run_leg` starts a fresh router process, and the view lives in
the process — `notes.md` §6. The first leg happened to catch a warm repeat.)

Two ways forward: bring up an MTP **prefill** leg (~9 min cold start per leg, and
it exercises the same decoder), or drive the decoder over a real socket with the
exact bigram shape. Took the second — same code path, and it can be run *both
ways*, which the live A/B could not.

### Round 4 — the control that actually discriminates

Added `subscriber_decodes_bigram_tokens_under_mtp` to
`rust/router/tests/kv_event_zmq.rs` — the wire-level twin of the in-crate unit
test: a real ZMQ PUB emitting `BlockStored` with `token_ids` as overlapping
pairs, asserting the view hashes to **the same two blocks** the query side
computes over the flat slice (not merely that it is non-empty).

Then ran it against both versions of `kv_event.rs` in the same container:

| `kv_event.rs` | result |
|---|---|
| **unpatched** (`as_u32_any` removed, `as_u32_vec` back to ints-only) | **FAILED** — `left: 0, right: 2` |
| **patched** | **ok** — 2 passed |

`left: 0` is the bug verbatim: every pair dropped, view permanently empty, no
error anywhere. This is the evidence the live A/B could not produce.

### Round 5 — rebuild the image from the branch and verify there

The three commits are only real if the deliverable carries them, so the branch
was exported with `git archive` and built on **both** nodes from
`deploy/docker/Dockerfile.sglang`, unchanged — it already copies `infera/` and
`rust/` before building, so group E needs no Dockerfile edit.

Built under the tag **`infera/engine-sglang:merged-e`**, deliberately not
`:merged`: the running `merged_run` containers on both nodes were created from
that tag and are this line of work's ground-truth reference. Overwriting it
would have left the reference unreproducible.

| node | image id |
|---|---|
| chi2879 | `sha256:bfcb6462fa30…` |
| chi2867 | `sha256:27667ee43291…` |

(Different ids are expected — independent builds, so Rust objects and layer
timestamps differ. Content equivalence is what is checked, not digests.)

Verified in a throwaway container from the image on each node — never the live
`merged_run`, whose in-place patches would otherwise answer for the image's own
content:

| check | chi2879 | chi2867 |
|---|---|---|
| `net::_reserved_nodeport_range` in `net*.pyc` | OK | OK |
| `worker::INFERA_SGLANG_READY_TIMEOUT` in `worker*.pyc` | OK | OK |
| 40× `free_tcp_port_block(8)`: outside 30000-32767, >1 base | OK | OK |
| `_wait_ready(timeout=None)` (env-resolved signature) | OK | OK |
| `infera-router` present and runs | OK | OK |
| `rust/` removed by the build, as designed | OK | OK |

**One limitation, stated rather than papered over.** The Rust half cannot be
verified from the artifact the way bytecode can: `as_u32_any` is a small private
fn the release profile inlines, and doc comments are not in the binary. So the
script checks the *source the build consumed* (`/root/merged_e_src`, byte-for-byte
what the build read) for `fn as_u32_any` and both new tests — all present on both
nodes — and the behavioural evidence remains round 4's control, which is stronger
than any grep: revert the fix and the real-socket test fails `0 vs 2`.

## Result

| # | patch | evidence |
|---|---|---|
| P1 | `d3c0d6f` Rust bigram decode | **control**: unpatched FAILS `left: 0, right: 2` over a real ZMQ socket, patched passes. Whole crate 70/70. |
| P2 | `fd3540d` `INFERA_SGLANG_READY_TIMEOUT` | 5 new tests, executing (not marker-skipped) in the image |
| P3 | `eef9bfc` NodePort skip, hand-merged with `826619b` | 10 tests; 40 live allocations avoid the window and stay spread |

Branch `yihou.dev.glm52.merged.experiment`, now 31 commits. Both temporary
groups verified droppable by running the rebases, not by asserting them:
dropping E lands exactly on `c0450a4`; dropping C afterwards replays D onto
`7f2dac8` cleanly.

Unit suite locally: **1168 passed, 1 skipped** (was 1162; +6 NodePort tests).
