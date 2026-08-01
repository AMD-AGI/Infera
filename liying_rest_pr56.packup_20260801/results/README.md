# Results index

The raw evidence lives in `../rounds/`, one directory per round. This maps each
claim in `../README.md` to the file that backs it.

| claim | file | what to look for |
|---|---|---|
| P1 fix is in the bytecode/source and the crate is green | `../rounds/r2_image_patch/patch_and_test.log` | `as_u32_any … 2`, `decodes_sglang_bigram_batch_under_mtp ... ok`, `10 passed` |
| the patched release binary differs from the shipped one | `../rounds/r2_image_patch/cargo_release_build.log` | two different md5s, `build_rc=0` |
| P2 + P3 tests **execute** (not marker-skipped) | `../rounds/r2_image_patch/pytest_rerun_no_asyncio_marker.log` | 22 items listed individually as PASSED under `-v` |
| **the live A/B proved nothing** | `../rounds/r3_rust_ab/cache_hits_ansi_stripped.log` | the *unpatched* leg reads `cache_hits=51` |
| …and its run log | `../rounds/r3_rust_ab/ab_summary.log` | both legs came up; the naive grep found nothing (see `../notes.md` §6) |
| **the control that discriminates** | `../rounds/r4_rust_control/control_then_treatment.log` | `FAILED … left: 0, right: 2` then `ok` |
| no regressions across the whole crate | `../rounds/r4_rust_control/full_regression.log` | cargo `56 + 12 + 2` passed, 0 failed |
| the image builds from the branch on both nodes | `../rounds/r5_built_image/build_chi2879.log`, `build_chi2867.log` | `=== done: sha256:… ===` |
| group E is in the **built** image | `../rounds/r5_built_image/verify_chi2879.log`, `verify_chi2867.log` | `=== GROUP E VERIFIED IN THE BUILT IMAGE ===` |

## Other files here

| file | what |
|---|---|
| `kv_event_zmq.rs.final` | the integration suite as shipped, including `subscriber_decodes_bigram_tokens_under_mtp` — the test the control run flips from FAILED to ok |

## Two things the logs show that are *not* failures

- **`pytest_rc=4` in `patch_and_test.log`.** A usage error, not a test failure:
  the image's `tests/` tree predates the bigram suites, so one path does not
  exist there. `rt_run.sh` stages them and is the real pytest step.
- **4 failures in `full_regression.log`** (`test_kv_event_e2e.py`). The image has
  no `pytest-asyncio`. They pass on a workstation that does — 21/21. Pre-existing
  and unrelated to group E. See `../notes.md` §3.
