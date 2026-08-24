# Task: verify and finalize three upstream sglang GLM-5.2 PRs

Review, validate on real gfx950 hardware, and flip from draft to ready the three
upstream PRs opened on 2026-08-07 for infera's local sglang GLM-5.2 fixes.

Spec: `pr.verify.md` (repo root). Prior session's record: `pr.done.md`.
Workspace for all temporary artifacts: `/data/yihou/workspace.temp/pr-verify-20260819/`.

## The three PRs

| PR | Fix | Upstream file(s) | Local patch |
|----|-----|------------------|-------------|
| [#33968](https://github.com/sgl-project/sglang/pull/33968) | HiCache ROCm host-pool allocator | `mem_cache/pool_host/common.py` | `sglang_rocm/patch_hicache_rocm_host_alloc.py` |
| [#33970](https://github.com/sgl-project/sglang/pull/33970) | mooncake KV transfer waits on prefill forward | `disaggregation/{common/utils,mooncake/conn,prefill}.py` | `sglang_disagg/patch_mooncake_early_send_wait_event.py` |
| [#33973](https://github.com/sgl-project/sglang/pull/33973) | DSA decode DP-divergent D2H syncs | `layers/attention/dsa_backend.py` | `sglang_dsa/dsa_backend_dp_sync_and_page_table_rows.diff` (2a half only) |

Branches live on `dorado269/sglang`; upstream checkout is `/home/yihou/dev/git.16-10/sglang`
(remotes: `origin`=sgl-project, `fork`=dorado269).

## Why this session exists

The prior session could not do the hardware validation: it had only gfx942
(the *negative control* for #33968) and the vultr MI355X cluster was unreachable.
All three PRs were therefore left as drafts, with `pr.done.md` recording exactly
which evidence was historical.

**This session has the hardware.** n06-33 (this box) and n01-33 are both
8x MI355X / gfx950, idle, on the same rack with 8 rails each.

## Environment — established first-hand this session

- **n06-33** (local): 8x MI355X gfx950, 0 KFD procs. 8 ionic HCAs all `PORT_ACTIVE`.
  Rails `benic{1..8}p1` = `192.168.{1..8}.14/31`. fenic `10.235.192.60/22`.
- **n01-33** (peer, newly claimed): same, rails `192.168.{1..8}.70/31`,
  fenic `10.235.192.136/22`. SSH works with BatchMode, no password. 0.078 ms RTT.
  **MVP-test RDMA before trusting it for the 2-node #33970 run — do not blind-experiment.**
- Disk: `/data` 45T free; docker data-root is `/data/docker-data`.
- Base image pinned by `Dockerfile.sglang`: `lmsysorg/sglang:v0.5.17-rocm720-mi35x`
  (pulled, 79.9 GB). Note it is a *different repo/tag* from the local
  `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260809` — do not substitute; the
  DSA context diffs apply at `--fuzz=0` against the pinned one.

## Upstream state — checked this session, differs from `pr.done.md`

- All three PRs: still **draft**, **zero** comments, **zero** reviews since 2026-08-07.
- All three were **609 commits behind** `main`; rebased onto `c863760ae1` this
  session, all three **without textual conflict**.
- All three defects **re-confirmed present** on current `main`:
  `ALLOC_MEMORY_FUNCS` still has no HIP entry; `mooncake/conn.py` still has
  `wait_event` 0 times vs `mori/conn.py` 6; `dsa_backend.py:796` still has
  `seq_lens.max().item()`.
- **`#30350` was CLOSED unmerged on 2026-08-17 by its author** (Emmanuel0612).
  `pr.done.md` says it is the better upstream fix for
  `patch_hicache_rocm_staged_write_back` and that the action is a re-review nudge.
  **That record is now stale.** User's decision: record a TODO only, they will
  look at it themselves. Do not open a replacement PR this session.
- CI red on all three is the `Block draft PR` repo policy, not a real failure.

## Core principles

1. **Suspend, don't conclude.** State what was measured. A clean rebase is not
   evidence the fix is still correct; verify semantics separately.
2. **Rank evidence.** Code read / experiment run = high confidence. Anything from
   `pr.done.md` is second-hand until re-verified this session.
3. **Scope.** Three PRs, as listed. Do not widen to the four patches that
   correctly get no PR. `#30350` fallout is a TODO, not work.
4. **DCO.** Every commit needs `-s`, signed off as the actual author. Never a bot
   or assistant identity, never a copied colleague line. Upstream (sgl-project)
   also rejects assistant `Co-Authored-By` trailers — do not add them.
5. Work in English; report to the user in Chinese.

## Deferred validation this session must close

| PR | Needs | Run |
|----|-------|-----|
| #33968 | 1x gfx950 | `probe_host_devptr.py` expecting `same=False` (the *positive* control; gfx942 measured `same=True`). Plus a HiCache write-back repro: stock must fault at the host VA, patched must not. |
| #33970 | 2 nodes with RDMA | GLM-5.2 1P1D over mooncake, `--chunked-prefill-size 131072`, overlap on. Needle retrieval 5/9 -> 9/9. **Plus** the added `synchronize()`'s cost on prefill throughput — unmeasured, and the first thing a reviewer will ask. |
| #33973 | 1x gfx950, PD + DP-attention + MTP | Group must not deadlock on first routed request; `py-spy` shows no rank inside `dsa_backend`. |

## Key references

- `pr.done.md` — what was opened and on what evidence (treat as second-hand).
- `work.todo.md` — inventory of all seven sglang patches, and why four get no PR.
- `work/upstream-glm52-sglang-prs/` — prior session's adapted diffs, PR bodies,
  and `validate_{A,B,C}.py` / `probe_host_devptr.py`.
- `examples/sglang_1p1d_glm5.2/` — how infera launches GLM-5.2 1P1D.
- Model: `/apps/data/models/GLM-5.2-MXFP4`.
- Test bench source: `/home/yihou/dev/git.16-19/Optimus-AgenticBench/.claude/worktrees/feat+fixlen-cache-bench`.
- Per-patch records: `deploy/docker/patches/*/**.upstream.status.yaml`,
  validated by `scripts/validate-patch-status.py`.

## Other notable details

- Do not consult packups on this box other than those named in `pr.verify.md`.
- `gh` is at `~/.local/bin/gh`, authenticated as `dorado269`.
- The build wrapper is `.github/scripts/build_test_push.sh build sglang`; it tags
  `${IMAGE}:sglang-${ID}`. Save built images to `/data/yihou/images.backup`.
- One build hit a transient `Could not resolve host: index.crates.io` in the Rust
  step. Re-probed: DNS and IPv4 egress work inside a BuildKit `--network=host`
  RUN step (`http=200`). Treat as transient; retry rather than "fix" the Dockerfile.
