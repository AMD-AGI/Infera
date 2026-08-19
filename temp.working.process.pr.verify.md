# Working process — verifying the three upstream sglang GLM-5.2 PRs

**Status: INCOMPLETE.** Hardware validation is unfinished; this file exists so the work
can be resumed on a different machine. Session 2026-08-19 on n06-33 (8x MI355X / gfx950).

Task spec: `pr.verify.md`. Prior session's record: `pr.done.md` (second-hand — two of its
claims turned out stale, see below). Session config: `.claude/CLAUDE.md`.
Scratch workspace on n06-33: `/data/yihou/workspace.temp/pr-verify-20260819/`.

## Where this stands

| # | Stage | State |
|---|-------|-------|
| 1 | Upstream status + scope selection | done |
| 2 | Rebase all three branches onto current `main` | done |
| 3 | Code review of each PR | not started |
| 4 | **gfx950 hardware validation** | **in progress — blocked, see below** |
| 5 | Deep review (code read + LSP + serena) | not started |
| 6 | Flip draft -> ready, update records | not started |

All three PRs are still **draft**, with **zero** comments and **zero** reviews since they
were opened on 2026-08-07.

## The three PRs

| PR | Fix | Branch (on `dorado269/sglang`) | Rebased onto `c863760ae1` |
|----|-----|--------------------------------|---------------------------|
| [#33968](https://github.com/sgl-project/sglang/pull/33968) | HiCache ROCm host-pool allocator | `fix-hicache-rocm-pin-memory-allocator` | `33f0ea6cd3` |
| [#33970](https://github.com/sgl-project/sglang/pull/33970) | mooncake KV transfer waits on prefill forward | `fix-mooncake-pd-chunked-prefill-kv-race` | `780fbb3018` |
| [#33973](https://github.com/sgl-project/sglang/pull/33973) | DSA decode DP-divergent D2H syncs | `fix-dsa-decode-dp-host-sync-deadlock` | `2b3c9ea7a3` |

Upstream checkout: `/home/yihou/dev/git.16-10/sglang` (`origin`=sgl-project, `fork`=dorado269).

## Completed, with first-hand evidence

- **Rebase.** All three were 609 commits behind. Rebased onto `c863760ae1`, all three
  **without textual conflict**.
- **Defects re-confirmed present on current `main`** by source read, not by assuming the
  clean rebase implied it: `ALLOC_MEMORY_FUNCS` still has no HIP entry; `mooncake/conn.py`
  still contains `wait_event` 0 times vs `mori/conn.py` 6; `dsa_backend.py:796` still has
  `seq_lens.max().item()`.
- **PR C semantic gap closed.** `dsa_backend.py:1055` constructs
  `DSAMetadata(seq_lens_sum=forward_batch.seq_lens_sum)` on a path DRAFT_EXTEND_V2 reaches.
  Checked whether the PR's removal of the `.cpu()` mirrors could strand that field:
  `metadata.seq_lens_sum` has **zero** readers repo-wide (checked `dsa_indexer_metadata.py`,
  `dsa_topk_backend.py`, `nsa_backend.py`; no `asdict` / `astuple` / `replace` either).
- **Image built and saved.** `infera-local:sglang-prverify-20260819`, 80 GB, saved to
  `/data/yihou/images.backup/`. Base pinned by `Dockerfile.sglang` is
  `lmsysorg/sglang:v0.5.17-rocm720-mi35x` — a *different repo/tag* from the local
  `lmsysorg/sglang-rocm:...-20260809`; do not substitute, the DSA context diffs apply at
  `--fuzz=0` only against the pinned one.
- **`validate_A/B/C` all PASS** on gfx950 against current `main`.
- **The image's sglang is a git checkout** (`/sgl-workspace/sglang`, HEAD `2948168546`)
  with the infera patches applied as working-tree modifications. This is useful: a stock
  arm is `git checkout --` of the three PR-B files inside the container, so the two arms of
  an A/B differ by exactly those files and nothing else — no rebuild needed.
- **The applied mooncake diff matches PR #33970 exactly** (verified by `git diff` in the
  container against the PR body).

## Established environment facts (n06-33)

- 8x MI355X gfx950, 288 GB/GPU. ROCm 7.2.0, torch 2.9.1, amdgpu **6.14.14**.
- Model: **`/data/models/GLM-5.2-MXFP4` is a complete local copy** (408 GB, 282 shards).
  Prefer it over `/apps/data/models/...`, which is NFS at 716 MB/s and 100% full.
- Rails `benic{1..8}p1` = `192.168.{1..8}.14/31`; HCA map is **not** in numeric order:
  `ionic_0->benic1p1, ionic_1->benic2p1, ionic_2->benic4p1, ionic_3->benic3p1,
  ionic_4->benic5p1, ionic_5->benic6p1, ionic_6->benic8p1, ionic_7->benic7p1`.
- **`MC_GID_INDEX` must be 1 on this fabric, not 3.** `ionic_0` port 1 exposes only gid[0]
  (link-local) and gid[1] (`::ffff:c0a8:010e` = 192.168.1.14, RoCE v2). With 3, mooncake
  dies with "GID is NULL ... No available RNIC".
  **Trap: `ib_write_bw -x 3` works anyway** — perftest and mooncake index GIDs differently,
  so a passing perftest run does **not** validate mooncake's GID setting.
- **`RDMAV_FORK_SAFE=1` is required**, else "RDMA context setup failed: fork compatibility:
  Invalid argument".
- Rails route only to `192.168.N.12` (= n06-25). **n01-33 has no rail route to n06-33 and
  vice versa** (ARP INCOMPLETE) — this is what blocks the 2-node #33970 run.
- **A measurement error to not repeat:** `ping -I <source-IP>` only sets the source address;
  traffic still egresses over the `fenic` management NIC, which made all 8 rails look
  healthy when none were. Use `ping -I <ifname>`.

## Errors hit and how they were resolved

| Symptom | Root cause | Resolution |
|---|---|---|
| `docker pull` log showed "Pull complete" but no image | `nohup ... &` inside the background wrapper truncated it | run `docker pull` as the foreground command of a background task |
| Build failed: `Could not resolve host: index.crates.io` (exit 101) | **Hypothesised IPv6-only resolution with no v6 route — then tested it and the hypothesis was wrong.** A dedicated BuildKit probe with `--network=host --no-cache` showed DNS and IPv4 egress both fine (`http=200`) inside a RUN step | transient. **Retried; it succeeded. Dockerfile deliberately not modified.** |
| `validate_A.py`: "ALLOC_MEMORY_FUNCS block not found" | it resolved the container's already-patched sglang; the validator needs a *stock* tree to patch itself | `git worktree add ... origin/main --detach`, mount that |
| `validate_B.py` FAILED sections [1] and [2] | pointed at the stock tree, which correctly lacks the fix | worktrees at the rebased branches |
| `ib_write_bw` to n01-33 hung | no rail route between the two nodes | 2-node path abandoned; switched to single-node TP4+TP4 |
| `ib_write_bw` produced no data rows even to the routed peer | needs `-R` (rdma_cm) | with `-R`: 330–360 Gb/s on all 8 rails. mooncake also uses rdma_cm, so not a blocker |
| mooncake MVP: `GID is NULL`, `No available RNIC`, fork-compat error | GID index 3 does not exist on this fabric | `MC_GID_INDEX=1` + `RDMAV_FORK_SAFE=1` |
| `AttributeError: ... no attribute 'get_session_id'` | this mooncake build exposes `get_rpc_port`, not `get_session_id` | session id built as `f"{ip}:{eng.get_rpc_port()}"` |

## #33968 — negative result, stays draft

`pr.done.md` predicted gfx950 would measure `same=False` (host VA != device pointer), i.e.
the positive control that gfx942 lacks. Measured here:

```
device: AMD Instinct MI355X gcn=gfx950:sramecc+:xnack-
torch:  2.9.1+rocm7.2.0  hip=7.2.26015
  [pin_memory] / [mmap+hipHostRegister] / [+Mapped] / [+Portable|Mapped]  -> same=True (all four)
```

Suspected buffer size was the uncontrolled variable (the original fault report used a
7.33 GB indexer buffer; the probe uses 8 MiB), so swept 8 MiB / 256 MiB / 1 GiB / 4 GiB /
7.33 GB x 4 strategies: **all `same=True`**. Size is ruled out.

Only known remaining difference is amdgpu **6.14.14**, which the patch record attributes to
the MI300X *negative* control; the original gfx950 fault report does not record its driver
version. **No mechanism is claimed.** The statement that survives: this machine cannot
reproduce the fault, so it is a negative control like gfx942, and #33968's write-back
evidence remains historical.

`validate_A.py` still PASSes here — it tests equivalence and scope against the proven local
fix, which does not require the fault to reproduce. It also independently confirms the PR's
core argument: stock dispatch for a `torch.device('cuda:0')` key resolves to
`alloc_with_host_register`, so the two pools that key with a device object do fall through
the defaultdict.

**Action: needs a machine that reproduces `same=False`. User is sourcing one.**

## r01 — single-node TP4+TP4 premise checks (all passed)

Hypothesis: the #33970 race is between the mooncake transfer *thread* and the CUDA stream,
not between two hosts. If so, a single-node 1P1D with two TP4 legs over loopback RDMA
reproduces it, and the blocked 2-node path is not required.

1. **No local/loopback shortcut in mooncake.** `mooncake/conn.py` on current main has no
   `is_local` / `same_host` / `loopback` / `local_transfer` branch and no assertion that
   prefill and decode are on different hosts.
2. **The race is thread-vs-stream.** `conn.py:252` starts
   `threading.Thread(target=self.transfer_worker)`; that worker calls
   `engine.batch_transfer_sync` (`:657`, `:1112`). A CPU thread reads GPU memory outside the
   CUDA stream — exactly what the fix gates with an event. Nothing there depends on the peer
   being remote.
3. **RDMA loopback works at the verb layer.** `ib_write_bw -R`: cross-HCA (ionic_0 ->
   ionic_4) **348.59 Gb/s**, same-HCA **335.92 Gb/s**.
4. **mooncake itself works loopback** — the decisive check, with the real engine rather than
   perftest and deliberately without sglang: two containers on this host,
   `transfer_sync_read` of 8 MiB, `rc=0`, every byte 0xAB, **PASS**. Log also showed
   "HIP transport installed for intra-node GPU P2P".
   Script: `scripts/mvp_mooncake_loopback.py`.
5. **Capacity.** 408 GB model, TP4 -> ~102 GB/GPU on 288 GB cards.

**What this configuration can and cannot establish — must go in the PR:**
- CAN: the correctness claim (needle retrieval degraded -> clean) and the `synchronize()`
  cost, which is the reviewer's obvious first question.
- CANNOT: behaviour under real cross-node RDMA latency. Loopback is *faster*, so the race
  window is *narrower* and reproduction is *harder*. A positive reproduction here implies
  the cross-node case is at least as bad; a failure to reproduce here would **not** clear
  the cross-node case.

## r02 — stock positive control: FAILED to launch, two causes found

Both are artifacts of running two legs on one host. **Neither is a defect in the patch.**

**1. ZMQ port collision -> prefill SIGKILLed.** `ZMQ_TCP_PORT_DELTA = 233`, and both
containers are `--network=host` so they share 127.0.0.1:

| leg | `--port` | `port_base` | reserved |
|---|---|---|---|
| prefill | 30000 | 30234 | 30234–30240 |
| decode | 30001 | **30235** | 30235–30241 |

prefill's detokenizer wants 30235; decode already holds it ->
`zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:30235')` ->
`sglang subprocess exited with code -9`.

Fix for the next attempt: separate the two legs' `--port` by at least
`NUM_DERIVED_PORTS + ZMQ_TCP_PORT_DELTA` (e.g. 30000 / 31000). Note decode also binds
`0.0.0.0:5557` and `0.0.0.0:8801` **even with `KVAWARE=0`** (infera's
`--kv-events-bind` defaults to `tcp://0.0.0.0:5557` in
`/opt/infera/infera/engine/sglang/args.py:180`), so those must be moved too.

**2. The effective chunk size was 32768, not the 131072 that was passed.** From the
resolved `server_args`: `chunked_prefill_size=32768`. With DP-attention on, sglang divides
the global budget by `dp_size` (=4). `leg.sh` warns about exactly this.

Consequence: the needle probe's chunk-index arithmetic was **wrong** and must be recomputed
against the *resolved* value, not the requested one. Silver lining — a smaller chunk means a
200k-token prompt splits into ~7 chunks instead of ~2, so there are more non-final chunks
and the race window is easier to hit. **The needle must land in a non-final chunk**: the
final chunk goes through the sampling path, which already has a real `copy_done.synchronize()`,
so a final-chunk needle is retrieved correctly even on a broken build and would read as a
false PASS.

**Discipline for the retry: the positive control comes first.** Stock (unpatched) sglang
must reproduce a degraded needle score. Without that, a clean score on the patched tree
proves nothing.

## Scratch artifacts on n06-33

Under `/data/yihou/workspace.temp/pr-verify-20260819/`:

- `working_process.md` — the in-workspace log (this file supersedes it for hand-off)
- `scripts/mvp_mooncake_loopback.py` — the loopback MVP that proved premise 4
- `scripts/probe_host_devptr_sizes.py` — the #33968 size sweep
- `rounds/r02-stock-positive-control/scripts/up_singlenode.sh` — single-node 1P1D bring-up;
  takes `ARM=stock|patched` and **guards on `grep -c wait_event` (stock=0, patched=9)**,
  refusing to run a mislabelled experiment. Needs the two port fixes above.
- `rounds/r02-stock-positive-control/scripts/needle.py` — needle probe; tokenizes with the
  real tokenizer so each needle's chunk index is known rather than assumed, and reports
  non-final-chunk and final-chunk scores separately. Needs the chunk-size fix above.
- `sglang-stock/`, `sglang-B/`, `sglang-C/` — git worktrees used by the validators
- Image tar in `/data/yihou/images.backup/`

## Open items for whoever resumes this

1. **#33970**: fix the two r02 causes (leg ports 30000/31000 + move 5557/8801; recompute
   needle chunk math against the resolved 32768), run the **stock** arm to get a degraded
   score, then the **patched** arm. Also measure the `synchronize()` cost on prefill
   throughput — unmeasured, and the first thing a reviewer will ask.
2. **#33973**: not started. Needs 1x gfx950 with PD + DP-attention + MTP; the group must not
   deadlock on the first routed request, and `py-spy` should show no rank inside
   `dsa_backend`.
3. **#33968**: blocked on a machine that reproduces `same=False`.
4. **Stale record to correct:** `pr.done.md` says upstream **#30350** is the better fix for
   `patch_hicache_rocm_staged_write_back` and that the action is a re-review nudge.
   **#30350 was CLOSED unmerged on 2026-08-17 by its author** (Emmanuel0612). Per the user's
   decision this is a TODO only — they will look at it themselves; do not open a replacement
   PR. `pr.done.md` has not yet been updated.
5. CI red on all three is the `Block draft PR` repo policy, not a real failure.
6. **DCO**: every commit needs `-s`, signed off as the actual author — never a bot or
   assistant identity, and upstream rejects assistant `Co-Authored-By` trailers.

## Note on n06-33 machine state at hand-off

All containers from this session (`glm52_p`, `glm52_d`, `glm52-etcd`) were removed and their
GPUs released. A container `sla-decode-limou` belonging to **another user** (started
2026-08-19 07:08, running `gpt-oss-120b`, using this session's image
`infera-local:sglang-prverify-20260819`) holds ~286 GB on GPUs 1 and 2. It was left
untouched.
