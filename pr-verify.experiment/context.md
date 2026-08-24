# Context — what the next machine needs to know before starting

Everything here was established first-hand on n06-33 on 2026-08-19 unless marked
otherwise. `plan.md` is the remaining work; `working_process.md` is the narrative of
what happened. This file is the reference material: the task, the artifacts, the
environment, and the traps.

## The task

Verify, review, hardware-validate, and flip from draft to ready the three upstream
sglang PRs that upstream infera's local GLM-5.2 fixes. Spec: `pr.verify.md` at the
repo root.

| PR | Fix | Upstream file(s) | Local patch |
|----|-----|------------------|-------------|
| [#33968](https://github.com/sgl-project/sglang/pull/33968) | HiCache ROCm host-pool allocator | `mem_cache/pool_host/common.py` | `sglang_rocm/patch_hicache_rocm_host_alloc.py` |
| [#33970](https://github.com/sgl-project/sglang/pull/33970) | mooncake KV transfer waits on prefill forward | `disaggregation/{common/utils,mooncake/conn,prefill}.py` | `sglang_disagg/patch_mooncake_early_send_wait_event.py` |
| [#33973](https://github.com/sgl-project/sglang/pull/33973) | DSA decode DP-divergent D2H syncs | `layers/attention/dsa_backend.py` | `sglang_dsa/dsa_backend_dp_sync_and_page_table_rows.diff` (2a half only) |

All three are still **draft** with **zero** comments and **zero** reviews since
2026-08-07. The red CI on all three is the `Block draft PR` repo policy, not a real
failure.

Branches live on `dorado269/sglang`. Upstream checkout used here was
`/home/yihou/dev/git.16-10/sglang` (`origin`=sgl-project, `fork`=dorado269).
Rebased onto `c863760ae1` this session, all three without textual conflict:

| Branch | SHA after rebase |
|---|---|
| `fix-hicache-rocm-pin-memory-allocator` | `33f0ea6cd3` |
| `fix-mooncake-pd-chunked-prefill-kv-race` | `780fbb3018` |
| `fix-dsa-decode-dp-host-sync-deadlock` | `2b3c9ea7a3` |

All three defects were **re-confirmed present on current `main` by source read** —
not inferred from the clean rebase: `ALLOC_MEMORY_FUNCS` still has no HIP entry;
`mooncake/conn.py` still contains `wait_event` 0 times vs `mori/conn.py` 6;
`dsa_backend.py:796` still has `seq_lens.max().item()`.

## Repo documents

- `pr.verify.md` — the task spec.
- `pr.done.md` — what the prior session opened and on what evidence. **Second-hand.**
  Two of its claims turned out stale: the #30350 status (see `plan.md` step 6) and the
  prediction that gfx950 would reproduce #33968's fault (see below).
- `work.todo.md` — inventory of all seven sglang patches and why four get no PR.
- `work/upstream-glm52-sglang-prs/` — the prior session's adapted diffs (`*_upstream.diff`)
  and PR bodies (`*_pr_body.md`).
- `deploy/docker/patches/*/**.upstream.status.yaml` — per-patch records, validated by
  `scripts/validate-patch-status.py`.
- `examples/sglang_1p1d_glm5.2/` — how infera launches GLM-5.2 1P1D. `engine/leg.sh` is
  the real per-leg launcher and is fully env-driven; `common.sh:start_container` takes
  `CTR`; `engine/up.sh` is 2-node-shaped and cannot be used as-is for a single-node run.

## Artifacts in this folder

- `scripts/validate_{A,B,C}.py` — equivalence-and-scope checks of each upstream diff
  against the proven local patch. **All three PASS on gfx950 vs current main**
  (`logs/validate_*_gfx950.log`). They do not require the defect to reproduce, so a
  PASS is not evidence the fix works — see `plan.md`.
- `scripts/probe_host_devptr.py`, `scripts/probe_host_devptr_sizes.py` — the #33968
  host-VA vs device-pointer probe and its size sweep.
- `scripts/mvp_mooncake_loopback.py` — two mooncake `TransferEngine`s on one host.
  This is the MVP that proved the single-node #33970 plan is viable. Note this build
  exposes `get_rpc_port`, **not** `get_session_id`; the session id is
  `f"{ip}:{eng.get_rpc_port()}"`.
- `rounds/r02-stock-positive-control/scripts/up_singlenode.sh` — single-node 1P1D
  bring-up, `ARM=stock|patched`, with an arm guard. Needs the two port fixes in `plan.md`.
- `rounds/r02-stock-positive-control/scripts/needle.py` — needle probe. Needs the
  chunk-size fix in `plan.md`.
- `logs/` — build, pull, probe, RDMA MVP, and validator logs from this session.

## Environment on n06-33 (the box this ran on)

- 8x MI355X gfx950, 288 GB/GPU. ROCm 7.2.0, torch 2.9.1, amdgpu **6.14.14**.
- `/data` had 45T free; docker data-root is `/data/docker-data`.
- **Model: `/data/models/GLM-5.2-MXFP4` is a complete local copy** (408 GB, 282 shards).
  Prefer it. `/apps/data/models/GLM-5.2-MXFP4` is NFS, measured 716 MB/s, and the
  filesystem is 100% full (6.5 G avail).
- Image built this session: `infera-local:sglang-prverify-20260819`, 80 GB, saved to
  `/data/yihou/images.backup/`. Built via
  `IMAGE=... ID=... bash .github/scripts/build_test_push.sh build sglang`.
- Base image is **pinned** by `Dockerfile.sglang` to `lmsysorg/sglang:v0.5.17-rocm720-mi35x`.
  This is a *different repo/tag* from the local `lmsysorg/sglang-rocm:...-20260809` — do
  not substitute; the DSA context diffs apply at `--fuzz=0` only against the pinned one.

### The image's sglang is a git checkout — exploit this

`/sgl-workspace/sglang` inside the image is a git working tree (HEAD `2948168546`) with
the infera patches applied as uncommitted modifications. So a **stock arm is
`git checkout --` of the relevant files inside the container** — the two arms of an A/B
then differ by exactly those files and nothing else, with no rebuild. `up_singlenode.sh`
already does this for the three PR-B files and verifies it with
`grep -c wait_event` (stock=0, patched=9).

The applied mooncake diff was verified to match PR #33970 exactly, by `git diff` in the
container against the PR body.

## Traps — each of these cost time here

**1. `MC_GID_INDEX` must be 1 on this fabric, not 3.** `ionic_0` port 1 exposes only
gid[0] (link-local) and gid[1] (`::ffff:c0a8:010e` = 192.168.1.14, RoCE v2). With 3,
mooncake fails: "GID is NULL ... No available RNIC".
**`ib_write_bw -x 3` works anyway** — perftest and mooncake index GIDs differently, so a
passing perftest run does **not** validate mooncake's GID setting. Enumerate the GID
table on any new fabric rather than copying this value.

**2. `RDMAV_FORK_SAFE=1` is required**, else "RDMA context setup failed: fork
compatibility: Invalid argument".

**3. `ping -I <source-IP>` does not test a rail.** It only sets the source address;
traffic still egresses over the management NIC. This made all 8 rails look healthy when
none of them were. Use `ping -I <ifname>`, and confirm with `ip route get`.

**4. HCA-to-rail mapping is not in numeric order** on this box:
`ionic_0->benic1p1, ionic_1->benic2p1, ionic_2->benic4p1, ionic_3->benic3p1,
ionic_4->benic5p1, ionic_5->benic6p1, ionic_6->benic8p1, ionic_7->benic7p1`.

**5. `ib_write_bw` needs `-R`** (rdma_cm) here or it produces no data rows. With it:
330–360 Gb/s on all 8 rails, and 348.59 / 335.92 Gb/s cross-HCA / same-HCA in loopback.
mooncake also uses rdma_cm, so this is a perftest quirk, not a fabric problem.

**6. Two same-host legs collide on ZMQ ports.** `ZMQ_TCP_PORT_DELTA = 233` and both
containers are `--network=host`, so they share `127.0.0.1`. Ports 30000 and 30001 give
overlapping derived ranges (30234–30240 and 30235–30241) and prefill dies with
`Address already in use (addr='tcp://127.0.0.1:30235')` -> `exited with code -9`.
Decode also binds `0.0.0.0:5557` and `0.0.0.0:8801` **even with `KVAWARE=0`**.

**7. DP-attention divides `chunked-prefill-size` by `dp_size`.** Requesting 131072 at
`dp_size=4` resolves to 32768. Always read the effective value off the leg's own
`server_args` line. `leg.sh` warns about this; it still caught this session out and
invalidated the needle probe's chunk arithmetic.

**8. The gfx950 DSA recipe is mandatory or the model returns garbage with HTTP 200:**
`SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0 SGLANG_OPT_USE_TILELANG_INDEXER=1
SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_USE_JIT_NORM=0`. `leg.sh` sets these.

**9. A transient build failure that is not a Dockerfile bug.** One build died on
`Could not resolve host: index.crates.io` in the Rust step. The IPv6 hypothesis was
tested and **disproved**: a dedicated BuildKit probe with `--network=host --no-cache`
showed DNS and IPv4 egress both fine (`http=200`) inside a RUN step. Retry rather than
"fix" the Dockerfile.

## The #33968 negative result

`pr.done.md` predicted gfx950 would measure `same=False` (host VA != device pointer) —
the positive control that gfx942 lacks. Measured on n06-33:

```
device: AMD Instinct MI355X gcn=gfx950:sramecc+:xnack-
torch:  2.9.1+rocm7.2.0  hip=7.2.26015
  [pin_memory] / [mmap+hipHostRegister] / [+Mapped] / [+Portable|Mapped]  -> same=True (all four)
```

Buffer size was the obvious uncontrolled variable (the original fault report used a
7.33 GB indexer buffer; the probe uses 8 MiB), so it was swept: 8 MiB / 256 MiB / 1 GiB /
4 GiB / 7.33 GB x 4 strategies -> **all `same=True`** (`logs/probe_gfx950_sizes.log`).
Size is ruled out.

**No mechanism is claimed.** The statement that survives is that this machine cannot
reproduce the fault, so it is a negative control like gfx942, and #33968's write-back
evidence remains historical. The one known uncontrolled variable left is amdgpu 6.14.14 —
a lead to record, not a conclusion.

`validate_A.py` still PASSes here, and independently confirms the PR's core argument:
stock dispatch for a `torch.device('cuda:0')` key resolves to `alloc_with_host_register`,
so the two pools that key with a device object do fall through the defaultdict.

## Machine state at hand-off

All containers from this session (`glm52_p`, `glm52_d`, `glm52-etcd`) were removed and
their GPUs released. A container `sla-decode-limou` belonging to **another user** (started
2026-08-19 07:08, running `gpt-oss-120b`, using this session's image) held ~286 GB on
GPUs 1 and 2. It was left untouched.

n01-33 was claimed as a peer for the 2-node #33970 run but **has no rail route to n06-33**
in either direction (ARP INCOMPLETE), which is what forced the single-node approach.
n06-25 is rail-routed and GPU-idle but had three other users logged in, so it was not taken.
