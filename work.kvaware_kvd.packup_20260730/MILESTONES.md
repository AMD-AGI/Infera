# Milestones — the whole arc, in order

One page for "what happened, in what order, and what each step actually
established". Read `README.md` first for the verdict table; this is the
narrative and the audit trail. Every row links to its evidence file.

All on 2026-07-30. Model GLM-5.2-MXFP4 unless noted.

| # | Milestone | Verdict | Evidence |
|---|---|---|---|
| 0 | Support survey: what infera implements for sglang | ✅ | `results/support_matrix.md` |
| 0b | Static arg-compatibility matrix (no GPU) | ✅ | `results/support_matrix.md` §coupling |
| 1 | MVP rounds on Qwen3-1.7B (single node) | ⚠️ found 2 bugs, garbled output | `notes.md` |
| 2 | **Bug #1** — `free_tcp_port_block` port collision | ✅ fixed + 4 tests | `patches/0001-note.md` |
| 3 | GLM-5.2 2-node PD+DPA **baseline** (switches OFF) | ✅ 4/4 | `results/baseline_probe_4of4.txt` |
| 4 | GLM-5.2 2-node PD+DPA **kvaware+kvd ON** | ✅ 4/4 | `results/step1_kvaware_kvd_4of4.txt` |
| 5 | Prefix-reuse workload — does kvd actually serve? | ✅ 32/32, kvd `gets 0→170` | `results/step2_prefix_reuse.txt` |
| 6 | Cross-restart reuse — is kvd *the* thing working? | ✅ +170 hits, 0 new sets | `results/step3_restart_reload.txt` |
| 7 | Routing effect with 2 decode workers | ✅ kv-aware 17/0 vs RR 6/8 | `results/step4_role_weights_routing.txt` |
| 8 | Real block-device L3 | ◐ partial | `results/step5_nvme_l3.txt` |
| 9 | **Bug #2** — `storage_classify` bind-mount subpath | ✅ fixed + 2 tests | `patches/0002-note.md` |

## How the questions narrowed

The value of the sequence is that each step removed a confounder the previous
one left behind. Quoting a middle step out of context overstates it.

**Step 4 — "it works"** ... but kvd's counters were `gets=0 sets=0`. Four short
prefix-disjoint prompts gave the offload path nothing to do. So: *wired and
harmless*, not *useful*.

**Step 5 — "kvd serves"** ... `gets 0→170, hits=170, misses=0`, 573 MB resident.
But every hit repeated something the same live process had just stored, and the
GPU radix cache was warm throughout. So: *serving*, but not provably *the thing
doing the work*.

**Step 6 — "kvd is the thing"** — killed the engine, kept the daemon. VRAM back
to idle, so the GPU cache cannot explain anything; the writer process is dead,
so same-process reuse cannot either. A brand-new engine got **+170 gets / +170
hits with `sets_total` perfectly flat** — it read 170 blocks it never wrote.
That is the confounder-free result.

**Step 7 — "the router really routes"** — steps 1-6 could only confirm the role
weights were *loaded*; with one worker per role the scorer has no alternative to
choose between. Adding a second decode worker made a decision possible: same
workload, same workers, flip the policy and the distribution inverts
(kv-aware **17/0**, round-robin **+6/+8**).

## What is still NOT established

Kept deliberately short and blunt; the long form is in `notes.md`.

1. **Latency benefit of kvd** — none observed. Step 6's reload was 0.76 s vs a
   0.71 s cold run. Re-reading ~6200 tokens of KV over a UDS is not obviously
   cheaper than recomputing prefill on 8× MI355X. kvd's win here is capacity and
   survival, not speed. **The 2.7× figure elsewhere in these results is GPU
   radix-cache reuse — kvd's counters are flat across that run. Do not
   attribute it to kvd.**
2. **The prefill overlap weight (20.0)** — unmeasured; only one prefill worker.
3. **Affinity among competing prefixes** — step 7 used a single shared prefix,
   so it shows affinity, not *correct* affinity. Needs N prefixes × N workers.
4. **O_DIRECT on genuine NVMe** — this node's 8 NVMe drives are unmounted and
   one holds another team's 120 GB kvd store. Untouched on purpose.
5. **Concurrency** — everything here was sequential. The load term in
   `cost = w*(blocks−hits) + active_blocks` never pushed back against locality.

## Two bugs, both real, both in the kvaware/kvd path

| | Bug #1 | Bug #2 |
|---|---|---|
| Where | `infera/common/net.py:free_tcp_port_block` | `infera/kvd/storage_classify.py:_findmnt` |
| Trigger | 2 workers/host with kv-events on, `dp_size>1` | any bind-mounted kvd L3 |
| Symptom | 2nd leg dies: `ZMQError: Address already in use` | L3 silently forced to buffered I/O |
| Root cause | fixed scan start + probe released before use → deterministic collision | bind subpath `/dev/md0[/sub]` passed to `lsblk` |
| Verified | MVP: 10/10 identical bases pre-fix | A/B on node: `[(none)]` → `[md0 (?, ssd)]` |
| Tests | 4 new | 2 new (47 total in that file) |

Both are **uncommitted**, in the working tree of
`yihou.dev.glm5.2.mxfp4.experiment`. `patches/apply_all.sh` installs both plus
their tests onto a clean checkout — verified end-to-end against a fresh clone of
base commit `362192e`, producing a byte-identical tree and 51 passing tests.

## Three defaults that collide, and one that lies

Found one per debugging round — same class of problem, three unrelated code
paths:

1. sglang `--kv-events-config` port block → **bug #1**
2. `--kv-events-bind` default `5557`
3. `--kv-snapshot-port` default `8801` — **the dangerous one**: the leg prints
   `ready to roll` and *then* dies during etcd registration. The engine looks
   healthy; the worker just never appears in `/v1/workers`.

And one false alarm worth not chasing again: the startup warning
`SGLang version has no recognized prefetch_threshold field` is **cosmetic**. On
0.5.15.post1 that field isn't in `ServerArgs` at all — it's read from the
backend extra-config (`hiradix_cache.py:675`), which is exactly where infera
puts it. The value takes effect.
