# Raw engine logs — the SECOND (post-restart) engine

| File | Node | Role | Process lifetime (from the file) |
|---|---|---|---|
| `pd_prefill_r2.log` | chi2879 (10.2.122.10) | prefill TP8, gmu 0.88, KVAWARE=1 KVD=1 | 11:53:46 → 12:20:09 |
| `pd_decode_r2.log` | chi2867 (10.2.122.44) | decode TP8, gmu 0.85, KVAWARE=1 KVD=1 | 11:54 → 11:59 |

Copied verbatim off the shared FS (`/mnt/vast/c_huggingface/glm52_kvexp/`).
Run 2026-07-30.

**These are the RELAUNCHED engine's logs — the "new process" of the
experiment.** Each file contains exactly one startup:

```bash
grep -ac 'server_args=ServerArgs' pd_prefill_r2.log   # 1
grep -ac 'ready to roll'          pd_prefill_r2.log   # 1  (11:56:21)
grep -ac 'Load weight begin'      pd_prefill_r2.log   # 8  (one per DP rank, one startup)
```

The *first* engine's logs are not here. That is deliberate, not an omission:
what matters about the first engine is that it **died**, and death is evidenced
by the VRAM reading and the counter deltas, not by a log tail.

## ⚠ Read this before grepping: the prefill file spans two experiments

`pd_prefill_r2.log` covers **more than this experiment**, because the prefill
leg was never restarted afterwards. Its request traffic falls into two clearly
separated windows:

```bash
grep -a 'POST /v1/chat/completions' pd_prefill_r2.log \
  | grep -oE '2026-07-30 [0-9]{2}:[0-9]{2}' | uniq -c
#   11 2026-07-30 11:58   \
#   21 2026-07-30 11:59   /  32 requests  <- THIS experiment's replay
#   19 2026-07-30 12:17   \
#   13 2026-07-30 12:18    \  64 requests <- a LATER routing experiment
#   14 2026-07-30 12:19    /               (2 arms x 32), NOT this one
#   18 2026-07-30 12:20   /
```

**This experiment is the 11:56–11:59 window: 32 requests.** That matches the
workload exactly (4 sessions × 4 turns × 2 phases) and matches the 31/32 score.

The 12:17–12:20 block is a later run in which the TP8 decode leg was replaced by
two TP4 decode workers; this same prefill process served it. Note the decode
file here **ends at 11:59** — that leg *was* torn down, which is the tell.

Counting requests across the whole file gives 96 and is wrong for this
experiment. Always window the greps:

```bash
grep -a 'POST /v1/chat/completions' pd_prefill_r2.log | grep -cE '2026-07-30 11:5[6-9]'   # 32
grep -a 'Decode batch' pd_decode_r2.log | grep -cE '2026-07-30 11:5[89]'                   # 22
```

## Grep recipes — what these logs can show

**The new engine is genuinely new, and re-attached to the surviving daemon:**

```bash
grep -ac 'infera-kvd adapter connected' pd_prefill_r2.log   # 8  (one per DP rank)
grep -ac 'infera-kvd adapter connected' pd_decode_r2.log    # 8
grep -a  'infera-kvd adapter connected' pd_prefill_r2.log | grep -oE '2026-07-30 [0-9:]+'
#   11:55:59 ... 11:56:09 — all eight, at THIS startup
grep -ac 'KV plane up:' pd_prefill_r2.log   # 1
grep -ac 'ready to roll' pd_prefill_r2.log  # 1
```

Eight fresh adapter connections is the log-side evidence that a different
process attached to a daemon that was already running — the daemon did not
restart, so it printed no startup banner; the engine did.

**It really is a distinct process from the one that populated the store.** The
randomised kv-events port base differs from the earlier engine's:

```bash
grep -aoE '"endpoint": "tcp://\*:[0-9]+"' pd_prefill_r2.log | sort -u   # tcp://*:27591
# the engine that WROTE the 340 entries had tcp://*:25075
```

Two different draws from `free_tcp_port_block` — two different processes.

**The configuration is unchanged from the populating run** (it has to be, or the
block hashes would not match and every lookup would miss):

```bash
grep -aoE 'enable_hierarchical_cache=[A-Za-z]+' pd_prefill_r2.log | sort -u  # True
grep -aoE 'hicache_size=[0-9]+'                 pd_prefill_r2.log | sort -u  # 16
grep -ah 'Tree cache initialized' pd_prefill_r2.log | head -1                # HiRadixCache
grep -a 'Allocating .* host memory for hierarchical' pd_prefill_r2.log       # 8x 16.00 GB
grep -aoE "disaggregation_mode='[a-z]+'|enable_dp_attention=True|dp_size=8|ep_size=8|tp_size=8|mem_fraction_static=[0-9.]+" \
     pd_prefill_r2.log | sort -u
grep -ac 'MC_FORCE_TCP'        pd_prefill_r2.log   # 0
grep -ac 'HIP dmabuf disabled' pd_prefill_r2.log   # 8
```

Worth knowing *why* the hashes match: the leg script exports `PYTHONHASHSEED=0`,
which makes block hashing stable across processes and therefore makes
cross-restart lookup possible at all. It is set in the script's environment, so
it does not appear in the log — check `scripts/glm52_leg.sh`. Without it a new
engine would compute different keys for identical content and every get would
miss.

**The prefix really was being served from cache** — the replay shows a large
`#cached-token` against a tiny `#new-token`:

```bash
grep -a 'Prefill batch' pd_prefill_r2.log | grep -E '11:5[89]' | head -2
# ... #new-seq: 1, #new-token: 64, #cached-token: 5440 ...
```

That is the *GPU* radix cache reporting, not kvd — do not mistake it for the
result. It is included because it confirms the workload's prefix was shared as
intended.

## What these logs do NOT contain — and it is the important part

**The result of this experiment is not in these files.**

The conclusion rests on three things, none of which an engine log records:

| Evidence | Source | Quoted in |
|---|---|---|
| kvd counters `gets 170→340`, `hits 170→340`, `sets` flat at 340 | the daemon, over its unix socket (`scripts/kvdstats.sh`) | `results/step3_restart_reload.txt`, `results/confounders_removed.txt` |
| GPU VRAM back to idle (297840640 B / 297754624 B) before relaunch | `rocm-smi --showmeminfo vram` on the hosts | same |
| kvd daemon alive across the kill (`pgrep -fc 'infera.kvd'` = 1) | the host/container | same |

Grepping these logs for `gets_total` or `VRAM` finds nothing. That is expected —
an engine log cannot report that a *different* process still holds a store, nor
that the GPU was empty before it started.

Also absent:

- **kvd daemon logs** (`/tmp/kvd.log`) — container-local, containers removed at
  teardown. The daemon's own view of serving 170 reads to a brand-new client is
  therefore unavailable. This is the most regrettable gap in this experiment: it
  would have been the most direct possible confirmation.
- **The first engine's logs**, so the store's write side is not directly
  visible here (only its result, in the BEFORE counters).
- **The router log.**
- **The text of the 32 completions.** `prefix_reuse.py` prints response text only
  for failures. The single failure *was* printed, which is how it was identified
  as a truncation ("largest planet", session 2 — the model spent its 128-token
  budget on a reasoning preamble and never reached "Jupiter") rather than a wrong
  answer. The 31 passes left no text behind.
- **Any explanation for the decode node's `+36 sets / +22 entries`.** Not
  investigated; recorded as unresolved.

## Regenerating

```bash
bash ../scripts/run.sh
```

~20 minutes end to end: cold start, populate, kill the legs (not the daemon),
verify VRAM and daemon liveness, second cold start, replay, diff the counters.
The script **gates on the VRAM precondition** — if VRAM is still high after the
kill it says so loudly and tells you not to cite the result, because a surviving
GPU cache would explain any reuse you then observe.

Outputs: `results/round1_populate.observed.txt`,
`results/preconditions.observed.txt`, `results/round2_reload.observed.txt`, and
a computed delta table.

With `KEEP=1`, afterwards:

```bash
# on the prefill node
cp /mnt/vast/c_huggingface/glm52_rr09/pd_*_r2.log ./
# and the daemon log this packup lacks
docker exec glm52_rr09 cat /tmp/kvd.log > kvd_prefill.log
```
