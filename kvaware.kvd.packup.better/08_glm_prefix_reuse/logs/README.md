# Raw engine logs — NONE for this experiment. Read this before looking.

**There are no `.log` files in this directory, and none can be produced from the
original run.**

This experiment did not restart the engine. It ran on the **same live
deployment** as the preceding kvaware+kvd run — same containers, same two legs,
same kvd daemons — and changed only the workload (and, for the second arm, the
router). The engine legs therefore kept appending to the *previous* run's log
files. Those files are the kvaware+kvd startup logs; they contain this
experiment's request traffic mixed into their tail, but nothing that identifies
which lines belong to which arm.

Copying them here would imply they are this experiment's evidence. They are not,
so they are not included. Fabricating stand-in logs would be worse.

## Where this experiment's evidence actually lives

**It is not in engine logs at all — it is in the kvd daemon's counters**, read
over its unix socket with `scripts/kvdstats.sh`. That is a deliberate property
of the design, not a gap: the engine log has no idea how many blocks the daemon
stored or served, and an engine log could not have produced this result.

| Claim | Where | Backed by |
|---|---|---|
| kvd went from idle to serving: `gets 0→170`, `hits=170`, `misses=0` | `results/step2_prefix_reuse.txt` | daemon counters before/after, quoted verbatim |
| 573 MB resident (`600837120` bytes), 340 entries | same | same |
| 32/32 correct, both arms | same | `prefix_reuse.py` stdout, captured in-session |
| median latency 0.71 s → 0.26 s | same | same |
| the speedup is the GPU radix cache, **not** kvd | `results/speedup_is_not_kvd.txt` | kvd counters *identical* across the fast run |
| role weights were loaded (`prefill=20.0 decode=2.0`) | `results/step2_prefix_reuse.txt` | one line from the router log, captured in-session |
| the 1/32 workload-design bug and its fix | `results/workload_design_bug.txt` | in-session transcript + the fix visible in `scripts/prefix_reuse.py` |

## Grep recipes — for the logs you generate yourself

There is nothing to grep here, so these are the commands to run against a
**fresh** run's logs (`scripts/run.sh` with `KEEP=1`). They re-derive the
preconditions this experiment assumes:

```bash
# kvd wired on every DP rank of both legs
grep -ac 'infera-kvd adapter connected' pd_prefill_px.log   # expect 8
grep -ac 'infera-kvd adapter connected' pd_decode_px.log    # expect 8

# hierarchical cache really on (this is what kvd plugs into)
grep -aoE 'enable_hierarchical_cache=[A-Za-z]+' pd_prefill_px.log | sort -u  # True
grep -ah 'Tree cache initialized' pd_prefill_px.log | head -1                # HiRadixCache

# host pool bounded by --hicache-size, not the 2.0 ratio
grep -a 'Allocating .* host memory for hierarchical' pd_prefill_px.log       # 8x 16.00 GB

# real RDMA, not the TCP fallback
grep -ac 'MC_FORCE_TCP'        pd_prefill_px.log   # 0
grep -ac 'HIP dmabuf disabled' pd_prefill_px.log   # 8

# the long prefix really was chunked as expected (sanity, not a claim)
grep -ac 'Prefill batch' pd_prefill_px.log
```

And the counters, which are the actual result — from inside either container:

```bash
docker exec <ctr> bash /kvdstats.sh
# StatsResponse(entries=..., host_bytes=..., gets_total=..., sets_total=...,
#               hits_total=..., misses_total=..., ...)
```

Run it **before** the workload and **again after**. A single reading proves
nothing; the delta is the evidence. `scripts/run.sh` does this automatically
around every phase.

## What is lost and cannot be recovered

- Which specific requests produced which cache events. The counters are
  aggregate; no per-request tracing was enabled.
- The kvd daemon's own log (`/tmp/kvd.log`) — container-local, containers
  removed at teardown. So the daemon's view of the 340 sets (eviction decisions,
  L2→L3 spill timing) is unavailable.
- The router log beyond the single `router-policy=kv-aware overlap_weight=1
  prefill=20.0 decode=2.0` line captured in-session.
- The full text of the 32 completions. Only the pass/fail per request and the
  median latency were recorded; `prefix_reuse.py` prints response text **only
  for failures** (`[XX] s{n} want=... got=...`). For the run that scored 31/32
  that is enough to identify the miss as a truncation; for a passing run no text
  survives.

## Regenerating

```bash
bash ../scripts/run.sh                # both arms, ~6 min cold start + ~8 min
KEEP=1 bash ../scripts/run.sh         # leave it up, then copy the logs off
ARMS=default bash ../scripts/run.sh   # just the arm where kvd actually moves
```

Writes `results/step2_prefix_reuse_default.observed.txt` and
`..._weighted.observed.txt`, each with the kvd counters bracketing the workload.
With `KEEP=1`, afterwards:

```bash
# on the prefill node
cp /mnt/vast/c_huggingface/glm52_px08/pd_*_px.log ./
# and the daemon log this packup lacks
docker exec glm52_px08 cat /tmp/kvd.log > kvd_prefill.log
```
