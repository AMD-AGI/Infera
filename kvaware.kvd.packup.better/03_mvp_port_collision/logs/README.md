# Raw logs — GONE. Read this before looking for them.

**There are no raw engine log files in this directory, and none can be
produced.**

The Qwen3-1.7B MVP rounds (r1–r5) all ran inside a single container named
`kvexp` on chi2879 on 2026-07-30. That container was **removed** at teardown,
before the GLM-5.2 two-node runs began. Its `/tmp/r2/prefill.log`,
`/tmp/r2/decode.log`, `/tmp/r2/kvd.log` and `/tmp/r2/router.log` went with it.
They were never copied to the shared FS (`/mnt/vast`), which is the mistake —
the later GLM-5.2 runs wrote their logs to `/mnt/vast/c_huggingface/glm52_kvexp`
precisely so this could not happen again.

Fabricating stand-in log files here would be worse than having none, so no
engine logs are present.

## Why this matters less here than elsewhere

**The root cause of this experiment does not depend on the lost logs.** The bug
is in a pure-Python function with no GPU, no cluster and no container in the
path, and `scripts/mvp_port_block.py` reproduces it from first principles in
about one second on any machine. `results/mvp_port_block.txt` is a *fresh* run
of that MVP, not a transcript excerpt — it was regenerated on 2026-07-31 and
prints the same `32764` ×10 that the original round did.

So the chain of evidence for bug #1 is intact:

| Claim | Backed by |
|---|---|
| pre-fix `free_tcp_port_block` is deterministic | `results/mvp_port_block.txt` — live, re-runnable |
| base is 32764 on these boxes | same, derived from `ip_local_port_range` = 32768 |
| holding the reservation would block our own child (errno 98) | same, section 3(a) — live |
| `SO_REUSEADDR` on 0.0.0.0 is not exclusive | same, section 3(b) — live |
| the fix spreads the picks | same, section 2, against the patched `net.py` |
| 4 regression tests pass; the third fails pre-fix | `results/mvp_port_block.txt` tail |

What the lost logs held is only the *live consequence* — the actual leg crash.

## What survived of the live round

Captured in-session, quoted verbatim in `results/r2_port_collision.txt`:

| Line | Significance |
|---|---|
| `zmq.error.ZMQError: Address already in use (addr='tcp://*:32765')` | the crash; note wildcard, not loopback |
| `RuntimeError: Rank 0 scheduler died during initialization (exit code: -3)` | decode leg dead |
| `--kv-events-config {"publisher": "zmq", "endpoint": "tcp://*:32764", ...}` | both legs got the same base |
| `[DP1 TP1 EP1] Creating dynamic storage backend 'infera-kvd'` | kvd connected — it was not the problem |
| `[DP1 TP1 EP1] infera-kvd adapter connected to /tmp/kvd/kvd.sock (model=qwen3, compat_key=tp0of1_pp0of1)` | same |
| `[DP1 TP1 EP1] Tree cache initialized: impl=HiRadixCache hierarchical=True` | same |
| `[DP*] Allocating 8.00 GB host memory for hierarchical KV cache.` | the previous round's fix holding |

## What is lost and cannot be recovered

- The full decode-leg traceback (only the two exception lines were kept).
- Which leg won port 32764 — the start ordering was not logged.
- The exact `--kv-events-bind` values used in r2.
- `kvd.log` from r2, so the daemon's view of the aborted decode startup is
  unknown.
- Any per-DP-rank detail beyond the DP1 lines quoted above.

## Regenerating logs

Desk check — no cluster, ~1 second, and it is the actual root cause:

```bash
bash scripts/run.sh                 # MODE=desk is the default
```

Live 1P1D on one node, ~8 min:

```bash
MODE=live NETPY=old bash scripts/run.sh   # reproduce r2's dead decode leg
MODE=live NETPY=new bash scripts/run.sh   # both legs up, distinct endpoints
```

`NETPY=old` leaves the container's stock (pre-fix) `net.py` alone; `NETPY=new`
`docker cp`s `scripts/net_fixed.py` over it. Output extracts land in
`results/r2_port_collision.observed.txt`. To keep the whole log files, pull them
before the script's `trap cleanup EXIT` removes the container:

```bash
docker cp <ctr>:/tmp/r2_old/. ./logs/
```
