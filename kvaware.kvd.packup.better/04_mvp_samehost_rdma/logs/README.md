# Raw logs — GONE. Read this before looking for them.

**There are no raw log files in this directory, and none can be produced.**

The Qwen3-1.7B MVP rounds (r1–r5) all ran inside a single container named
`kvexp` on chi2879 on 2026-07-30. That container was **removed** at teardown,
before the GLM-5.2 two-node runs began. Its `/tmp/r3/prefill.log`,
`/tmp/r3/decode.log`, `/tmp/r3/kvd.log` and `/tmp/r3/router.log` went with it.
They were never copied to the shared FS (`/mnt/vast`), which is the mistake —
the later GLM-5.2 runs wrote their logs to `/mnt/vast/c_huggingface/glm52_kvexp`
precisely so this could not happen again.

Fabricating stand-in log files here would be worse than having none, so this
directory is empty on purpose.

## What survived

Captured in-session while r3 was running, and quoted verbatim in
`results/r3_samehost_rdma.txt`:

| Line | Significance |
|---|---|
| `Failed to get kvcache from prefill instance` | the HTTP 500 body |
| `worker_pool.cpp:408 ... local_nic: ionic_0, peer_nic: ...@ionic_4: transport retry counter exceeded` | the mooncake transport giving up |
| `rdma_endpoint.cpp:472  Invalid argument: received packet mismatch` | the RDMA endpoint's own complaint |
| `prefill http://10.2.122.10:30000 kv_events_endpoint=tcp://10.2.122.10:17213` | router view — patch 0001 holding |
| `decode  http://10.2.122.10:31000 kv_events_endpoint=tcp://10.2.122.10:31215` | same, distinct port |

The router-view lines are the ones that make this a *transport* result rather
than a wiring result: they prove both legs came up and registered with
independent kv-event endpoints, so every layer above the KV transfer was
healthy. Without them r3 would be ambiguous.

## What is lost and cannot be recovered

- The **full** `worker_pool.cpp` / `rdma_endpoint.cpp` stanzas. Only the lines
  above were captured — the surrounding retry counts, QP numbers, and the
  peer's complete endpoint string are gone.
- How many completions were attempted before the round was abandoned. The
  HTTP 500 is recorded; the count is not.
- `kvd.log` and `router.log` from r3, so the daemon's and the router's own view
  of the failed transfers is unknown.
- Whether kvd's counters moved at all during the failed attempts (almost
  certainly not — no request completed — but it was not checked).
- The exact mooncake build string inside the image at that moment. The image
  digest is recorded in the environment section; the mooncake version inside it
  was not separately noted in this round.

## Regenerating logs

```bash
TRANSPORT=rdma bash scripts/run.sh   # reproduce the failure (EXPECTED to fail)
TRANSPORT=tcp  bash scripts/run.sh   # MC_FORCE_TCP=1, completions come back
```

About 8 minutes on one node (Qwen3-1.7B cold-starts in ~2 min; the rest is
waiting for both legs and the router).

**Check `active_ports=8` in the output before believing anything.** If host
`libionic` injection failed, RDMA has already silently degraded to TCP and
`TRANSPORT=rdma` would "pass" for entirely the wrong reason. `scripts/run.sh`
prints this and says so.

Extracts land in `results/r3_samehost_rdma.observed.txt`. To keep the whole log
files, pull them before the script's `trap cleanup EXIT` removes the container:

```bash
docker cp <ctr>:/tmp/r3_rdma/. ./logs/
```
