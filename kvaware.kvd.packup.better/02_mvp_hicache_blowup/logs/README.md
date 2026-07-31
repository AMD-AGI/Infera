# Raw logs — GONE. Read this before looking for them.

**There are no raw log files in this directory, and none can be produced.**

The Qwen3-1.7B MVP rounds (r1–r5) all ran inside a single container named
`kvexp` on chi2879 on 2026-07-30. That container was **removed** at teardown,
before the GLM-5.2 two-node runs began. Its `/tmp/r1/prefill.log`,
`/tmp/r1/decode.log`, `/tmp/r1/kvd.log` and `/tmp/r1/router.log` went with it.
They were never copied to the shared FS (`/mnt/vast`), which is the mistake —
the later GLM-5.2 runs wrote their logs to `/mnt/vast/c_huggingface/glm52_kvexp`
precisely so this could not happen again.

Fabricating stand-in log files here would be worse than having none, so this
directory is empty on purpose.

## What survived

Everything quoted in `results/` was captured **in-session**, live, while r1 was
running — i.e. it is a transcript excerpt, not a reconstruction. The specific
lines that survived are:

| Line | Where it is now |
|---|---|
| `zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:30235')` | `results/r1_hicache_blowup.txt` |
| `RuntimeError: sglang subprocess exited with code -9 before reporting ready` | `results/r1_hicache_blowup.txt` |
| `[DP0 TP0 EP0] max_total_num_tokens=1547424, ...` | `results/r1_hicache_blowup.txt` |
| `[DP0 TP0 EP0] Allocating 354.94 GB host memory for hierarchical KV cache.` | `results/r1_hicache_blowup.txt` |
| `[DP*] Allocating 8.00 GB host memory for hierarchical KV cache.` (post-fix, r2) | `results/r1_hicache_blowup.txt` |

The 354.94 GB figure is independently re-derivable from the model's
`config.json` plus the logged `max_total_num_tokens` — that derivation is in
`results/hicache_sizing_arithmetic.txt` and it reproduces the printed value
exactly. So the headline number does not rest on the lost log alone.

## What is lost and cannot be recovered

- The full startup tracebacks (only the two exception lines above were kept).
- The two legs' actual `--port` values in r1. Only the *derived* collision port
  (30235) was recorded.
- Per-GPU VRAM and the node's free-RAM figure at the moment of the failure.
- `kvd.log` from r1, so the daemon's view of the aborted startup is unknown.
- Whether the prefill leg would also have hit the hicache blow-up — it died on
  the port collision first.

## Regenerating logs

`scripts/run.sh` re-runs this round end to end and writes fresh logs. It takes
about 6 minutes on one node (Qwen3-1.7B cold-starts in ~2 min).

```bash
MODE=broken bash scripts/run.sh    # reproduce the failure
MODE=fixed  bash scripts/run.sh    # confirm --hicache-size bounds the pool
```

The script pulls the relevant lines out of the container and writes them to
`results/r1_hicache_blowup.observed.txt` for comparison with the committed
`results/r1_hicache_blowup.txt`. If you want the whole log files rather than the
extracts, pull them before the script's `trap cleanup EXIT` removes the
container:

```bash
docker cp <ctr>:/tmp/r1_broken/. ./logs/
```
