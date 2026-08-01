# Round 4 — kvd read-back on spur, PROVEN

Goal (Track 2): `gets_total > 0` **and** `hits_total > 0` while `sets_total` stays
**flat**, on an engine whose in-GPU radix cache is empty. Met.

## Why this experiment and not a latency measurement

CLAUDE.md: *"A latency win is not evidence that kvd did anything — sglang's in-GPU
radix cache serves a repeated prefix without touching L3."* Re-sending a prompt to a
warm server proves nothing: the pages never leave the GPU. The only clean attribution
is restart-and-replay.

The three conditions have to hold simultaneously, and each one is load-bearing:

| condition | how it was established |
|---|---|
| GPU radix cache empty | prefill engine killed; all 8 GPUs polled to **VRAM 0 %** before reboot |
| L3 store still populated | kvd daemon **never killed** — `kill_engine.sh` targets only `infera.engine.sglang` / `sglang::` PIDs, then prints the survivors |
| prompts byte-identical | `replay_probe.py` rebuilds the filler from `random.Random(20260731)` with the same word list and call order as `correctness.py` |

Killing the container instead of the engine would have failed all three at once — and
would have silently dropped the ROCm host-alloc and mooncake patches, which live in
the container's site-packages.

## Counters

    before                        after
    entries       12,942          entries       12,942
    host_bytes    22,870,688,256  host_bytes    22,870,688,256
    gets_total             0      gets_total    12,942     <- +12,942
    hits_total             0      hits_total    12,942     <- +12,942
    sets_total        12,942      sets_total    12,942     <- FLAT
    misses_total           0      misses_total       0
    evictions_total        0      evictions_total    0

`hits/gets = 12,942/12,942 = 100 %`, `sets` unchanged. Every page the replay needed
was found in L3 and none was re-written. A miss would have shown as `misses_total`
climbing and `sets_total` climbing behind it as the page was re-stored.

## Server-side corroboration

    depth=5%   prompt=120,047  cached=120,000 (100.0%)   8.1s
    depth=25%  prompt=120,045  cached=120,000 (100.0%)   6.8s
    depth=50%  prompt=120,046  cached=120,000 (100.0%)   5.8s
    depth=75%  prompt=120,046  cached=120,000 (100.0%)   5.7s
    depth=95%  prompt=120,047  cached=120,000 (100.0%)   3.4s

`cached_tokens` is reported by the engine (`--enable-cache-report`) and is an
independent path from the kvd daemon's own counters. 100 % cached on a cold GPU cache
is only possible if the pages came from L3. The two sources agree.

## Gate on the rebooted leg

    ready to roll                   1
    Memory access fault             0     <- the round-2/3 bug, still fixed
    infera-kvd adapter connected    8
    Attached hybrid DSA pool stack  8
    Scheduler hit an exception      0
    Errno 98 after ready            0

## Read path was structurally open

Checked before trusting the result, because CLAUDE.md records a configuration where
L3 is written and never read:

    cache_controller.py:467   self.prefetch_capacity_limit = int(0.5 * self.mem_pool_host.size)

With `--hicache-size 32` (absolute GB) this is non-zero. The failure mode in the note
comes from `--hicache-ratio` below ~1.5, which we never use.

## What this does and does not establish

**Establishes:** kvd on spur stores *and* serves. Combined with round 3 (write path,
0 faults after the ROCm host-alloc fix), the full L3 round-trip now works on
MI355X/gfx950 at 120K-token prompts — a regime no sanctioned kit had exercised
(`better/08` was sequential and ≤ ~6,200 tokens).

**Does not establish:** any performance claim. The replay was sequential, single
client, one shared prefix, no eviction pressure (`evictions_total = 0`). Latency
numbers above are not a benchmark and are not quoted as one.

## Artifacts

    kill_engine.sh              engine-only teardown, kvd daemon preserved
    replay_probe.py             byte-identical prompt replay, reports cached_tokens
    kvd_before_replay.json      counters before
    kvd_after_replay.json       counters after
    replay.log                  server-side cached_tokens
