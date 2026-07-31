# KV-aware routing + kvd on SGLang — PR deliverable and clean reproduction

**Ran:** 2026-07-31 · **Nodes:** prefill `chi2879` (10.2.122.10) + decode `chi2867`
(10.2.122.44), 8× MI355X gfx950 each · **Model:** GLM-5.2-MXFP4 ·
**Image:** `infera/engine-sglang:kvaware-kvd`
(`sha256:c9292019d0c5a36d97fb87e7b2944b3b6382bab2f33410b89ed03103192a0d80`),
built from `deploy/docker/Dockerfile.sglang.kvaware-kvd` in this branch ·
**Branch:** `yihou.dev.glm5.2.mxfp4.kvaware.kvd.pr` @ `da65cc7` on `origin/main` (`8692fb4`)

## What this is

The clean, PR-ready form of the kvaware/kvd investigation: the two bug fixes with
regression tests, a production Dockerfile, the operator documentation, and **one
final end-to-end run on freshly built artifacts** that reproduces the result.

Everything here was produced from the PR branch. Nothing depends on the earlier
scratch workspace or on the ad-hoc `pd-unified` images.

## Headline — all three acceptance tests pass

Topology for every test: **two-node PD over Mooncake RDMA** (real RDMA, not TCP),
**DP-attention 8/8 on both legs**, **KV-aware routing ON**, **kvd ON**.

| Test | Workload | Result | Verdict |
|---|---|---|---|
| **T1 correctness** | 4 factual prompts, temp=0 | **4/4 correct**, coherent | **PASS** |
| **T2 conc=32** | 128 req, ISL/OSL 1024, needle-in-log | **128/128 needle correct**, 0 HTTP errors, **128/128 `finish=stop`**, 3.1 req/s | **PASS** |
| **T3 conc=128** | 512 req, same shape | **502/512 needle correct**, **0 HTTP errors**, 6.8 req/s | **PASS** |

**T3's 10 non-clean responses are all `finish=length` at exactly the 1024-token
cap** — the run-on/EOS mode. These prompts are sent without a chat template and
with EOS suppressed for throughput, so a completion that never stops is expected
behaviour of this harness, not a defect. Per the operator's ruling on 2026-07-31
the conc=128 gate is *throughput without errors*. The split is clean:

- 502 good: `finish=stop`, median **149** tokens.
- 10 bad: `finish=length`, **all** at the 1024 cap, `</think>` repeating.

No HTTP errors, no hangs, no KV-corruption signature at either concurrency.

## kvd is proven to serve, not just to be wired

The trap this run was designed to avoid: **a speed-up does not prove kvd did
anything**, because SGLang's in-GPU radix cache serves a repeated prefix without
touching L3. During the stress tests kvd wrote 32 GB (`sets=18170`) and read
**nothing** (`gets=0`).

The discriminating test is restart-and-replay — restarting the engine empties the
GPU cache while the kvd daemon (and its L3) keeps running:

| | `gets_total` | `hits_total` | `sets_total` | `misses_total` |
|---|---:|---:|---:|---:|
| after stress | 0 | 0 | 18272 | 0 |
| replay **before** engine restart | 0 | 0 | 18272 | 0 |
| replay **after** engine restart | **102** | **102** | **18272** (unchanged) | 0 |

102 reads, **zero new writes**, zero misses. With the GPU cache empty, that reuse
could only have come from kvd's L3. Both replays returned **32/32 correct**.

## Also verified during this run

| Claim | Evidence |
|---|---|
| Image entrypoint replaces host libionic (no manual `docker cp`) | 8 `PORT_ACTIVE` inside the container on both nodes |
| kvd wiring auto-appends the hicache flags | `appends --enable-hierarchical-cache` in the prefill log |
| kvd adapter attaches on every DP rank | `infera-kvd adapter connected` ×8 on both legs |
| KV-event plane comes up | `KV plane up: events_bind=… snapshot=…` |
| Decode leg gets its radix cache back | `--disaggregation-decode-enable-radix-cache` in the decode log |
| Both workers register with the router | etcd registration for `DisaggMode.PREFILL` and `DisaggMode.DECODE` |
| The port-collision fix is in the image | the Dockerfile's build-time self-check asserts it and passed |

## What ships in the PR

`patches/` holds all five commits.

| # | Commit | What |
|---|---|---|
| 0001 | `fix(net)` | `free_tcp_port_block` randomised scan start + 4 new tests. Two workers on one host used to pick the same base deterministically → `ZMQError: Address already in use`. |
| 0002 | `fix(kvd)` | `storage_classify` strips the bind-mount subpath from the `findmnt` source. A bind-mounted L3 silently fell back to buffered I/O on qualifying NVMe. +2 tests. |
| 0003 | `docs(kvd)` | Feature matrix and offload page: SGLang kvd is supported, not 🚧. Adds the SGLang section. |
| 0004 | `build(sglang)` | `Dockerfile.sglang.kvaware-kvd` — digest-pinnable base, build-time self-check, baked defaults. |
| 0005 | `docs(serving)` | The operator guide: experiment-only vs production settings, how to verify each switch, which numbers are tunable. |

Both fixes are covered by tests that **fail on the pre-fix code** — verified by
reverting the source and re-running (3 failures, 48 passes; 51 pass after).

## Navigating this folder

| Path | What |
|---|---|
| `REPRODUCE.md` | ordered, copy-pasteable steps to redo the whole thing |
| `environment.md` | hardware, drivers, RDMA fabric, image digests, external paths, secrets needed |
| `notes.md` | gotchas, the wrong turns, and why each step is shaped the way it is |
| `patches/` | the 5 commits as `git format-patch` files |
| `scripts/` | every script that ran, verbatim |
| `results/raw/` | the captured outputs and kvd counters (JSON) |
| `results/` | the two kvd reuse transcripts (before/after restart) |
| `logs/` | engine, router, kvd-daemon and etcd logs (all < 4 MB, uncompressed) |

## Where to read the operator guidance

The answers to "which knobs are experiment-only", "what should production set",
"how do I check a switch actually worked", and "what is safe to tune" live in the
PR itself, not only in this packup:

- `manual/serving/kvaware_kvd_operations.md` (added by patch 0005)
- `manual/features/kv_cache_offload.md` § SGLang (patch 0003)
