# Notes — gotchas, wrong turns, and what is *not* proven

## What is NOT proven (read this before quoting anything from here)

Being precise about the boundary of the evidence, because it is easy to
over-read this packup:

Step 1 (2026-07-30, `results/step1_kvaware_kvd_4of4.txt`) closed the big one:
kvaware+kvd on GLM-5.2 two-node PD+DPA scores **4/4 on real RDMA**, matching the
switches-off baseline. What remains open:

Step 2 (`results/step2_prefix_reuse.txt`) then closed the kvd question: with a
~6200-token shared prefix, kvd went from idle to **gets=170 / hits=170 /
misses=0**, 573 MB resident, correctness 32/32. What remains open:

1. **The role weights' routing effect is unmeasured.** `20.0/2.0` was confirmed
   live in the router log, but with **one** prefill worker and **one** decode
   worker the scorer has no alternative to choose between. Needs ≥2 workers per
   role to show anything.
2. **`hits=170 / misses=0` is same-process reuse only.** Every lookup repeated
   something this same deployment had just stored. kvd's actual selling point —
   surviving a restart, or sharing across engines — is untested. That needs a
   restart-and-reload test.
3. **L3 is still on the container overlay** — and this one is environmental, so
   here is the full diagnosis for whoever picks it up:
   - `--long-path /tmp/kvd-long` lives on the container's overlayfs.
   - `python3 -m infera.kvd classify` inside the container reports
     `mount = overlay (overlay)`, `devices = [(none)]`,
     `rationale: unknown device, conservative buffered`.
   - **It is not a missing-tool problem.** `lsblk` and `findmnt` both exist in
     the container and work (`findmnt -T /tmp/kvd-long` → `overlay`); overlayfs
     simply has no backing block device for `lsblk` to report.
   - This node's eight 7 TB NVMe drives have ext4 but are **not mounted**;
     mounting them on a shared cluster is not ours to do.
   - The **write** path is proven (573 MB resident, recovered by the daemon
     across a restart: `long region recovered from /tmp/kvd-long`). What is
     untested is O_DIRECT against a real block device.
   - Fix: rebuild the container with a host bind-mount (e.g.
     `-v /mnt/nvme-raid/kvd-long:/kvd-long`) and point `--long-path` at it.
     `--io-mode direct` can force the mode but does not change the substrate.
4. **The 2.7× speedup is not kvd's.** kvd's counters were flat across the warm
   run; the GPU radix cache served it. Don't let this number migrate into a
   kvd claim.
5. **The prefill overlap weight (20.0) is still unmeasured** — step 4 added a
   second *decode* worker, so only the decode weight got exercised. Measuring
   the prefill side needs two prefill nodes.
6. **Step 4 used a single shared prefix.** It shows affinity, not *correct*
   affinity among competing prefixes. A stronger test: N distinct prefixes over
   N workers, each prefix landing consistently on its own worker.

### Refuted: "the prefetch_threshold override silently no-ops"

infera logs, at every startup:

```
WARNING: SGLang version has no recognized prefetch_threshold field
  (tried hicache_storage_prefetch_threshold, hicache_prefetch_threshold,
   prefetch_threshold)
```

I read this as "the override is lost". **It is a false alarm.** On sglang
0.5.15.post1 `prefetch_threshold` is *not* a `ServerArgs` field at all — grep
finds zero hits in `server_args.py`. It is read only from the backend
extra-config:

```
mem_cache/hiradix_cache.py:675   prefetch_threshold = extra_config.pop("prefetch_threshold", 256)
mem_cache/hiradix_cache.py:1603  or prefetch_length < self.prefetch_threshold: return
```

— and infera *does* pass `{"prefetch_threshold": 64}` in exactly that
extra-config. So the value takes effect; only infera's `ServerArgs`-probing
fallback path (`kvd_wiring.py:_finish_wiring`) is looking in the wrong place and
emitting a scary warning. Cosmetic bug in infera, not a functional one.

### Refuted: "infera's probe plane never attaches"

An earlier round suspected infera's own KV-event probe plane never attached,
because `_find_radix_cache()` looks for the RadixCache in the **wrapper**
process while sglang runs in a **subprocess**, and no `KV plane up:` line had
been seen. Step 1 **refuted** this — the line is there:

```
INFO:__main__:KV plane up: events_bind=tcp://0.0.0.0:5557
  events_advertise=tcp://10.2.122.10:5557 snapshot=http://10.2.122.10:8801
  engine_block_size=64 index_block_size=64
```

Kept here as a record of the wrong hypothesis: absence of evidence in a truncated
log was read as evidence of absence. The earlier rounds simply never got far
enough to print it.

## Gotchas that cost real time

### `docker exec -d … bash -lc '…'` does not persist

Bit us **twice** in this session (router, then the kvd daemon), and it is already
recorded in the packup-07 notes. The detached login-shell form exits and takes
the child with it. Symptom: no process, no log file, no error.

**Fix:** write a script file, `docker cp` it in, then
`docker exec -d $CTR bash /run_thing.sh`. Or, for the legs,
`docker exec -d $CTR env VAR=… bash /script`. `scripts/run_router.sh` and
`scripts/run_kvd.sh` exist precisely for this.

Related: nested `ssh jump "ssh node '…'"` quoting silently mangles `$f` in loops
(the outer shell expands it). Stage a script file instead of fighting the quoting.

### `--hicache-ratio` default sizes the host pool off the KV pool

Round 1 of the MVP tried to allocate **354.94 GB per DP rank** (×4 = 1.4 TB on a
3 TB box). Not a bug — `hicache_ratio=2.0` scales off `max_total_num_tokens`,
which was 1,547,424 tokens because the model was small and VRAM was plentiful.

**Fix:** `--hicache-size <GB>` (absolute, overrides the ratio). But note the
opposite trap from `hicache_validate.py`: a *ratio* below 1.5 makes sglang's
`prefetch_capacity_limit` compute to ~0, so L3 gets written and never read. Use
an absolute size, or keep ratio ≥ 2.0.

### THREE kvaware ports collide when two workers share a host

Found one per debugging round. They are unrelated code paths — fixing one does
nothing for the others.

| # | Port | Default | Failure mode |
|---|------|---------|--------------|
| 1 | sglang `--kv-events-config`, one PUB per DP rank at `base+rank` | from `free_tcp_port_block` | **deterministic** same base for every caller → `ZMQError: Address already in use`. This is patch 0001. |
| 2 | `--kv-events-bind` — infera's *own* publisher | `tcp://0.0.0.0:5557` | identical on every leg; second leg fails to bind |
| 3 | `--kv-snapshot-port` — infera's per-worker snapshot HTTP server | `8801` | **the nastiest**: the leg logs `ready to roll`, *then* dies during etcd registration. The engine looks healthy and the worker simply never appears in `/v1/workers`. |

#3's symptom is worth internalising, because "ready to roll" is the line
everyone greps for:

```
INFO:__main__:using etcd registration: endpoint=10.2.122.10:2379 ...
OSError: [Errno 98] error while attempting to bind on address ('0.0.0.0', 8801)
sys.exit(STARTUP_FAILURE)
```

`glm52_leg.sh` exposes `KV_PUB_PORT` and `KV_SNAP_PORT` for exactly this; give
every co-located worker its own.

### `--mem-fraction-static` is TP-dependent

`0.85` is tuned for TP8, where GLM-5.2's 408 GB is 51 GB/GPU. At **TP4** the
weights double to 102 GB/GPU and 0.85 leaves nothing — DP3 died on a 390 MiB
allocation with 120 MiB free. `0.70` works at TP4. Scale it with TP, don't carry
the number over.

### Same-host PD over mooncake RDMA does not work

Two legs on one box means RDMA loops back across rails (`ionic_0` → `ionic_4`):

```
worker_pool.cpp:408 ... transport retry counter exceeded
rdma_endpoint.cpp:472 Invalid argument: received packet mismatch
```

`MC_FORCE_TCP=1` gets a working (slow) path, at the cost of the garbled output
described above. **For any correctness claim, use two nodes and real RDMA.**

### Never probe a PD leg directly

`curl` to the prefill leg's own port just hangs — a PD leg only serves through
the pair. The discriminating experiment is a differential run (flip the feature,
same everything else), not a direct probe.

### The GLM-5.2 scripts in `glm5.2.mxfp4.packup_20260727/` bypass infera entirely

They call `python3 -m sglang.launch_server`. All kvaware/kvd wiring lives in the
**infera wrapper** (`python3 -m infera.engine.sglang`). `scripts/glm52_leg.sh`
is that packup's verified leg recipe with the entry point swapped and etcd
discovery added — everything else (DSA-ROCm envs, mooncake envs, DPA args, gmu,
chunk) is carried over byte-for-byte.

## Method note: use a small model to debug wiring

The single highest-leverage decision here. Qwen3-1.7B cut the loop from a ~30 min
GLM-5.2 cold start to ~2 min, which is what made 5 rounds affordable in an
afternoon — and rounds 1-3 were *all* wiring bugs (hicache sizing, port
collision, port collision again) that a 1.7B model surfaces exactly as well as a
400 GB one. Save the big model for the correctness run, where the model actually
matters.

## Wrong turns worth not repeating

- **Tried to fix the port collision by holding the reservation.** Would have
  locked out our own child (`127.0.0.1` probe vs `0.0.0.0` child bind). Caught by
  running the 6-line MVP *before* shipping the change. See `patches/0001-note.md`.
- **Then tried `0.0.0.0` + `SO_REUSEADDR`.** Not exclusive — a second probe takes
  the same port anyway. Also caught by MVP.
- **Tried `--kv-events-bind` as a workaround for the sglang port collision.**
  Wrong knob: it controls infera's own publisher, not the port passed to sglang
  (`worker.py:77` ignores it). Reverted after re-reading the code.
- **Directly probed a PD leg** to isolate the garbling. Returns nothing; wasted a
  round before switching to the differential baseline that actually answered it.
