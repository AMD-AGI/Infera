# Reproduction kit — kvd ROCm host-alloc fault

Two independent proofs, cheapest first:

* **A. The bug** — ~2 min on **one** node, no PD, no model load. Shows the pointer
  mismatch, then makes the real kernel fault and un-fault by flipping one allocator.
* **B. The fix end-to-end + read-back** — ~40 min on **two** nodes. Full PD deployment,
  zero faults, then restart-and-replay to prove L3 is read, not just written.

Do A first. It is the whole root cause for 2 minutes of GPU time, and if A does not
reproduce, B will only waste an hour telling you the same thing less clearly.

---

## 0. Prerequisites

* **Machines.** A: one node, 1 GPU is enough. B: two nodes, 8 GPUs each.

      sbatch -p amd-spur -q amd-burst-qos -N1 -G8 -t 12:00:00      # x2 for B

  Expect `JobHoldMaxRequeue` bounces; retry or `--exclude` the bad node. Health-gate
  every freshly held node — spur has nodes that enumerate 8 GPUs and report
  `torch.cuda.is_available() == False`:

      python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
      # want: True 8

  **`spur exec <job> <cmd>` only. SSH to compute nodes is banned.**

* **Secrets** (values not included — source them yourself): docker registry login.
  `export DOCKER_CONFIG=/tmp/dockercfg` before *every* docker call.

* **External deps:** model at `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`
  (B only; A needs no weights). Scratch at `/shared_nfs/<you>/`.

* **Repo state:** image built from `infera.kv.fix` @
  `52d71195498f9caaf8b84bcca3276a366b1e8010` (clean tree).

* **Image:** `infera/engine-sglang:kvaware-kvd`, base digests pinned in
  `environment.md`. Build **on each node** — a backgrounded `docker save` of a 79 GB
  image inside `spur exec` dies at namespace teardown, so building twice beats moving
  a tar. Stage 2 must print `kvaware+kvd self-check OK`.

---

## A. Reproduce the bug and the fix in 2 minutes

Both scripts run inside the container on a single node. Copy them to shared storage
first so `docker exec` can see them.

### A1. Measure the pointer mapping — cannot fault, launches no kernel

    docker exec $CTR python3 /shared_nfs/<you>/probe_hostreg.py

Expected: `pin_memory` reports `same=True`; every `mmap + hipHostRegister` variant
reports `same=False`, including `hipHostRegisterMapped`, `Portable|Mapped`, and
`MAP_PRIVATE`. Also prints `gcnArch` (want `xnack-`) and `amdgpu.noretry`.

This is the root cause in one screen. Everything else is consequence.

### A2. Make the real kernel fault, then not fault

    # faults:
    docker exec $CTR python3 /shared_nfs/<you>/micro_writeback.py --host-alloc mmap_default
    # also faults (proving no flag combination saves it):
    docker exec $CTR python3 /shared_nfs/<you>/micro_writeback.py --host-alloc mmap_mapped
    docker exec $CTR python3 /shared_nfs/<you>/micro_writeback.py --host-alloc mmap_portable_mapped
    docker exec $CTR python3 /shared_nfs/<you>/micro_writeback.py --host-alloc mmap_private
    # does NOT fault:
    docker exec $CTR python3 /shared_nfs/<you>/micro_writeback.py --host-alloc pin_memory

`micro_writeback.py` rebuilds exactly what `DSAIndexerPoolHost` + `DSATokenToKVPool`
construct (78 layers, page_size 64, stride 8448, 7.33 GB host buffer) and calls the
same `transfer_kv_all_layer_mla` kernel.

Expected: every `mmap_*` variant aborts with

    Memory access fault by GPU node-2 ... on address <host VA>

and the address **equals the host pointer** printed just above it. `pin_memory` prints
`ALL OK` for head / tail / last page ranges.

Each faulting run kills its own process — that is the expected result, not a failure of
the harness. Run them one at a time.

---

## B. End-to-end: fix, zero faults, and read-back proof

### B1. Containers, kvd daemon, etcd

    scripts/boot.sh            # see the two-node bring-up it wraps
    # per node: container `agbench`; infera.kvd on /tmp/kvd/kvd.sock
    #           --max-bytes 64G --long-path /tmp/kvd-long --long-bytes 512G
    # prefill node only: etcd

Never `docker exec -d $CTR bash -lc '...'` — the detached login shell exits and takes
the child with it, leaving no process, no log, no error. Always stage a script file and
run `docker exec -d $CTR bash /the_script.sh`.

### B2. Apply the patch in **both** containers, and prove it landed in bytecode

    docker exec $CTR python /shared_nfs/<you>/patch_hicache_rocm_host_alloc.py
    # -> "[patch] applied to .../mem_cache/pool_host/common.py"
    # -> "[patch] removed N stale .pyc"

Verify **bytecode**, not source, after the module has been imported once:

    strings .../pool_host/__pycache__/common.cpython-310.pyc | grep -c GLM52_ROCM_HOST_ALLOC
    # want: 1

This is not ceremony. A stale `__pycache__` entry silently running unpatched bytecode
has invalidated a full experiment in this repo. The patch deliberately writes a real
module-level string literal (`GLM52_ROCM_HOST_ALLOC = "applied"`) rather than a comment,
because comments do not survive compilation and would make this check always return 0.

The patch is self-locating and idempotent; re-running prints `already applied`.

### B3. Boot both legs

    scripts/boot.sh prefill 262144 1     # kvd ON
    scripts/boot.sh decode  262144 0     # kvd OFF (operator instruction)

Cold start ~5-8 min: weights ~2 min, then tilelang JIT and DP cudagraph capture. Eight
live `sglang::scheduler_DP*` processes means it is working, not hung. Be patient.

### B4. Gate — criterion 1

Server logs contain binary bytes, so plain `grep` reports "binary file matches" and
`grep -c` returns 0. Use `strings <log> | grep`.

| row | prefill | decode |
|---|---|---|
| `ready to roll` | 1 | 1 |
| **`Memory access fault`** | **0** | **0** |
| `Scheduler hit an exception` | 0 | 0 |
| `infera-kvd adapter connected` | 8 | 0 (by design) |
| `Attached hybrid DSA pool stack` | 8 | — |
| `Errno 98` **after** the ready line | 0 | 0 |
| `disaggregation_decode_enable_radix_cache=True` | — | 1 |

`Errno 98` must be checked *after* the ready line: a `--kv-snapshot-port` collision
lets the leg log `ready to roll` and *then* die during etcd registration, so it looks
healthy and simply never appears in `/v1/workers`.

Then confirm registration through the router — **never probe a leg's own port, it hangs**:

    curl -s http://<prefill-ip>:8190/v1/workers    # want both workers "active"

### B5. Correctness under kvd

    docker exec $CTR python3 scripts/correctness.py http://127.0.0.1:8190

Expect short factual 4/4 and needle 5/5 at ~120K tokens, with `faults` still 0.

### B6. Read-back proof — criterion 2

A latency win proves **nothing**: sglang's in-GPU radix cache serves a repeated prefix
without ever touching L3. The only clean attribution is restart-and-replay. Three
conditions must hold at once:

    snapshot counters:
      docker exec $CTR python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock

    kill the ENGINE only -- the kvd daemon must survive:
      bash scripts/kill_engine.sh
      # it targets only `infera.engine.sglang` / `sglang::` PIDs and then prints the
      # survivors, so you can see the daemon is still there.

    wait for VRAM to actually drain (asynchronous KFD teardown, ~1-2 min):
      rocm-smi --showmemuse | grep VRAM%     # poll until ALL 8 read 0

    reboot prefill, then replay byte-identical prompts:
      scripts/boot.sh prefill 262144 1
      docker exec $CTR python3 scripts/replay_probe.py http://127.0.0.1:8190

    snapshot counters again.

**Do not restart the container to do this.** It would drop the runtime patch from B2
and empty nothing useful — the whole point is that the daemon's store outlives the
engine.

`replay_probe.py` rebuilds the filler from `random.Random(20260731)` with the same word
list and call order as `correctness.py`, so the prefix is identical. Any drift and the
radix prefix differs and the replay silently tests nothing.

---

## Expected output

**A2:** `mmap_*` → `Memory access fault ... on address <host VA>` matching the printed
host pointer; `pin_memory` → `ALL OK`.

**B4/B5:** 0 faults, 0 exceptions, 8 adapters, correctness 4/4 + 5/5.

**B6:**

    before                    after
    gets_total        0       gets_total   12,942     <- climbs
    hits_total        0       hits_total   12,942     <- climbs
    sets_total   12,942       sets_total   12,942     <- FLAT
    misses_total      0       misses_total      0

plus server-side `cached=120000` of `prompt_tok=120047` at every depth — an independent
path from the daemon's own counters, and the two must agree.

`sets_total` climbing instead of staying flat means the pages were re-stored, i.e. a
miss, i.e. the read path is not working. That is the failure signature to watch for.

## If it doesn't reproduce

See `notes.md` — refuted hypotheses, the `.pyc` trap, VRAM teardown timing, and the
`prefetch_capacity_limit` configuration that makes L3 write-only.
