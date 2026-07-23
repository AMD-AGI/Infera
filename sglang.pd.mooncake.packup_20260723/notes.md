# Notes — root cause, dead-ends, gotchas

The whole point of this experiment turned into a debug: cross-node mooncake PD
broke on the new base. This file is the durable version of that story.

## The bug (what / why)

**Symptom:** DSv4 sglang PD 1P1D across two nodes. Both legs reach ready, etcd
auto-pairs prefill+decode, but every completion 500s. Prefill log:

    hip_transport.cpp:70 HipTransport: hipIpcOpenMemHandle failed (17 - invalid device pointer)
    -> KVTransferError: Failed to send kv chunk ... Decode instance could be dead

**Root cause:** the `v0.5.15.post1` base bundles Mooncake at commit `01d1eb2a` =
upstream **#2682 "Support rdma+hip multi-protocol segments for single-node
disaggregation"**. That commit installs a **HIP transport unconditionally**
(`transfer_engine_impl.cpp:403`, gated only by `#ifdef USE_HIP`, no runtime knob)
and registers the KV pool under BOTH `rdma` and `hip`. `MultiTransport::
selectTransport` then picks by fixed priority **hip(4) > rdma(2)**
(`multi_transport.cpp:468`) for the dual-registered pool — so even a **cross-node**
transfer uses HIP IPC. `hipIpcOpenMemHandle` only works same-host, so it fails.

**Why it worked before:** older bases predated #2682 — their mooncake had **no HIP
transport at all** (verified: `strings engine*.so | grep -c "HIP transport
installed"` = 0 on old infera sglang images, = 1 on the new base). So cross-node
PD used RDMA and just worked. This is a **regression the base upgrade introduced**,
not a pre-existing infera gap.

## The fix (how)

`patches/patch_mooncake_sglang.sh`: gate the HIP install behind
`if (getenv("MC_ENABLE_HIP_TRANSPORT"))` (default OFF) and rebuild the mooncake
engine in place against the base's own checkout. Wired into
`deploy/docker/Dockerfile.sglang` (`BUILD_MOONCAKE_GATE=1`). This mirrors the
existing `deploy/docker/patches/mooncake_cpp/transfer_engine_impl.diff` that
`Dockerfile.vllm` applies via a full mooncake rebuild — the sglang image never
needed it until #2682 landed in the base.

Differential proof: same image, HIP-gate ON vs OFF → 500 vs `Paris`. Single
variable, causal link closed.

## Dead-ends (do NOT retry these)

- **`MC_USE_HIP_IPC=0`** — looks like "disable HIP" but is NOT. Per
  `hip_transport.cpp:346`, it switches HIP from IPC mode to **fabric-memory** mode
  (needs VMM support). On gfx950 that path **segfaults** (scheduler dies exit -11
  at init). Wrong lever.
- **Any runtime env on stock mooncake** — the base's mooncake has NO env to skip
  the HIP install. The `MC_ENABLE_HIP_TRANSPORT` gate only exists AFTER our patch.
  So env alone can't fix it; you must rebuild.
- **Old `MC_DISABLE_HIP_TRANSPORT`** (set by `infera/engine/rocm_rdma_env.py:36`)
  — no longer exists in this mooncake → silent no-op. Misleading: it's in the
  worker env but does nothing.

## Build-time gotchas (fixing the Dockerfile step)

Two traps hit while wiring the patch into `docker build` (self-check caught both,
so no broken image shipped):

1. **Plain `ninja` does NOT rebuild the engine module.** The pybind
   `engine.cpython-*.so` is not in ninja's default target set — you must name it:
   `ninja engine.cpython-310-x86_64-linux-gnu.so`. Bare `ninja` "succeeds" in
   ~1 min but leaves the base's unpatched .so in place.
2. **`docker build` has NO GPU**, so ROCm's `amdgpu-arch` probe fails ("Failed to
   get device count") and cmake **drops the HIP engine target** from the build
   graph. Fix: pin the arch explicitly — `PYTORCH_ROCM_ARCH=gfx950` +
   `-DCMAKE_HIP_ARCHITECTURES=gfx950` — so the build is GPU-independent.
3. **Self-check false-negative:** `strings "$DEST" | grep -q PATTERN` under
   `set -o pipefail` — `grep -q` closes the pipe early → `strings` gets SIGPIPE
   (non-zero) → pipefail fails the `if` even when the pattern IS present. Use
   `grep -c ... | grep -qv '^0$'` (or capture to a var) instead of `grep -q` on a
   piped producer.

## RDMA / PD reproduction discipline

- **Reset ritual between every PD run:** `docker rm -f` both containers, reap GPU
  procs, wait until VRAM returns to ~0.3 GB/GPU baseline. Skipping this OOMs the
  next run or gives phantom failures. `scripts/pd_teardown.sh` does it.
- **Data-plane IP** must be the ionic-rail `10.2.x` on `enp193s0f1np1` — NOT the
  public NIC (get_ip()'s route-to-8.8.8.8 default picks the wrong one on a
  multi-homed host). The kit pins it via `SGLANG_HOST_IP`/`SGLANG_LOCAL_IP_NIC`.
- **SWA decode-radix gate:** DSv4 is a SWA model; the infera sglang decode leg
  must pass `--no-enable-kv-events` (the kit's `pd_mooncake/engine.sh` already
  does) or sglang rejects `--disaggregation-decode-enable-radix-cache` for SWA
  and decode dies. (Separate, older issue — mentioned so a reproducer doesn't
  re-hit it.)
- **Model path:** `-fixed` DSv4 has flipped to git-LFS stubs before. Always
  `stat -c %s` a shard (~13.8 GB real) + tokenizer.json (~6.3 MB) before trusting
  `ls`. Mount a parent covering any symlink target (`-v /mnt/vast:/mnt/vast`).
