# r03 — #33968 hardware validation: complete, with a real positive control

Measured 2026-08-24 on `crsuse2-m2m-237` (job 58799, 8x MI355X gfx950) inside
`infera-local:sglang-prverify-20260824`. This closes `plan.md` step 3, which had
been blocked since 2026-08-07 for want of a machine that reproduces the fault.

## The four links, all first-hand

### 1. The pointer identity fails here (`scripts/probe_host_devptr.py`)

```
[pin_memory]                same=True
[mmap + hipHostRegister]    same=False
[  + hipHostRegisterMapped] same=False
[  + Portable|Mapped]       same=False
```

Three consecutive runs, identical verdict, different VAs each time.
Details and the disproved amdgpu hypothesis: `probe_33968_positive.md`.

### 2. Stock dispatch really does pick the faulting allocator

`scripts/dispatch_check.py`, run against the container's own sglang tree.

| key | stock | patched |
|---|---|---|
| `"cuda"` | `alloc_with_host_register` | `alloc_with_pin_memory` |
| `torch.device("cuda:0")` | `alloc_with_host_register` | `alloc_with_pin_memory` |
| `torch.device("cuda")` | `alloc_with_host_register` | `alloc_with_pin_memory` |

This confirms the PR's secondary argument directly: the pools key the table with
a `torch.device` object, `torch.device("cuda:0")` is not dict-key-equal to
`"cuda"`, so those pools resolve through the **default** — which is why the PR
switches the default and not just the `"cuda"` entry.

### 3. The write-back kernel faults on stock, survives on patched

`scripts/writeback_repro.py` calls the same kernel the real path calls,
`sgl_kernel.kvcacheio.transfer_kv_all_layer_mla`, with `dst_layers` built exactly
as `pool_host/dsa.py:159` builds it — host `data_ptr()`s in a **device-resident**
`torch.uint64` tensor. One arm per allocation strategy, everything else identical.

```
arm=host_register   host VA == device pointer: False
                    -> Memory access fault by GPU node-2 on address 0x56a0af6b9000

arm=pin_memory      host VA == device pointer: True
                    -> kernel returned and synchronized without fault
                    -> data check: all copied pages carry the expected pattern
                    -> RESULT: PASS
```

Both arms reproduced on a second run (`logs/writeback_repro_237_repeat.log`).

### 4. The fault address is the host VA, not a coincidence

`scripts/addrcheck.py` prints both address families for the same buffers:

```
layer0: hostVA=0x5aa662b58b40..0x5aa662b60b40   devPtr=0x74b06fec8b40
layer1: hostVA=0x5aa6641b1340..0x5aa6641b9340   devPtr=0x74b06feb8340
layer2: hostVA=0x5aa6641b9fc0..0x5aa6641c1fc0   devPtr=0x74b06f850fc0
layer3: hostVA=0x5aa6641c2d00..0x5aa6641cad00   devPtr=0x74b06f840d00
```

Host VAs live in the `0x5...` heap range; the registered device mappings live in
`0x7...`. The faulting address `0x56a0af6b9000` is of the **host** form. The
kernel dereferenced the host VA — precisely what the commit message asserts:
*"the first HiCache write-back aborts with a GPU memory access fault at the host
address."*

## What this establishes, and what it does not

**Established.** On gfx950 with this software stack, the fault is real,
deterministic, reproduces on the stock allocator, and is fixed by the allocator
the PR selects. Stock faults; patched completes and the data arrives. That is the
positive-control A/B `plan.md` demanded, and it is the evidence #33968 lacked.

**Not established.** *Why* this machine reproduces and n06-33 did not. Two
variables differ (host ROCm 7.0.1 vs 7.2.0; container torch/hip build hash), and
they were not separated. The `plan.md` lead — amdgpu 6.14.14 — is **disproved**:
this cluster runs the same 6.14.14 and reproduces. No mechanism is claimed.

The repro is a targeted harness, not the full HiCache path: it drives the kernel
and the pointer table directly rather than standing up a model and forcing a real
write-back. It exercises the exact kernel, the exact pointer-table construction,
and the exact allocator dispatch, so it is sufficient for the causal claim; it
does not additionally prove the surrounding HiCache bookkeeping is correct.

## Reproducing

```bash
REPO=/home/yihou/dev/git/infera.upstream.pr.verify
IMG=infera-local:sglang-prverify-20260824
R=pr-verify.experiment/rounds/r03-newcluster/scripts
run() { spur exec 58799 bash -c "export HOME=/home/yihou && docker run --rm \
  --device /dev/dri --device /dev/kfd --group-add video --ipc=host \
  -v $REPO:$REPO -w $REPO $IMG python3 $*"; }

run $R/probe_host_devptr.py
run $R/writeback_repro.py --arm pin_memory      # PASS
run $R/writeback_repro.py --arm host_register   # must fault
```

Run each arm in its own process: a GPU memory access fault poisons the HIP
context for anything that follows it.
