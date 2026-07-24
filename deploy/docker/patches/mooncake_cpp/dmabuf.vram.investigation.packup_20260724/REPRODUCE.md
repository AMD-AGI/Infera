# REPRODUCE

Cold-start steps to reproduce every result. Assumes cluster access
(root@149.28.124.225) and chi2798 with card1 (PCI 75:00.0) idle. All paths are
on the shared fs `/mnt/vast/c_huggingface` unless noted.

## 0. Stage the files
```bash
# from this packup dir, push src + scripts to the shared fs
scp src/dmabuf_verdict_mvp.cpp   root@149.28.124.225:/tmp/
scp scripts/*.sh                 root@149.28.124.225:/tmp/
ssh root@149.28.124.225 'scp /tmp/dmabuf_verdict_mvp.cpp /tmp/*.sh chi2798:/mnt/vast/c_huggingface/'
```

## 1. Fresh container (kill/run — NEVER stop/restart)
```bash
ssh root@149.28.124.225 'ssh chi2798 "
  docker rm -f dmabuf_probe 2>/dev/null;
  docker run -d --name dmabuf_probe --network host --ipc host \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video --group-add render -v /mnt/vast:/mnt/vast \
    rocm/jax-training:maxtext-v26.5 sleep infinity"'
```
Verify ionic is visible IN the container (must show ionic_0, not \"no pd\"):
```bash
ssh root@149.28.124.225 'ssh chi2798 "docker exec dmabuf_probe ibv_devices | head"'
```

## 2. Compile the MVP inside the container
```bash
ssh root@149.28.124.225 'ssh chi2798 "
  docker cp /mnt/vast/c_huggingface/dmabuf_verdict_mvp.cpp dmabuf_probe:/root/mvp.cpp &&
  docker exec dmabuf_probe bash -lc \
    \"cd /root && hipcc -O2 mvp.cpp -o mvp -libverbs -L/opt/rocm/lib -lhsa-runtime64\""'
```

## 3. THE decisive experiment — reordered (occupy 150G, then register 100G)
The shipped MVP is already the reordered version. Run it under the
handshake-synced host TTM sampler:
```bash
ssh root@149.28.124.225 'ssh chi2798 "bash /mnt/vast/c_huggingface/joint_reorder.sh dmabuf"'
```
EXPECT (see logs/ro_dmabuf.log):
- P2 occupier 150G: hip_free 37, TTM_free 35.8  (physical ~36G left)
- **P3 register 100G: \"registered 100/100 blocks\" SUCCESS, TTM_free 35.8 UNCHANGED**
- P3 hip_free = 0.00   (KFD double-counts the pin; available collapses)
=> registration allocates NO new physical VRAM (TTM unmoved, succeeds with only
   36G physical free), yet KFD double-counts the pinned pool so the reported
   available memory drops below what physically remains.

## 4. The one-click no-pin capability check (root cause, self-contained)
```bash
ssh root@149.28.124.225 'ssh chi2798 "bash /mnt/vast/c_huggingface/check_nopin_capability.sh ionic_0"'
```
This ONE script covers all layers and prints a PASS/FAIL verdict (exit 0 =
no-pin possible; exit 1 = pin forced). EXPECT on chi2798 (see
`results/nopin_verdict_chi2798.txt`):
- Layer1 NIC-ODP = **RED** (`rc_odp_caps: NO SUPPORT`) → pin forced
- Layer1b peermem = RED (no module) → `ibv_reg_mr` GPU-direct unavailable
- Layer2/3/4 = GREEN
- **RESULT: PIN FORCED — KFD double-counts pinned GPU mem → available shrinks.**

(The older split scripts `nopin_diag.sh` + `peermem_chk.sh` are kept for
reference; `check_nopin_capability.sh` supersedes both.)

## 5. (Optional) supporting experiments
Instrument calibration + earlier phase orderings are in the git history of
`src/dmabuf_verdict_mvp.cpp`; the key harnesses are shipped:
- `joint_hs.sh <mode>` — in-order handshake VRAM/GTT (logs/hs_dmabuf.log).
- `joint_kfd.sh <mode>` — same + dmesg KFD capture (came back silent).
- `nopin_diag.sh` — the ODP/config/driver/rdma-core 4-layer check.
- `verify_cfg.sh` — multi-source CONFIG_DMABUF_MOVE_NOTIFY verification.
- `bar_probe.sh` — GPU/NIC BAR sizes.

To re-run bare vs dmabuf for contrast, pass `bare` or `dmabuf` as the mode arg.

## 6. Teardown
```bash
ssh root@149.28.124.225 'ssh chi2798 "docker rm -f dmabuf_probe"'
```

## Ground-truth rule
Trust `hipMalloc` / registration **success or failure**. Treat every gauge
(hip_free, TTM, sysfs, rocm-smi, dmesg) as suspect until validated against a
known-true control (a change you are certain happened). See notes.md.
