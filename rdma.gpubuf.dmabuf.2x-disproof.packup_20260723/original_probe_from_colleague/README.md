# Ionic GPU-VRAM RDMA Registration Probe — cross-cluster verification

> **中文 TL;DR（给转发的人）**：我们怀疑老版 AMD Pensando **ionic 驱动/固件**导致 GPU 显存做 RDMA 注册有问题（裸 `ibv_reg_mr` 直接失败，dma-buf 能过但显存翻倍 2×）。你们集群是**新版 ionic**，请在一台机器的 ROCm 容器里跑一下 `bash run.sh`，把完整输出发回来即可。只读、单机、几秒钟、不需要 Mooncake、不需要对端。

---

## 1. Context / why we are asking

We run **PD (prefill/decode) disaggregated inference**. The KV cache is transferred **GPU→GPU across nodes over RDMA** (via Mooncake / Mori) on **AMD Pensando ionic** NICs (RoCEv2).

On **our** cluster every node runs an **OLD ionic stack**, and we have reproduced two hard problems on it:

| what we tested (on GPU VRAM) | result on our OLD stack |
|---|---|
| plain `ibv_reg_mr()` — the default path Mooncake/Mori use | **FAILS, errno 14 (EFAULT)** → KV registration cannot even start |
| `ibv_reg_dmabuf_mr()` — the modern GPUDirect/dma-buf path | **succeeds, BUT VRAM usage doubles (2× shadow copy)** — registering a 2 GiB buffer consumes 4 GiB |

Our OLD stack (all nodes identical):
- ionic driver `25.08.4.004`, ionic_rdma `25.08.4.004`
- NIC firmware `1.117.1-a-63`
- libionic userspace `54.0-149`
- kernel `6.8.0-107-generic`

**Your** cluster reportedly runs a **NEWER ionic stack** where PD/Mooncake works:
- ionic driver `26.01.30.001`
- NIC firmware `1.117.5-a-66`
- kernel `6.8.0-136-generic`

## 2. Hypothesis to confirm

> The **old ionic driver + firmware** is the root cause. On the newer 26.x stack we expect **`ibv_reg_mr(VRAM)` = OK** and **dma-buf = OK at 1× (no shadow)**.

Confirming this turns "probably a driver issue" into hard evidence for an infra upgrade ticket. If it does **not** reproduce as expected, that is equally valuable (it means the difference is something other than the driver — BIOS/kernel/PCIe — and we keep looking).

## 3. What the probe does

`ionic_vram_test.py` (pure `ctypes`, no Mooncake, no pyverbs needed):
1. **TEST B** — allocates a VRAM buffer, exports a dma-buf fd (`hipMemGetHandleForAddressRange`), calls **`ibv_reg_dmabuf_mr`**, and measures **free VRAM before/after** to detect the 2× shadow.
2. **TEST A** — allocates a small VRAM buffer and calls plain **`ibv_reg_mr`** (what Mooncake/Mori do by default).
3. Prints a `VERDICT` block.

It is **read-only and single-node**: it allocates a few GiB of VRAM, registers then immediately deregisters it, and frees everything. No RDMA partner, no disk writes, no effect on other jobs beyond transient VRAM use.

## 4. How to run

Run **inside a ROCm container that can see the ionic NICs** (the same kind of container where Mooncake works). Requirements: `python3`, `libibverbs.so.1`, `libamdhip64.so`, and ≥ a few GiB free VRAM on GPU 0.

```bash
cd ionic_vram_probe
bash run.sh                 # default: registers a 4 GiB buffer
# PROBE_GIB=2 bash run.sh   # use if free VRAM is tight
# PROBE_DEV=ionic_3 bash run.sh   # force a specific NIC (default = first ionic*)
```

If you are **not** already inside a suitable container, start one first (adjust the image name):

```bash
docker run --rm -it \
  --network=host \
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --group-add video --group-add render --cap-add=IPC_LOCK \
  -v "$PWD":/probe -w /probe \
  <YOUR_ROCM_IMAGE> bash
# then inside:  bash run.sh
```

## 5. How to interpret the result

First check the `ENV` block: it must show **driver `26.x` / firmware `1.117.5-*`**. If it shows `25.08` this node is *not* the newer stack — please find one that is.

Then the `VERDICT`:

| VERDICT line | meaning |
|---|---|
| `bare ... = OK` | **The key result.** Old ionic was the culprit; on 26.x Mooncake works with no code change. |
| `dma-buf ... = OK_1x` | The 2× VRAM-doubling is also fixed on 26.x (our old stack is `OK_2x`). |
| both as above | Strongest possible confirmation — proceed with the upgrade. |
| `bare = FAIL_14` **and** `dma-buf = OK_2x` (same as our old baseline) | It is **not** just the driver version; something else differs. Please note exact BIOS / kernel / PCIe settings. |

For reference, our OLD stack prints:
```
  bare ibv_reg_mr(VRAM) : FAIL_14
  dma-buf reg(VRAM)     : OK_2x
```

## 6. What to send back

Just paste the **entire output** of `run.sh` (the `ENV` block + TEST A + TEST B + `VERDICT`). That is all we need.

## 7. Files in this package
- `README.md` — this file.
- `ionic_vram_test.py` — the probe (ctypes; libibverbs + HIP).
- `run.sh` — prints driver/fw versions, then runs the probe.
