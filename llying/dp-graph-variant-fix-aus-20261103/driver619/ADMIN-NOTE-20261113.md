# 草稿：给集群管理员 / AMD 驱动团队的说明（待用户过目，未发送）

实测数字标了"[待实测]"，等 6.19 / 6.16 节点上的最小复现跑完再填。

---

## 中文

**主题：amdgpu-dkms 6.19.14（试点驱动）把 RDMA 固定的显存算两次，大块显存注册给网卡后应用报假的显存不足；上游已修复，已发布的 DKMS 还没有**

各位好，

我们在 SMCI355 AUS 集群上跑 SGLang + Mooncake 的 PD 分离推理（GLM-5.2）。驱动升级到 amdgpu-dkms 6.19.14.31400100 的节点上，
推理服务每次起栈都会崩，升级前的 6.16.13 节点都正常。已知 6.19 节点是 n01-21、n02-25、n03-33、n04-25、n04-29。
我们查到根因在驱动，上游已经修复。想请你们帮忙给试点节点打上这个补丁，或者转给 AMD 驱动团队。

**现象**：推理服务要把每张卡约 110–190 GB 的 KV 缓存注册给 ionic RDMA 网卡（`ibv_reg_mr` 或 `ibv_reg_dmabuf_mr`）。
注册完之后，这个进程任何新的显存申请都失败，`hipStreamCreate` 返回显存不足，PyTorch 在 `c10::hip::HIPStream::stream()` 段错误。
实际显存还空着很多。

**根因**：`amd/amdgpu/amdgpu_amdkfd_gpuvm.c` 的 `amdgpu_amdkfd_reserve_mem_limit()`。
- 6.16.13：可分配上限是 `vram_size − 预留 − vram_pin_size + kfd.vram_pinned`；
- 6.18.4 起（我们核对了 31.10、31.20、31.30、31.40、31.40.1、31.50、31.60 的 DKMS 源码），`+ kfd.vram_pinned` 被删掉了。
  已经记在 `kfd.vram_used` 里的显存被网卡固定后，又从 `vram_pin_size` 里扣一次。

**上游修复**：[torvalds/linux `9131cc85d0de` "drm/amdkfd: correct vram_available calculation"](https://github.com/torvalds/linux/commit/9131cc85d0dedd8ba4cae60dbe5985d081551f30)
（cherry-pick 自 `894cbb31c50e`，作者 Zhen Han，AMD，2026-09-20，`Cc: stable`）。截至 31.60（7.1.9），repo.radeon.com 上已发布的 DKMS 都还没有包含它。

**附件**：
- `backport-9131cc85d0de-amdgpu-dkms-6.19.14.31400100.patch`：把上游补丁回移到集群现用的 DKMS 源码，11 处改动 10 处直接打上，1 处只是缩进不同。[待编译验证]
- `kfd_pin_repro.py`：单 GPU、单进程的最小复现，只依赖 HIP、HSA 和 libibverbs，几分钟跑完。11-13 在 n04-29（6.19.14）和 n04-33（6.16.13）上的结果：
  先 `hipMalloc` 再注册给网卡，然后看驱动还肯再分配多少（注册前两台都是 287 GiB）。

  | 注册的显存 | 方式 | 6.19.14 | 6.16.13 |
  |---|---|---|---|
  | 40 GiB | `ibv_reg_dmabuf_mr` | 剩 207 GiB（扣了 2 × 40） | 剩 207 GiB（扣了 2 × 40） |
  | 40 GiB | `ibv_reg_mr`（peer memory） | 剩 207 GiB（扣了 2 × 40） | 剩 247 GiB（扣 1 × 40） |
  | 150 GiB | `ibv_reg_mr`（peer memory） | 剩 0，`hipStreamCreate` 后段错误 | 剩 137 GiB，正常 |

  另外，peer memory 注册会让 `hipMemGetInfo` 报出远大于显存的数（如 22,487 GiB）：`amd_get_pages` 每注册一次就给 `kfd.vram_pinned` 加一整个 BO 的大小。
  这个问题上游补丁一并修了。

**请求**（任选其一）：
1. 在 6.19 试点节点上用附件补丁重建 amdgpu-dkms 并重载驱动；
2. 在 AMD 发布带这个修复的 DKMS 之前，保留一部分 6.16.13 节点，或者暂停继续升级；
3. 把这个问题转给 AMD 驱动团队，请他们把 `9131cc85d0de` 带进下一个 DKMS 版本。

谢谢！

---

## English

**Subject: amdgpu-dkms 6.19.14 (pilot driver) charges RDMA-pinned VRAM twice — fake OOM after registering large VRAM with a NIC; fixed upstream, not in any released DKMS**

Hi,

We run SGLang + Mooncake prefill/decode-disaggregated inference (GLM-5.2) on the SMCI355 AUS cluster. On every node upgraded to
amdgpu-dkms 6.19.14.31400100 (n01-21, n02-25, n03-33, n04-25, n04-29) the server crashes at start-up; the 6.16.13 nodes are fine.
We traced it to the driver, and the fix is already upstream. Could you patch the pilot nodes, or forward this to the AMD driver team?

**Symptom**: the server registers its KV cache, about 110–190 GB per GPU, with the ionic RDMA NICs (`ibv_reg_mr` or
`ibv_reg_dmabuf_mr`). After that every new device allocation in the process fails: `hipStreamCreate` returns out-of-memory, and
PyTorch segfaults in `c10::hip::HIPStream::stream()`, although most of the VRAM is free.

**Root cause**: `amdgpu_amdkfd_reserve_mem_limit()` in `amd/amdgpu/amdgpu_amdkfd_gpuvm.c`.
- 6.16.13: the limit is `vram_size − reserved − vram_pin_size + kfd.vram_pinned`.
- From 6.18.4 on (checked in the DKMS sources of 31.10, 31.20, 31.30, 31.40, 31.40.1, 31.50 and 31.60), the `+ kfd.vram_pinned`
  term is gone, so VRAM already counted in `kfd.vram_used` is subtracted again through `vram_pin_size` once a NIC pins it.

**Upstream fix**: [torvalds/linux `9131cc85d0de` "drm/amdkfd: correct vram_available calculation"](https://github.com/torvalds/linux/commit/9131cc85d0dedd8ba4cae60dbe5985d081551f30)
(cherry-picked from `894cbb31c50e`, Zhen Han, AMD, 2026-09-20, `Cc: stable`). None of the DKMS releases on repo.radeon.com through
31.60 (7.1.9) include it.

**Attached**:
- `backport-9131cc85d0de-amdgpu-dkms-6.19.14.31400100.patch`: the upstream fix on the cluster's DKMS source. 10 of 11 hunks apply
  as is; one differs only in whitespace. [build to be verified]
- `kfd_pin_repro.py`: a single-GPU, single-process repro that needs only HIP, HSA and libibverbs and runs in a few minutes.
  Results on n04-29 (6.19.14) and n04-33 (6.16.13), 2026-11-13: `hipMalloc`, register with the NIC, then the largest
  allocation the driver still accepts (287 GiB on both before registration).

  | Registered | Method | 6.19.14 | 6.16.13 |
  |---|---|---|---|
  | 40 GiB | `ibv_reg_dmabuf_mr` | 207 GiB left (2 × 40 charged) | 207 GiB left (2 × 40 charged) |
  | 40 GiB | `ibv_reg_mr` (peer memory) | 207 GiB left (2 × 40 charged) | 247 GiB left (1 × 40) |
  | 150 GiB | `ibv_reg_mr` (peer memory) | 0 left; segfault after `hipStreamCreate` | 137 GiB left; fine |

  Peer-memory registration also makes `hipMemGetInfo` report far more than the VRAM (e.g. 22,487 GiB): `amd_get_pages` adds the
  whole BO size to `kfd.vram_pinned` on every registration. The upstream patch fixes that as well.

**Request** (any one of these):
1. rebuild amdgpu-dkms on the 6.19 pilot nodes with the attached patch and reload the driver;
2. until AMD ships a DKMS with the fix, keep some nodes on 6.16.13, or pause the rollout;
3. forward this to the AMD driver team so `9131cc85d0de` lands in the next DKMS release.

Thanks!
