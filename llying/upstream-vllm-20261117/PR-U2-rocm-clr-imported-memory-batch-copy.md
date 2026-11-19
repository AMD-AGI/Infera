# Issue U2：ROCm clr——`hipMemcpyBatchAsync` 把非 0 设备上导入的外部内存的拷贝放到 device 0 执行

- 类型：issue（不是 PR）。目标：ROCm/rocm-systems（`projects/clr`，HIP runtime）。
- 背景：U2 的根因。Infera 的 GEM 分配器用 `hipImportExternalMemory` 导入 dma-buf BO 作 KV 缓存，vLLM 的 offload 用 `swap_blocks_batch`
  （即 `hipMemcpyBatchAsync`）拷贝；PP rank k > 0 的拷贝被放到 device 0 上执行，触发 GPU 访存故障（B 线 README 1.1–1.3）。
  Infera 侧已绕开（GEM BO 改用 HIP VMM 导入，B 线 1.4），所以这个 issue 不阻塞我们，只是报给上游。
- 查重（11-19）：`gh` 的 token 没有对 ROCm 组织做 SAML 授权，`gh search` / `gh repo view` 都被拒；改用网页搜索，没找到同样的 issue。
  最接近的是 [ROCm/rocm-systems#4817](https://github.com/ROCm/rocm-systems/issues/4817) 和
  [ROCm/rocm-systems#14258](https://github.com/ROCm/rocm-systems/issues/14258)（多卡下 `hipMemcpyAsync` 对可分页 / 文件映射主机内存的访存故障），
  机制不同。**提交前需要用户在浏览器里给 token 授权 ROCm 组织（或直接在网页上开 issue），并再查一次重。**
- 源码核对：rocm-systems develop 最新提交 `fca23cc6`（11-19 14:32 UTC）里问题仍在（引用的行号都按这个提交）。
- 状态：正文写完，**尚未提交到 GitHub。**

## 复现数据

| 项 | 内容 | 出处 |
|---|---|---|
| 最小复现 | `kit/checks/rocm-clr/batch_ext_repro.c`（74 行，不依赖 GEM / vLLM）：`hipMalloc` → `hsa_amd_portable_export_dmabuf` → `hipImportExternalMemory` 导回同一设备 → `hipExternalMemoryGetMappedBuffer` → `hipMemcpyBatchAsync` D2H 到锁页主机内存 | — |
| 结果（每个用例一个进程，单次） | device 0 导入：正常；device 1、3 导入：`Memory access fault by GPU node-2`，ROCr 报 `GPU index: 0, kernel: __amd_rocclr_copyBufferBatch`；device 1 上直接用 `hipMalloc` 指针：正常。`hipPointerGetAttributes` 对导入指针一律报 device 0 | J1 `rocm-clr-repro`（作业 298936，v3-057），`upstream-20261119/results/rocm-clr-repro/repro.txt` |
| 环境 | 镜像 `vllm/vllm-openai-rocm:v0.34.0`（ROCm 10.0.0，HIP 7.15.26333）；8 × MI355X（SPX / NPS1）；宿主 amdgpu 6.16.13、ROCm 7.2.0、Ubuntu 24.04.5、内核 6.8.12-322.crusoe.amd64、2 × EPYC 9575F（虚拟机直通） | `versions.txt`；宿主信息来自同型号节点 v3-092 的采集作业 299146（`results/platform-info/`）和 v3-006 的 `amd-smi` |
| 更早的隔离（B 线，v3-258，Infera 实验镜像，同为 ROCm 10.0） | 批量 D2H / H2D 对 device 1、3 上导入的 GEM BO 都在 device 0 上故障，与拷贝流在哪个设备无关；单次 `hipMemcpyAsync` 和 torch `copy_` 正常；改用 VMM 导入（`hipMemImportFromShareableHandle` + `hipMemMap`）后全部正常 | B 线 README 1.2，`results/b1/swapdev*.txt` |

## Issue body (English)

**Title:** [Issue]: hipMemcpyBatchAsync runs copies of memory imported with hipImportExternalMemory on device 0 (memory access fault when the import lives on another device)

### Problem Description

`hipMemcpyBatchAsync` picks the device that executes each copy from the device memory object's
`getUserData().deviceId`. Memory imported with `hipImportExternalMemory` + `hipExternalMemoryGetMappedBuffer` never gets
`deviceId` set, so it stays at its default of 0. A batch copy from (or to) such memory on any device other than 0 is
therefore enqueued on device 0's null stream, where the buffer is not mapped, and the GPU faults:

```
Memory access fault by GPU node-2 (Agent handle: 0x63d3fa95c1c0) on address 0x7707dee02000. Reason: Unknown.
[... GPU index: 0, faulting addr: 0x7707dee02000, kernel: __amd_rocclr_copyBufferBatch]
```

The same copy works when the import lives on device 0, when the source is a plain `hipMalloc` pointer on the same
device, and when it is done with a single `hipMemcpyAsync`. `hipPointerGetAttributes` also reports device 0 for the
imported pointer, for the same reason.

We hit this with vLLM and a custom allocator that places the KV cache in dma-buf imports: vLLM offloads KV blocks to
pinned host memory with `hipMemcpyBatchAsync` on each pipeline-parallel rank's own device, and every rank other than the
first faulted on its first offload.

### Operating System

Ubuntu 24.04.5 LTS (host, VM with GPU passthrough), kernel 6.8.12; the reproducer runs in the
`vllm/vllm-openai-rocm:v0.34.0` container.

### CPU

2 x AMD EPYC 9575F 64-Core Processor

### GPU

8 x AMD Instinct MI355X (gfx950, SPX / NPS1)

### ROCm Version

ROCm 10.0.0 in the container (HIP 7.15.26333); host amdgpu driver 6.16.13 (host ROCm 7.2.0). The code paths below are
unchanged on rocm-systems `develop` at `fca23cc6` (2026-11-19).

### ROCm Component

clr (HIP runtime)

### Steps to Reproduce

`batch_ext_repro.c`:

```c
// hipMemcpyBatchAsync on memory imported with hipImportExternalMemory, on a device other than 0.
//   batch_ext_repro DEVICE [external|plain]
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
#include <hsa/hsa_ext_amd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                                    \
    do {                                                                            \
        hipError_t e_ = (x);                                                        \
        if (e_ != hipSuccess) {                                                     \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

int main(int argc, char **argv) {
    int dev = argc > 1 ? atoi(argv[1]) : 1;
    int external = !(argc > 2 && strcmp(argv[2], "plain") == 0);
    const size_t sz = 64ull << 20, n = 64, chunk = sz / n;
    CHECK(hipSetDevice(dev));
    void *buf = NULL;
    CHECK(hipMalloc(&buf, sz));
    CHECK(hipMemset(buf, 7, sz));
    CHECK(hipDeviceSynchronize());
    void *src = buf;
    if (external) {
        int fd = -1;
        uint64_t off = 0;
        if (hsa_amd_portable_export_dmabuf(buf, sz, &fd, &off) != HSA_STATUS_SUCCESS || off != 0) {
            fprintf(stderr, "dma-buf export failed\n");
            return 1;
        }
        hipExternalMemoryHandleDesc hd;
        memset(&hd, 0, sizeof hd);
        hd.type = hipExternalMemoryHandleTypeOpaqueFd;
        hd.handle.fd = fd;
        hd.size = sz;
        hipExternalMemory_t ext;
        CHECK(hipImportExternalMemory(&ext, &hd));
        hipExternalMemoryBufferDesc bd;
        memset(&bd, 0, sizeof bd);
        bd.size = sz;
        CHECK(hipExternalMemoryGetMappedBuffer(&src, ext, &bd));
    }
    hipPointerAttribute_t at;
    memset(&at, 0, sizeof at);
    CHECK(hipPointerGetAttributes(&at, src));
    printf("current device %d, %s pointer %p: hipPointerGetAttributes type %d device %d\n", dev,
           external ? "imported" : "hipMalloc", src, at.type, at.device);
    char *host = NULL;
    CHECK(hipHostMalloc((void **)&host, sz, 0));
    memset(host, 0, sz);
    hipStream_t s;
    CHECK(hipStreamCreate(&s));
    void *srcs[64], *dsts[64];
    size_t sizes[64], fail = 0;
    for (size_t i = 0; i < n; ++i) {
        srcs[i] = (char *)src + i * chunk;
        dsts[i] = host + i * chunk;
        sizes[i] = chunk;
    }
    fflush(stdout);
    CHECK(hipMemcpyBatchAsync(dsts, srcs, sizes, n, NULL, NULL, 0, &fail, s));
    CHECK(hipStreamSynchronize(s));
    printf("batch D2H ok, host[0]=%d host[last]=%d\n", host[0], host[sz - 1]);
    return 0;
}
```

Build and run (one process per case, since a fault kills the process):

```bash
gcc -O2 -D__HIP_PLATFORM_AMD__ -I$ROCM_PATH/include batch_ext_repro.c -o repro \
    -L$ROCM_PATH/lib -lamdhip64 -lhsa-runtime64 -Wl,-rpath,$ROCM_PATH/lib
for c in "0 external" "1 external" "3 external" "1 plain"; do echo "== $c"; ./repro $c; done
```

Output:

```
== 0 external
current device 0, imported pointer 0x7ea7ed000000: hipPointerGetAttributes type 2 device 0
batch D2H ok, host[0]=7 host[last]=7
== 1 external
current device 1, imported pointer 0x7707dd000000: hipPointerGetAttributes type 2 device 0
Warning: Queue error - HSA_STATUS_ERROR_MEMORY_FAULT
:0:rocdevice.cpp            :4030: 1669358727271 us:  [host: ..., GPU index: 0, faulting addr: 0x7707dee02000, kernel: __amd_rocclr_copyBufferBatch]
Memory access fault by GPU node-2 (Agent handle: 0x63d3fa95c1c0) on address 0x7707dee02000. Reason: Unknown.
== 3 external
current device 3, imported pointer 0x7d756d000000: hipPointerGetAttributes type 2 device 0
Warning: Queue error - HSA_STATUS_ERROR_MEMORY_FAULT
:0:rocdevice.cpp            :4030: 1669362132492 us:  [host: ..., GPU index: 0, faulting addr: 0x7d7570a00000, kernel: __amd_rocclr_copyBufferBatch]
Memory access fault by GPU node-2 (Agent handle: 0x58afaf20e1c0) on address 0x7d7570a00000. Reason: Unknown.
== 1 plain
current device 1, hipMalloc pointer 0x70651ee00000: hipPointerGetAttributes type 2 device 1
batch D2H ok, host[0]=7 host[last]=7
```

KFD node-2 is device 0 on this machine. In a larger test the same happens for H2D batch copies, and regardless of
which device the stream belongs to; single `hipMemcpyAsync` copies of the same imported memory work.

### Additional Information

Where it comes from (rocm-systems `develop` at `fca23cc6`, `projects/clr`):

- `amd::Memory::UserData::deviceId` defaults to 0
  ([`rocclr/platform/memory.hpp#L164`](https://github.com/ROCm/rocm-systems/blob/fca23cc6cb11fc5618e6d2d49c31bc43b30f559a/projects/clr/rocclr/platform/memory.hpp#L164)).
- `hipImportExternalMemory` creates the `amd::ExternalBuffer` in the current device's context, and
  `hipExternalMemoryGetMappedBuffer` creates a view, gets its device memory for the current device and adds it to
  `MemObjMap`; neither sets `deviceId`
  ([`hip_memory.cpp#L192-L261`](https://github.com/ROCm/rocm-systems/blob/fca23cc6cb11fc5618e6d2d49c31bc43b30f559a/projects/clr/hipamd/src/hip_memory.cpp#L192-L261)).
  By contrast `hipIpcOpenMemHandle` sets it to the current device
  ([`#L4178`](https://github.com/ROCm/rocm-systems/blob/fca23cc6cb11fc5618e6d2d49c31bc43b30f559a/projects/clr/hipamd/src/hip_memory.cpp#L4178)),
  and so does `hipMemImportFromShareableHandle`
  ([`hip_vm.cpp#L351`](https://github.com/ROCm/rocm-systems/blob/fca23cc6cb11fc5618e6d2d49c31bc43b30f559a/projects/clr/hipamd/src/hip_vm.cpp#L351)).
- The batch copy groups device<->host copies by the device memory's `deviceId`
  ([`#L3296-L3303`](https://github.com/ROCm/rocm-systems/blob/fca23cc6cb11fc5618e6d2d49c31bc43b30f559a/projects/clr/hipamd/src/hip_memory.cpp#L3296-L3303),
  and `#L3326` / `#L3332` for write / read buffers), and `EnqueueBatchCommands` sends every group whose device differs
  from the stream's to that device's null stream
  ([`#L2971-L2983`](https://github.com/ROCm/rocm-systems/blob/fca23cc6cb11fc5618e6d2d49c31bc43b30f559a/projects/clr/hipamd/src/hip_memory.cpp#L2971-L2983)).
  For an import on device 1 that is device 0, which has no mapping for it.
- `hipPointerGetAttributes` reports the same field
  ([`#L4281`](https://github.com/ROCm/rocm-systems/blob/fca23cc6cb11fc5618e6d2d49c31bc43b30f559a/projects/clr/hipamd/src/hip_memory.cpp#L4281)),
  which is why the imported pointer shows up as device 0.

Suggested fix: set `getUserData().deviceId` to the current device in `hipImportExternalMemory` (on the external buffer)
and in `hipExternalMemoryGetMappedBuffer` (on the view), as `hipIpcOpenMemHandle` does. Alternatively, the batch path
could take the device from the memory object's context rather than from `deviceId`.

Workaround: import the dma-buf with the VMM API (`hipMemImportFromShareableHandle` + `hipMemAddressReserve` /
`hipMemMap` / `hipMemSetAccess`), which records the device; batch copies then run on the right device (in our test
51 GB/s for 5,120 random 42 KiB descriptors, the same as plain device memory).
