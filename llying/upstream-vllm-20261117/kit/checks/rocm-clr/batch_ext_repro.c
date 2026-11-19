// hipMemcpyBatchAsync on memory imported with hipImportExternalMemory, on a device other than 0.
//   batch_ext_repro DEVICE [external|plain]
// external (default): hipMalloc a buffer on DEVICE, export it as a dma-buf (hsa_amd_portable_export_dmabuf), import it
// back on DEVICE (hipImportExternalMemory + hipExternalMemoryGetMappedBuffer) and copy the import to pinned host memory
// with hipMemcpyBatchAsync on a DEVICE stream. plain: the same copy from the hipMalloc pointer.
// Prints what hipPointerGetAttributes reports, then "ok" or dies with the GPU memory fault.
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
