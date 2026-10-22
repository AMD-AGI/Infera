// t_hipxfer.yihou.c -- local smoke test for hipxfer_trace.yihou.c: mimics Mooncake HipTransport::submitTransfer
// (78 x 4.5 MiB hipMemcpyAsync(hipMemcpyDefault) GPU0 -> GPU1 over 2 round-robin streams, hipEventRecord after
// each, then hipEventSynchronize on each), 5 batches.
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
#include <stdio.h>

int fakemc_run(void) {
    const size_t len = 4608 << 10;
    const int n = 78;
    char *src, *dst;
    hipSetDevice(1);
    if (hipMalloc((void**)&dst, len * n) != hipSuccess) return 1;
    hipSetDevice(0);
    if (hipMalloc((void**)&src, len * n) != hipSuccess) return 1;
    hipDeviceEnablePeerAccess(1, 0);
    hipStream_t st[2];
    for (int i = 0; i < 2; i++) hipStreamCreateWithFlags(&st[i], hipStreamNonBlocking);
    hipEvent_t ev[78];
    for (int i = 0; i < n; i++) hipEventCreateWithFlags(&ev[i], hipEventDisableTiming);
    for (int b = 0; b < 5; b++) {
        for (int i = 0; i < n; i++) {
            hipMemcpyAsync(dst + i * len, src + i * len, len, hipMemcpyDefault, st[i & 1]);
            hipEventRecord(ev[i], st[i & 1]);
        }
        for (int i = 0; i < n; i++) hipEventSynchronize(ev[i]);
    }
    printf("done %zu bytes x 5\n", len * n);
    return 0;
}
