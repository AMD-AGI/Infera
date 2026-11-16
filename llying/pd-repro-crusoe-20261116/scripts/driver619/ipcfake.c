// Test-only LD_PRELOAD (crusoe 11-16): Mooncake's HIP transport is installed even with MC_DISABLE_HIP=1 and calls
// hipIpcGetMemHandle for every registered buffer; on crusoe v3 nodes that fails at random for plain device memory
// ("invalid argument"), failing the whole batch_register_memory. When the real call fails, return an unusable handle
// so registration proceeds; with MC_DISABLE_HIP=1 the HIP route is never taken, the RDMA one is measured.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <hip/hip_runtime_api.h>
hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t *handle, void *devPtr) {
    static hipError_t (*real)(hipIpcMemHandle_t *, void *);
    if (!real) real = (hipError_t (*)(hipIpcMemHandle_t *, void *))dlsym(RTLD_NEXT, "hipIpcGetMemHandle");
    hipError_t r = real(handle, devPtr);
    if (r != hipSuccess) {
        static int warned;
        if (!warned++) fprintf(stderr, "[ipcfake] hipIpcGetMemHandle(%p) failed (%d); returning an unusable handle\n", devPtr, (int)r);
        memset(handle, 0xA5, sizeof *handle);
        return hipSuccess;
    }
    return r;
}
