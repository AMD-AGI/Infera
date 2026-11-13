// DRIVER619-20261113: torch.cuda.memory.CUDAPluggableAllocator backend that places HIP buffers meant for RDMA
// registration (SGLang's KV pools) in amdgpu GEM BOs instead of KFD BOs.
//
// amdgpu-dkms 6.18.4 .. 7.1.9 (fixed upstream in torvalds/linux 9131cc85d0de) charges a KFD BO that an RDMA NIC pins
// twice in the KFD allocation limit: once in kfd.vram_used, once in vram_pin_size. A BO created through the render
// node and imported into the process with hipImportExternalMemory is not in kfd.vram_used (import_obj_create does not
// reserve), so once pinned it is charged once. Each BO is pinned at allocation by a one-page dma-buf MR so KFD sees it
// before Mooncake registers it.
//
// Peer-memory registration (ibv_reg_mr) of these buffers must never happen: kfd_peerdirect.c amd_get_pages reads
// bo->kfd_bo->domain, and kfd_bo is NULL for a GEM BO (kernel NULL dereference). Each allocation therefore checks that
// HIP reports it as device memory and that hsa_amd_portable_export_dmabuf exports it (Mooncake then takes the dma-buf
// path, never ibv_reg_mr), and the process must run with MOONCAKE_DISABLE_HIP_DMABUF unset or 0 and
// MC_HIP_KFD_PIN_FIX=0 (the Mooncake anchor is an ibv_reg_mr). Any failed check aborts the process.
//
// HIP does not track the address range of imported memory (hipMemGetAddressRange returns a NULL base) and ROCr's
// hsa_amd_portable_export_dmabuf segfaults on it, which is how Mooncake exports HIP memory. Loaded with LD_PRELOAD, this
// library also answers those two calls for addresses inside its allocations (base, size, a dup of the BO's dma-buf fd)
// and refuses ibv_reg_mr / ibv_reg_mr_iova / ibv_reg_mr_iova2 there; every other address goes to the real function.
// The allocation self-check fails unless the library is preloaded.
//
// Environment: HIP_GEM_PIN_NIC  NIC used for the one-page pin (default: the first one that accepts it)
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <hip/hip_runtime_api.h>
#include <hsa/hsa_ext_amd.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define DRM_IOCTL_AMDGPU_GEM_CREATE 0xC0206440UL
#define DRM_IOCTL_PRIME_HANDLE_TO_FD 0xC00C642DUL
#define DRM_IOCTL_GEM_CLOSE 0x40086409UL
#define AMDGPU_GEM_DOMAIN_VRAM 0x4
#define ALIGN_2M(x) (((x) + (2ull << 20) - 1) & ~((2ull << 20) - 1))

struct gem_create {
    uint64_t bo_size, alignment, domains, domain_flags;
};
struct prime_handle {
    uint32_t handle, flags;
    int32_t fd;
};
struct gem_close {
    uint32_t handle, pad;
};

struct rec {
    void *ptr;
    size_t size;
    uint32_t handle;
    int dmabuf_fd, render_fd;
    hipExternalMemory_t ext;
    struct ibv_mr *anchor;
    struct rec *next;
};

static pthread_mutex_t mu = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
static struct rec *recs;
static int render_fds[64];
static struct ibv_pd *pin_pd;
static int checked_env;

static void die(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[hip_gem_alloc] fatal: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    abort();
}

static void check_env(void) {
    const char *d = getenv("MOONCAKE_DISABLE_HIP_DMABUF");
    const char *k = getenv("MC_HIP_KFD_PIN_FIX");
    if (d && strcmp(d, "0") != 0)
        die("MOONCAKE_DISABLE_HIP_DMABUF=%s would register GEM buffers through peer memory (kernel NULL dereference)", d);
    if (!k || strcmp(k, "0") != 0)
        die("MC_HIP_KFD_PIN_FIX must be 0: the Mooncake KFD pin anchor is a peer-memory ibv_reg_mr");
    const char *h = getenv("MC_DISABLE_HIP");
    if (!h || strcmp(h, "1") != 0)
        die("MC_DISABLE_HIP must be 1: GEM buffers have no HIP IPC handle, so Mooncake's HIP route cannot carry them");
    checked_env = 1;
}

static int render_fd(int dev) {
    if (dev < 0 || dev >= 64) return -1;
    if (render_fds[dev] > 0) return render_fds[dev];
    char bus[64], pat[160];
    if (hipDeviceGetPCIBusId(bus, sizeof bus, dev) != hipSuccess) return -1;
    for (char *c = bus; *c; ++c)
        if (*c >= 'A' && *c <= 'F') *c += 'a' - 'A';
    snprintf(pat, sizeof pat, "/sys/bus/pci/devices/%s/drm/renderD*", bus);
    glob_t g;
    if (glob(pat, 0, NULL, &g) || g.gl_pathc < 1) return -1;
    char node[96];
    snprintf(node, sizeof node, "/dev/dri/%s", strrchr(g.gl_pathv[0], '/') + 1);
    globfree(&g);
    int fd = open(node, O_RDWR | O_CLOEXEC);
    if (fd >= 0) render_fds[dev] = fd;
    return fd;
}

// The allocation holding addr, or NULL. Caller holds mu.
static struct rec *find_locked(const void *addr) {
    for (struct rec *r = recs; r; r = r->next)
        if ((uintptr_t)addr >= (uintptr_t)r->ptr && (uintptr_t)addr < (uintptr_t)r->ptr + r->size) return r;
    return NULL;
}

static int is_gem(const void *addr, size_t len) {
    pthread_mutex_lock(&mu);
    int hit = find_locked(addr) || (len && find_locked((const char *)addr + len - 1));
    pthread_mutex_unlock(&mu);
    return hit;
}

#define REAL(name, ...)                                                    \
    static __typeof__(name) *real_##name;                                  \
    if (!real_##name) real_##name = (__typeof__(name) *)dlsym(RTLD_NEXT, #name); \
    if (!real_##name) die("dlsym(RTLD_NEXT, %s)", #name)

hipError_t hipMemGetAddressRange(hipDeviceptr_t *pbase, size_t *psize, hipDeviceptr_t dptr) {
    pthread_mutex_lock(&mu);
    struct rec *r = find_locked(dptr);
    if (r) {
        if (pbase) *pbase = r->ptr;
        if (psize) *psize = r->size;
    }
    pthread_mutex_unlock(&mu);
    if (r) return hipSuccess;
    REAL(hipMemGetAddressRange);
    return real_hipMemGetAddressRange(pbase, psize, dptr);
}

hsa_status_t hsa_amd_portable_export_dmabuf(const void *ptr, size_t size, int *dmabuf, uint64_t *offset) {
    pthread_mutex_lock(&mu);
    struct rec *r = find_locked(ptr);
    int fd = -1;
    if (r && (uintptr_t)ptr + size <= (uintptr_t)r->ptr + r->size) {
        fd = dup(r->dmabuf_fd);
        if (offset) *offset = (uintptr_t)ptr - (uintptr_t)r->ptr;
    }
    pthread_mutex_unlock(&mu);
    if (r) {
        if (fd < 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
        *dmabuf = fd;
        return HSA_STATUS_SUCCESS;
    }
    REAL(hsa_amd_portable_export_dmabuf);
    return real_hsa_amd_portable_export_dmabuf(ptr, size, dmabuf, offset);
}

// Mooncake's HIP transport asks every registered buffer for an IPC handle, and HIP has none for imported memory
// (hipErrorInvalidValue, which also fails the whole registration and stays as the sticky last error). GEM buffers get a
// random handle instead: no KFD IPC object matches it, so a peer that tried to open it would fail rather than map other
// memory; MC_DISABLE_HIP=1 (checked at the first allocation) keeps Mooncake from routing to it at all.
hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t *handle, void *devPtr) {
    if (is_gem(devPtr, 1)) {
        static int warned;
        if (!warned++)
            fprintf(stderr, "[hip_gem_alloc] GEM memory %p has no HIP IPC handle; returning an unusable one\n", devPtr);
        int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
        if (fd < 0 || read(fd, handle, sizeof *handle) != (ssize_t)sizeof *handle) memset(handle, 0xA5, sizeof *handle);
        if (fd >= 0) close(fd);
        return hipSuccess;
    }
    REAL(hipIpcGetMemHandle);
    return real_hipIpcGetMemHandle(handle, devPtr);
}

#undef ibv_reg_mr
#undef ibv_reg_mr_iova
struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access) {
    if (is_gem(addr, length)) {
        fprintf(stderr, "[hip_gem_alloc] refused ibv_reg_mr of GEM memory %p (peer memory would oops)\n", addr);
        errno = EINVAL;
        return NULL;
    }
    REAL(ibv_reg_mr);
    return real_ibv_reg_mr(pd, addr, length, access);
}

struct ibv_mr *ibv_reg_mr_iova(struct ibv_pd *pd, void *addr, size_t length, uint64_t iova, int access) {
    if (is_gem(addr, length)) {
        fprintf(stderr, "[hip_gem_alloc] refused ibv_reg_mr_iova of GEM memory %p\n", addr);
        errno = EINVAL;
        return NULL;
    }
    REAL(ibv_reg_mr_iova);
    return real_ibv_reg_mr_iova(pd, addr, length, iova, access);
}

struct ibv_mr *ibv_reg_mr_iova2(struct ibv_pd *pd, void *addr, size_t length, uint64_t iova, unsigned int access) {
    if (is_gem(addr, length)) {
        fprintf(stderr, "[hip_gem_alloc] refused ibv_reg_mr_iova2 of GEM memory %p\n", addr);
        errno = EINVAL;
        return NULL;
    }
    REAL(ibv_reg_mr_iova2);
    return real_ibv_reg_mr_iova2(pd, addr, length, iova, access);
}

static struct ibv_pd *get_pin_pd(void) {
    if (pin_pd) return pin_pd;
    int n = 0;
    struct ibv_device **devs = ibv_get_device_list(&n);
    const char *want = getenv("HIP_GEM_PIN_NIC");
    for (int i = 0; i < n && !pin_pd; ++i) {
        if (want && strcmp(ibv_get_device_name(devs[i]), want) != 0) continue;
        struct ibv_context *ctx = ibv_open_device(devs[i]);
        if (!ctx) continue;
        pin_pd = ibv_alloc_pd(ctx);
        if (!pin_pd) ibv_close_device(ctx);
    }
    return pin_pd;
}

void *hip_gem_alloc(ssize_t size, int device, hipStream_t stream) {
    (void)stream;
    pthread_mutex_lock(&mu);
    if (!checked_env) check_env();
    size_t sz = ALIGN_2M((size_t)size);
    int prev = -1;
    hipGetDevice(&prev);
    if (hipSetDevice(device) != hipSuccess) die("hipSetDevice(%d)", device);
    int rfd = render_fd(device);
    if (rfd < 0) die("no render node for HIP device %d", device);
    struct gem_create gc = {sz, 2ull << 20, AMDGPU_GEM_DOMAIN_VRAM, 0};
    if (ioctl(rfd, DRM_IOCTL_AMDGPU_GEM_CREATE, &gc)) {
        int e = errno;
        pthread_mutex_unlock(&mu);
        fprintf(stderr, "[hip_gem_alloc] GEM_CREATE %zu bytes on device %d: %s\n", sz, device, strerror(e));
        return NULL;
    }
    uint32_t handle = (uint32_t)gc.bo_size;  // union: out.handle overlays in.bo_size
    struct prime_handle ph = {handle, O_CLOEXEC | O_RDWR, -1};
    if (ioctl(rfd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &ph)) die("PRIME_HANDLE_TO_FD: %s", strerror(errno));
    hipExternalMemoryHandleDesc hd;
    memset(&hd, 0, sizeof hd);
    hd.type = hipExternalMemoryHandleTypeOpaqueFd;
    hd.handle.fd = dup(ph.fd);  // HIP owns the fd it is given
    hd.size = sz;
    hipExternalMemory_t ext;
    hipError_t he = hipImportExternalMemory(&ext, &hd);
    if (he != hipSuccess) die("hipImportExternalMemory: %s", hipGetErrorString(he));
    hipExternalMemoryBufferDesc bd;
    memset(&bd, 0, sizeof bd);
    bd.size = sz;
    void *ptr = NULL;
    he = hipExternalMemoryGetMappedBuffer(&ptr, ext, &bd);
    if (he != hipSuccess) die("hipExternalMemoryGetMappedBuffer: %s", hipGetErrorString(he));

    struct rec *r = calloc(1, sizeof *r);
    *r = (struct rec){ptr, sz, handle, ph.fd, rfd, ext, NULL, recs};
    recs = r;

    // What Mooncake's RdmaContext::exportDmabuf will see: device memory, its own allocation, dma-buf exportable.
    hipPointerAttribute_t attr;
    memset(&attr, 0, sizeof attr);
    if (hipPointerGetAttributes(&attr, ptr) != hipSuccess || attr.type != hipMemoryTypeDevice)
        die("HIP does not report the GEM buffer as device memory (type %d): Mooncake would use ibv_reg_mr", attr.type);
    hipDeviceptr_t base = NULL;
    size_t range = 0;
    if (hipMemGetAddressRange(&base, &range, (hipDeviceptr_t)ptr) != hipSuccess || base != ptr || range < sz)
        die("hipMemGetAddressRange on the GEM buffer: base %p size %zu (is libhip_gem_alloc.so in LD_PRELOAD?)", base,
            range);
    int efd = -1;
    uint64_t eoff = 1;
    if (hsa_amd_portable_export_dmabuf(ptr, sz, &efd, &eoff) != HSA_STATUS_SUCCESS || eoff != 0)
        die("hsa_amd_portable_export_dmabuf of the GEM buffer (is libhip_gem_alloc.so in LD_PRELOAD?)");
    close(efd);

    struct ibv_pd *pd = get_pin_pd();
    if (!pd) die("no RDMA NIC for the pin%s%s", getenv("HIP_GEM_PIN_NIC") ? " " : "", getenv("HIP_GEM_PIN_NIC") ?: "");
    r->anchor = ibv_reg_dmabuf_mr(pd, 0, getpagesize(), (uint64_t)(uintptr_t)ptr, ph.fd, IBV_ACCESS_LOCAL_WRITE);
    if (!r->anchor) die("one-page dma-buf pin: %s", strerror(errno));
    if (prev >= 0) hipSetDevice(prev);
    pthread_mutex_unlock(&mu);
    return ptr;
}

void hip_gem_free(void *ptr, ssize_t size, int device, hipStream_t stream) {
    (void)size;
    pthread_mutex_lock(&mu);
    struct rec **pp = &recs, *r = NULL;
    for (; *pp; pp = &(*pp)->next)
        if ((*pp)->ptr == ptr) {
            r = *pp;
            *pp = r->next;
            break;
        }
    pthread_mutex_unlock(&mu);
    if (!r) {
        fprintf(stderr, "[hip_gem_alloc] free of unknown pointer %p\n", ptr);
        return;
    }
    int prev = -1;
    hipGetDevice(&prev);
    hipSetDevice(device);
    if (stream) hipStreamSynchronize(stream);
    hipDeviceSynchronize();
    ibv_dereg_mr(r->anchor);
    hipFree(ptr);
    hipDestroyExternalMemory(r->ext);
    close(r->dmabuf_fd);
    struct gem_close gcl = {r->handle, 0};
    ioctl(r->render_fd, DRM_IOCTL_GEM_CLOSE, &gcl);
    if (prev >= 0) hipSetDevice(prev);
    free(r);
}
