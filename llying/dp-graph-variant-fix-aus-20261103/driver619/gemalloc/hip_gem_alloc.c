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
// Environment: HIP_GEM_PIN_NIC  NIC used for the one-page pin (default: the first one that accepts it)
#define _GNU_SOURCE
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

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
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

    // What Mooncake's RdmaContext::exportDmabuf will see: device memory, its own allocation, dma-buf exportable.
    hipPointerAttribute_t attr;
    memset(&attr, 0, sizeof attr);
    if (hipPointerGetAttributes(&attr, ptr) != hipSuccess || attr.type != hipMemoryTypeDevice)
        die("HIP does not report the GEM buffer as device memory (type %d): Mooncake would use ibv_reg_mr", attr.type);
    hipDeviceptr_t base = NULL;
    size_t range = 0;
    if (hipMemGetAddressRange(&base, &range, (hipDeviceptr_t)ptr) != hipSuccess || base != ptr || range < sz)
        die("hipMemGetAddressRange on the GEM buffer: base %p size %zu", base, range);
    int efd = -1;
    uint64_t eoff = 0;
    if (hsa_amd_portable_export_dmabuf(ptr, sz, &efd, &eoff) != HSA_STATUS_SUCCESS)
        die("hsa_amd_portable_export_dmabuf of the GEM buffer failed: Mooncake could not register it");
    hsa_amd_portable_close_dmabuf(efd);

    struct ibv_pd *pd = get_pin_pd();
    if (!pd) die("no RDMA NIC for the pin%s%s", getenv("HIP_GEM_PIN_NIC") ? " " : "", getenv("HIP_GEM_PIN_NIC") ?: "");
    struct ibv_mr *mr = ibv_reg_dmabuf_mr(pd, 0, getpagesize(), (uint64_t)(uintptr_t)ptr, ph.fd, IBV_ACCESS_LOCAL_WRITE);
    if (!mr) die("one-page dma-buf pin: %s", strerror(errno));

    struct rec *r = calloc(1, sizeof *r);
    *r = (struct rec){ptr, sz, handle, ph.fd, rfd, ext, mr, recs};
    recs = r;
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
