/* LD_PRELOAD shim for the UMBP standalone server (MORI v1.2.3.post1): back its DRAM pool with HIP VMM host memory
 * (hipMemCreate on a host NUMA node), the way vLLM places the Engram tables (VLLM_ROCM_ENGRAM_HIP_VMM=1).
 *
 * UMBP allocates the pool with one mmap(MAP_ANONYMOUS | MAP_PRIVATE) and registers it whole with
 * hipHostRegister(Mapped | Portable). KFD tracks such memory as userptr: when the kernel migrates or write-protects any
 * page of it (compaction, khugepaged), KFD evicts every GPU queue of the server and revalidates the whole pool before
 * restoring them, and prefill finds UMBP unavailable meanwhile (EXPERIMENT 4.6-4.11). hipHostMalloc memory is userptr
 * too (4.11: the hipHostMalloc pool still had migration-triggered evictions). HIP VMM host memory is allocated by the
 * driver (TTM pages, not on the kernel's movable LRU), so nothing the kernel does to process pages reaches it; the
 * Engram tables in it have had no eviction since hangC2.
 *
 * Layout: the pool, rounded up to the allocation granularity, is created in UMBP_VMM_CHUNK_GB (2) GiB handles (the
 * last one smaller). The GPU sees it at one reserved device address
 * (hipMemMap of every handle, hipMemSetAccess for every visible device); the CPU sees it at a second address where
 * each handle's exported dmabuf FD is mmapped (HIP gives no CPU access to host VMM memory otherwise). UMBP keeps the
 * CPU address. For that range:
 *   hipHostRegister / hipHostUnregister succeed without doing anything; madvise is a no-op (the pages are resident);
 *   hipHostGetDevicePointer returns the device address (the gather kernel's alias, the same on every device);
 *   hipMemcpy / hipMemcpyAsync with an end in the pool use its device address and a device-to-device kind, so copies
 *   take the GPU path instead of HIP's pageable staging; a host-to-host copy is left alone;
 *   munmap of the whole range unmaps and releases everything.
 * Only an mmap that looks exactly like the pool is taken: addr NULL, fd -1, PROT_READ | PROT_WRITE,
 * flags MAP_ANONYMOUS | MAP_PRIVATE, length >= UMBP_VMM_MIN_GB GiB (64). If any HIP step fails, everything created so
 * far is released and the mmap goes through unchanged (the original behaviour), with a log line.
 *   UMBP_VMM_NUMA: host NUMA node of the pool (default: the node of the first visible GPU, else 0), or "interleave":
 *   chunk i on the i-th (mod n) node that has memory. All 418 GiB on the GPUs' node (which also holds the Engram
 *   tables) left that node so little movable memory that compaction there rose ~6x and reached the engines' own
 *   pinned buffers (EXPERIMENT 4.11, hangE).
 *   gcc -shared -fPIC -O2 -o umbp_vmm_pool.so umbp_vmm_pool.c -ldl -lpthread
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

typedef int hipError_t;
enum { kHipSuccess = 0 };
enum { kMemcpyHostToHost = 0, kMemcpyHostToDevice = 1, kMemcpyDeviceToHost = 2, kMemcpyDeviceToDevice = 3,
       kMemcpyDefault = 4 };
#define MAX_POOLS 4
#define MAX_CHUNKS 4096

/* hipMemAllocationProp and friends, as in vllm/models/deepseek_v41/amd/ops/engram_host_table.py. */
typedef struct { int type; int id; } HipLocation;
typedef struct {
    int type;                /* hipMemAllocationTypePinned = 1 */
    int requestedHandleTypes; /* hipMemHandleTypePosixFileDescriptor = 1 */
    HipLocation location;    /* hipMemLocationTypeHostNuma = 3 */
    void *win32HandleMetaData;
    unsigned allocFlags;
} HipAllocationProp;
typedef struct { HipLocation location; int flags; } HipAccessDesc; /* device = 1, read-write = 3 */

typedef struct {
    uintptr_t cpu, gpu;
    size_t len, padded, chunk; /* len: what the caller asked for; padded: len rounded up to the granularity */
    int nhandles;
    void **handles;
} Pool;

static Pool pools[MAX_POOLS];
static int npools;
static pthread_mutex_t pools_lock = PTHREAD_MUTEX_INITIALIZER;
/* Set while this shim calls into HIP, whose own mmaps / copies must go straight through. */
static __thread int in_hip;

static void note(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void note(const char *fmt, ...)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    gmtime_r(&ts.tv_sec, &tm);
    char head[64];
    strftime(head, sizeof head, "[%Y-%m-%d %H:%M:%S]", &tm);
    fprintf(stderr, "%s [umbp_vmm_pool] ", head);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
}

static long parse_num(const char *env, long fallback)
{
    const char *s = getenv(env);
    if (!s || !*s)
        return fallback;
    char *end;
    long v = strtol(s, &end, 10);
    return *end ? fallback : v;
}

static void *hip_sym(const char *name)
{
    return dlsym(RTLD_DEFAULT, name);
}

static void *(*real_mmap_fn(void))(void *, size_t, int, int, int, off_t)
{
    static void *(*fn)(void *, size_t, int, int, int, off_t);
    if (!fn)
        fn = (void *(*)(void *, size_t, int, int, int, off_t))dlsym(RTLD_NEXT, "mmap");
    return fn;
}

static int (*real_munmap_fn(void))(void *, size_t)
{
    static int (*fn)(void *, size_t);
    if (!fn)
        fn = (int (*)(void *, size_t))dlsym(RTLD_NEXT, "munmap");
    return fn;
}

/* Pool containing [addr, addr + len) at its CPU address, or NULL. Caller holds pools_lock. */
static Pool *find_locked(const void *addr, size_t len)
{
    uintptr_t a = (uintptr_t)addr;
    for (int i = 0; i < npools; i++)
        if (a >= pools[i].cpu && len <= pools[i].len && a - pools[i].cpu <= pools[i].len - len)
            return &pools[i];
    return NULL;
}

static int in_pool(const void *addr, size_t len)
{
    pthread_mutex_lock(&pools_lock);
    Pool *p = find_locked(addr, len);
    pthread_mutex_unlock(&pools_lock);
    return p != NULL;
}

/* Device address of a CPU address in a pool (the whole [addr, addr + len) inside it), else 0. */
static uintptr_t to_gpu(const void *addr, size_t len)
{
    pthread_mutex_lock(&pools_lock);
    Pool *p = find_locked(addr, len ? len : 1);
    uintptr_t g = p ? p->gpu + ((uintptr_t)addr - p->cpu) : 0;
    pthread_mutex_unlock(&pools_lock);
    return g;
}

/* Nodes with memory, in order; returns how many (at most max). */
static int memory_nodes(int *nodes, int max)
{
    int n = 0;
    for (int i = 0; i < 64 && n < max; i++) {
        char path[96];
        snprintf(path, sizeof path, "/sys/devices/system/node/node%d/meminfo", i);
        FILE *f = fopen(path, "r");
        if (!f)
            continue;
        char line[256];
        long total = 0;
        while (fgets(line, sizeof line, f))
            if (sscanf(line, "Node %*d MemTotal: %ld", &total) == 1)
                break;
        fclose(f);
        if (total > 0)
            nodes[n++] = i;
    }
    return n;
}

static int gpu_numa_node(void)
{
    long forced = parse_num("UMBP_VMM_NUMA", -1);
    if (forced >= 0)
        return (int)forced;
    hipError_t (*bus_id)(char *, int, int) = (hipError_t(*)(char *, int, int))hip_sym("hipDeviceGetPCIBusId");
    char id[64] = {0};
    if (!bus_id || bus_id(id, sizeof id, 0) != kHipSuccess)
        return 0;
    for (char *c = id; *c; c++)
        if (*c >= 'A' && *c <= 'F')
            *c = (char)(*c - 'A' + 'a');
    char path[128];
    snprintf(path, sizeof path, "/sys/bus/pci/devices/%s/numa_node", id);
    FILE *f = fopen(path, "r");
    int node = 0;
    if (f) {
        if (fscanf(f, "%d", &node) != 1 || node < 0)
            node = 0;
        fclose(f);
    }
    return node;
}

typedef struct {
    hipError_t (*create)(void **, size_t, const HipAllocationProp *, unsigned long long);
    hipError_t (*export_fd)(void *, void *, int, unsigned long long);
    hipError_t (*release)(void *);
    hipError_t (*reserve)(void **, size_t, size_t, void *, unsigned long long);
    hipError_t (*addr_free)(void *, size_t);
    hipError_t (*map)(void *, size_t, size_t, void *, unsigned long long);
    hipError_t (*unmap)(void *, size_t);
    hipError_t (*set_access)(void *, size_t, const HipAccessDesc *, size_t);
    hipError_t (*granularity)(size_t *, const HipAllocationProp *, int);
    hipError_t (*device_count)(int *);
    hipError_t (*get_last_error)(void);
} Vmm;

static int load_vmm(Vmm *v)
{
    v->create = (hipError_t(*)(void **, size_t, const HipAllocationProp *, unsigned long long))hip_sym("hipMemCreate");
    v->export_fd = (hipError_t(*)(void *, void *, int, unsigned long long))hip_sym("hipMemExportToShareableHandle");
    v->release = (hipError_t(*)(void *))hip_sym("hipMemRelease");
    v->reserve = (hipError_t(*)(void **, size_t, size_t, void *, unsigned long long))hip_sym("hipMemAddressReserve");
    v->addr_free = (hipError_t(*)(void *, size_t))hip_sym("hipMemAddressFree");
    v->map = (hipError_t(*)(void *, size_t, size_t, void *, unsigned long long))hip_sym("hipMemMap");
    v->unmap = (hipError_t(*)(void *, size_t))hip_sym("hipMemUnmap");
    v->set_access = (hipError_t(*)(void *, size_t, const HipAccessDesc *, size_t))hip_sym("hipMemSetAccess");
    v->granularity = (hipError_t(*)(size_t *, const HipAllocationProp *, int))hip_sym("hipMemGetAllocationGranularity");
    v->device_count = (hipError_t(*)(int *))hip_sym("hipGetDeviceCount");
    v->get_last_error = (hipError_t(*)(void))hip_sym("hipGetLastError");
    return v->create && v->export_fd && v->release && v->reserve && v->addr_free && v->map && v->unmap &&
           v->set_access && v->granularity && v->device_count;
}

static size_t chunk_at(const Pool *p, size_t off)
{
    return p->padded - off < p->chunk ? p->padded - off : p->chunk;
}

/* Undo a partly or fully built pool. Caller has set in_hip. */
static void destroy(const Vmm *v, Pool *p, size_t mapped_gpu)
{
    if (p->cpu)
        real_munmap_fn()((void *)p->cpu, p->padded);
    for (size_t off = 0; off < mapped_gpu; off += chunk_at(p, off))
        v->unmap((void *)(p->gpu + off), chunk_at(p, off));
    if (p->gpu)
        v->addr_free((void *)p->gpu, p->padded);
    for (int i = 0; i < p->nhandles; i++)
        v->release(p->handles[i]);
    free(p->handles);
    if (v->get_last_error)
        (void)v->get_last_error();
}

/* Build a pool of len bytes; on success fill *out and return 0, else return -1 with a reason. */
static int build(size_t len, Pool *out, char *why, size_t whylen)
{
    Vmm v;
    if (!load_vmm(&v)) {
        snprintf(why, whylen, "HIP VMM symbols missing");
        return -1;
    }
    int nodes[64], nnodes = 0;
    const char *policy = getenv("UMBP_VMM_NUMA");
    int interleave = policy && !strcmp(policy, "interleave");
    if (interleave)
        nnodes = memory_nodes(nodes, 64);
    if (nnodes == 0) {
        nodes[0] = gpu_numa_node();
        nnodes = 1;
        interleave = 0;
    }
    int node = nodes[0];
    HipAllocationProp prop = {1, 1, {3, node}, NULL, 0};
    size_t gran = 0;
    hipError_t e = v.granularity(&gran, &prop, 1);
    if (e != kHipSuccess || gran == 0) {
        snprintf(why, whylen, "hipMemGetAllocationGranularity error %d", e);
        return -1;
    }
    size_t chunk = (size_t)parse_num("UMBP_VMM_CHUNK_GB", 2) << 30;
    chunk = (chunk + gran - 1) / gran * gran;
    Pool p = {0};
    p.len = len;
    p.padded = (len + gran - 1) / gran * gran;
    p.chunk = chunk;
    size_t n = (p.padded + chunk - 1) / chunk, mapped = 0;
    if (n > MAX_CHUNKS || !(p.handles = calloc(n, sizeof(void *)))) {
        snprintf(why, whylen, "%zu chunks", n);
        return -1;
    }
    void *gpu = NULL;
    e = v.reserve(&gpu, p.padded, gran, NULL, 0);
    if (e != kHipSuccess || !gpu) {
        snprintf(why, whylen, "hipMemAddressReserve error %d", e);
        destroy(&v, &p, 0);
        return -1;
    }
    p.gpu = (uintptr_t)gpu;
    void *cpu = real_mmap_fn()(NULL, p.padded, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (cpu == MAP_FAILED) {
        snprintf(why, whylen, "CPU address reservation failed (errno %d)", errno);
        destroy(&v, &p, 0);
        return -1;
    }
    p.cpu = (uintptr_t)cpu;
    for (size_t i = 0; i < n; i++) {
        size_t off = i * chunk, size = chunk_at(&p, off);
        void *h = NULL;
        int fd = -1;
        prop.location.id = nodes[i % (size_t)nnodes];
        if ((e = v.create(&h, size, &prop, 0)) != kHipSuccess || !h) {
            snprintf(why, whylen, "hipMemCreate of chunk %zu/%zu on node %d error %d", i, n, prop.location.id, e);
            destroy(&v, &p, mapped);
            return -1;
        }
        p.handles[p.nhandles++] = h;
        if ((e = v.map((void *)(p.gpu + off), size, 0, h, 0)) != kHipSuccess) {
            snprintf(why, whylen, "hipMemMap of chunk %zu error %d", i, e);
            destroy(&v, &p, mapped);
            return -1;
        }
        mapped = off + size;
        if ((e = v.export_fd(&fd, h, 1, 0)) != kHipSuccess || fd < 0) {
            snprintf(why, whylen, "hipMemExportToShareableHandle of chunk %zu error %d", i, e);
            destroy(&v, &p, mapped);
            return -1;
        }
        void *c = real_mmap_fn()((void *)(p.cpu + off), size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
        int merr = errno;
        close(fd); /* the mapping holds its own reference */
        if (c != (void *)(p.cpu + off)) {
            snprintf(why, whylen, "mmap of chunk %zu's dmabuf failed (errno %d)", i, merr);
            destroy(&v, &p, mapped);
            return -1;
        }
    }
    int devices = 0;
    if ((e = v.device_count(&devices)) != kHipSuccess || devices <= 0) {
        snprintf(why, whylen, "hipGetDeviceCount error %d", e);
        destroy(&v, &p, mapped);
        return -1;
    }
    for (int d = 0; d < devices; d++) {
        HipAccessDesc acc = {{1, d}, 3};
        if ((e = v.set_access((void *)p.gpu, p.padded, &acc, 1)) != kHipSuccess) {
            snprintf(why, whylen, "hipMemSetAccess for device %d error %d", d, e);
            destroy(&v, &p, mapped);
            return -1;
        }
    }
    char where[96];
    if (interleave)
        snprintf(where, sizeof where, "interleaved over %d NUMA nodes", nnodes);
    else
        snprintf(where, sizeof where, "on NUMA node %d", node);
    snprintf(why, whylen, "%zu chunks of up to %zu MiB %s, %d devices, %zu bytes padded", n, chunk >> 20, where,
             devices, p.padded - len);
    *out = p;
    return 0;
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off)
{
    if (!in_hip && addr == NULL && fd == -1 && prot == (PROT_READ | PROT_WRITE) &&
        flags == (MAP_ANONYMOUS | MAP_PRIVATE) && len >= ((size_t)parse_num("UMBP_VMM_MIN_GB", 64) << 30)) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        Pool p;
        char why[256] = {0};
        in_hip = 1;
        int rc = npools < MAX_POOLS ? build(len, &p, why, sizeof why) : (snprintf(why, sizeof why, "pool table full"), -1);
        in_hip = 0;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
        if (rc == 0) {
            pthread_mutex_lock(&pools_lock);
            pools[npools++] = p;
            pthread_mutex_unlock(&pools_lock);
            note("pool of %zu MiB in HIP VMM host memory (%s): CPU %p, device %p, in %.1f s", len >> 20, why,
                 (void *)p.cpu, (void *)p.gpu, secs);
            return (void *)p.cpu;
        }
        note("HIP VMM pool of %zu MiB failed after %.1f s (%s); falling back to anonymous mmap", len >> 20, secs, why);
    }
    return real_mmap_fn()(addr, len, prot, flags, fd, off);
}

void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off_t off)
{
    return mmap(addr, len, prot, flags, fd, off);
}

int munmap(void *addr, size_t len)
{
    pthread_mutex_lock(&pools_lock);
    Pool *p = in_hip ? NULL : find_locked(addr, len);
    int whole = p && p->cpu == (uintptr_t)addr && p->len == len;
    Pool gone = {0};
    if (whole) {
        gone = *p;
        *p = pools[npools - 1];
        npools--;
    }
    pthread_mutex_unlock(&pools_lock);
    if (!p)
        return real_munmap_fn()(addr, len);
    if (!whole) {
        note("partial munmap of the VMM pool at %p (%zu bytes) ignored", addr, len);
        return 0;
    }
    Vmm v;
    load_vmm(&v);
    in_hip = 1;
    destroy(&v, &gone, gone.padded);
    in_hip = 0;
    note("pool at %p unmapped and released", addr);
    return 0;
}

int madvise(void *addr, size_t len, int advice)
{
    static int (*real_madvise)(void *, size_t, int);
    if (!real_madvise)
        real_madvise = (int (*)(void *, size_t, int))dlsym(RTLD_NEXT, "madvise");
    if (!in_hip && in_pool(addr, len))
        return 0;
    return real_madvise(addr, len, advice);
}

hipError_t hipHostRegister(void *ptr, size_t len, unsigned flags)
{
    static hipError_t (*real_register)(void *, size_t, unsigned);
    if (!real_register)
        real_register = (hipError_t(*)(void *, size_t, unsigned))dlsym(RTLD_NEXT, "hipHostRegister");
    if (in_pool(ptr, len)) {
        note("hipHostRegister(%p, %zu MiB, 0x%x) skipped: the pool is HIP VMM memory mapped on every device", ptr,
             len >> 20, flags);
        return kHipSuccess;
    }
    return real_register(ptr, len, flags);
}

hipError_t hipHostUnregister(void *ptr)
{
    static hipError_t (*real_unregister)(void *);
    if (!real_unregister)
        real_unregister = (hipError_t(*)(void *))dlsym(RTLD_NEXT, "hipHostUnregister");
    if (in_pool(ptr, 1))
        return kHipSuccess;
    return real_unregister(ptr);
}

hipError_t hipHostGetDevicePointer(void **dev, void *host, unsigned flags)
{
    static hipError_t (*real_get)(void **, void *, unsigned);
    if (!real_get)
        real_get = (hipError_t(*)(void **, void *, unsigned))dlsym(RTLD_NEXT, "hipHostGetDevicePointer");
    uintptr_t g = to_gpu(host, 1);
    if (g && dev) {
        *dev = (void *)g;
        return kHipSuccess;
    }
    return real_get(dev, host, flags);
}

/* Rewrite a copy with an end in a pool to use the pool's device address. Returns the kind to use. */
static int translate(void **dst, const void **src, size_t n, int kind)
{
    if (in_hip || kind == kMemcpyHostToHost)
        return kind;
    uintptr_t gd = to_gpu(*dst, n), gs = to_gpu(*src, n);
    if (!gd && !gs)
        return kind;
    if (gd)
        *dst = (void *)gd;
    if (gs)
        *src = (const void *)gs;
    return kMemcpyDeviceToDevice;
}

hipError_t hipMemcpy(void *dst, const void *src, size_t n, int kind)
{
    static hipError_t (*real_copy)(void *, const void *, size_t, int);
    if (!real_copy)
        real_copy = (hipError_t(*)(void *, const void *, size_t, int))dlsym(RTLD_NEXT, "hipMemcpy");
    kind = translate(&dst, &src, n, kind);
    return real_copy(dst, src, n, kind);
}

hipError_t hipMemcpyAsync(void *dst, const void *src, size_t n, int kind, void *stream)
{
    static hipError_t (*real_copy)(void *, const void *, size_t, int, void *);
    if (!real_copy)
        real_copy = (hipError_t(*)(void *, const void *, size_t, int, void *))dlsym(RTLD_NEXT, "hipMemcpyAsync");
    kind = translate(&dst, &src, n, kind);
    return real_copy(dst, src, n, kind, stream);
}
