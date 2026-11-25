/* LD_PRELOAD shim for the UMBP standalone server (MORI v1.2.3.post1): back its DRAM pool with driver-allocated pinned
 * host memory (hipHostMalloc) instead of anonymous pages registered with hipHostRegister.
 *
 * UMBP allocates the pool with one mmap(MAP_ANONYMOUS | MAP_PRIVATE) and registers it whole with
 * hipHostRegister(Mapped | Portable) (src/umbp/local/host_mem_allocator.cpp, local/tiers/host_registration.cpp).
 * Those are ordinary movable pages that KFD tracks as userptr: when compaction migrates any of them, KFD evicts every
 * GPU queue of the server and revalidates the whole 418 GiB before restoring them, which takes tens of seconds to
 * minutes, and prefill finds UMBP unavailable meanwhile (EXPERIMENT 4.6-4.9). hipHostMalloc memory is allocated by the
 * driver, is not on the kernel's movable LRU and is not userptr, so compaction never touches it. The GPU still maps it
 * (hipHostGetDevicePointer gives the gather kernel its alias) and hipMemcpy still takes the pinned path.
 *
 * Only an mmap that looks exactly like the pool is taken: addr NULL, fd -1, PROT_READ | PROT_WRITE,
 * flags MAP_ANONYMOUS | MAP_PRIVATE, length >= UMBP_PINNED_MIN_GB GiB (64). For that range hipHostRegister and
 * hipHostUnregister succeed without doing anything, madvise is a no-op (the pages are resident) and munmap frees with
 * hipHostFree. If hipHostMalloc fails the mmap goes through unchanged (the original behaviour), with a log line.
 *   UMBP_PINNED_COHERENT=1: hipHostMallocCoherent instead of the default hipHostMallocNonCoherent.
 *   gcc -shared -fPIC -O2 -o umbp_pinned_pool.so umbp_pinned_pool.c -ldl -lpthread
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <time.h>

typedef int hipError_t;
enum { kHipSuccess = 0 };
#define HOST_MALLOC_PORTABLE 0x1u
#define HOST_MALLOC_MAPPED 0x2u
#define HOST_MALLOC_COHERENT 0x40000000u
#define HOST_MALLOC_NONCOHERENT 0x80000000u
#define MAX_POOLS 16

static struct { uintptr_t base; size_t len; } pools[MAX_POOLS];
static int npools;
static pthread_mutex_t pools_lock = PTHREAD_MUTEX_INITIALIZER;
/* Set while this shim calls into HIP, whose own mmaps must go straight through. */
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
    fprintf(stderr, "%s [umbp_pinned_pool] ", head);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
}

static size_t parse_size(const char *env, size_t fallback)
{
    const char *s = getenv(env);
    if (!s || !*s)
        return fallback;
    size_t v = 0;
    for (; *s >= '0' && *s <= '9'; s++)
        v = v * 10 + (size_t)(*s - '0');
    return v;
}

/* Index of the pool that contains [addr, addr + len), or -1. Caller holds pools_lock. */
static int find_locked(const void *addr, size_t len)
{
    uintptr_t a = (uintptr_t)addr;
    for (int i = 0; i < npools; i++)
        if (a >= pools[i].base && len <= pools[i].len && a - pools[i].base <= pools[i].len - len)
            return i;
    return -1;
}

static int in_pool(const void *addr, size_t len)
{
    pthread_mutex_lock(&pools_lock);
    int i = find_locked(addr, len);
    pthread_mutex_unlock(&pools_lock);
    return i >= 0;
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off)
{
    static void *(*real_mmap)(void *, size_t, int, int, int, off_t);
    if (!real_mmap)
        real_mmap = (void *(*)(void *, size_t, int, int, int, off_t))dlsym(RTLD_NEXT, "mmap");
    if (!in_hip && addr == NULL && fd == -1 && prot == (PROT_READ | PROT_WRITE) &&
        flags == (MAP_ANONYMOUS | MAP_PRIVATE) && len >= (parse_size("UMBP_PINNED_MIN_GB", 64) << 30)) {
        hipError_t (*host_malloc)(void **, size_t, unsigned) =
            (hipError_t(*)(void **, size_t, unsigned))dlsym(RTLD_DEFAULT, "hipHostMalloc");
        unsigned hflags = HOST_MALLOC_PORTABLE | HOST_MALLOC_MAPPED |
                          (parse_size("UMBP_PINNED_COHERENT", 0) ? HOST_MALLOC_COHERENT : HOST_MALLOC_NONCOHERENT);
        void *p = NULL;
        hipError_t err = -1;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (host_malloc) {
            in_hip = 1;
            err = host_malloc(&p, len, hflags);
            in_hip = 0;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
        if (err == kHipSuccess && p) {
            pthread_mutex_lock(&pools_lock);
            int ok = npools < MAX_POOLS;
            if (ok) {
                pools[npools].base = (uintptr_t)p;
                pools[npools].len = len;
                npools++;
            }
            pthread_mutex_unlock(&pools_lock);
            if (ok) {
                note("pool of %zu MiB in hipHostMalloc(0x%x) pinned host memory at %p in %.1f s", len >> 20, hflags, p,
                     secs);
                return p;
            }
            void (*host_free)(void *) = (void (*)(void *))dlsym(RTLD_DEFAULT, "hipHostFree");
            if (host_free) {
                in_hip = 1;
                host_free(p);
                in_hip = 0;
            }
            note("pool table full; %zu MiB falls back to anonymous mmap", len >> 20);
        } else {
            note("hipHostMalloc(%zu MiB, 0x%x) failed (%s %d); falling back to anonymous mmap", len >> 20, hflags,
                 host_malloc ? "error" : "symbol missing", err);
        }
    }
    return real_mmap(addr, len, prot, flags, fd, off);
}

void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off_t off)
{
    return mmap(addr, len, prot, flags, fd, off);
}

int munmap(void *addr, size_t len)
{
    static int (*real_munmap)(void *, size_t);
    if (!real_munmap)
        real_munmap = (int (*)(void *, size_t))dlsym(RTLD_NEXT, "munmap");
    pthread_mutex_lock(&pools_lock);
    int i = in_hip ? -1 : find_locked(addr, len);
    int whole = i >= 0 && pools[i].base == (uintptr_t)addr;
    if (whole) {
        pools[i] = pools[npools - 1];
        npools--;
    }
    pthread_mutex_unlock(&pools_lock);
    if (i < 0)
        return real_munmap(addr, len);
    if (!whole) {
        note("partial munmap of the pinned pool at %p (%zu bytes) ignored", addr, len);
        return 0;
    }
    void (*host_free)(void *) = (void (*)(void *))dlsym(RTLD_DEFAULT, "hipHostFree");
    if (host_free) {
        in_hip = 1;
        host_free(addr);
        in_hip = 0;
    }
    note("pool at %p freed with hipHostFree", addr);
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
        note("hipHostRegister(%p, %zu MiB, 0x%x) skipped: the pool is pinned and mapped already", ptr, len >> 20,
             flags);
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
